"""Verify reusable artifacts in an isolated, contract-driven subprocess."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# This file is launched directly by L2Builder, so Python otherwise exposes only
# memory_pool/builder on sys.path and cannot import project-level helpers.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runtime_utils import (
    redact_local_paths,
    resolve_within,
    sanitized_subprocess_env,
    validate_storage_identifier,
)


def _limit_verification_process() -> None:
    """Apply conservative Unix resource limits to untrusted artifact checks."""
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_CPU, (20, 20))
        resource.setrlimit(resource.RLIMIT_FSIZE, (20 * 1024 * 1024,) * 2)
        resource.setrlimit(resource.RLIMIT_NOFILE, (256, 256))
        if hasattr(resource, "RLIMIT_AS"):
            resource.setrlimit(resource.RLIMIT_AS, (8 * 1024**3,) * 2)
    except (ImportError, OSError, ValueError):
        # Process-group isolation remains active on unsupported hosts.
        pass


def _load_card(json_path: Path) -> dict | None:
    try:
        with open(json_path, "r", encoding="utf-8") as stream:
            card = json.load(stream)
    except Exception as exc:
        print(f"Error reading JSON {json_path}: {exc}")
        return None
    if not isinstance(card, dict):
        print(f"Error: Model card must be a JSON object: {json_path}")
        return None
    return card


def _verification_request(
    card: dict, json_path: Path
) -> tuple[dict, Path] | None:
    artifact_dir = json_path.parent
    try:
        artifact_id = validate_storage_identifier(
            card.get("artifact_id"), "artifact_id"
        )
        category = validate_storage_identifier(
            card.get("category"), "category"
        )
        if artifact_dir.name not in {category, artifact_id}:
            # Local, not-yet-committed artifacts live in a node directory, so
            # enforce category folders only for global pool paths.
            if artifact_dir.parent.name == "l2_store":
                raise ValueError(
                    "Model-card category does not match its directory"
                )
        expected_code_name = f"{artifact_id}.py"
        if card.get("code_path") != expected_code_name:
            raise ValueError(
                "Model-card code_path must match '<artifact_id>.py'"
            )
        code_file = resolve_within(artifact_dir, expected_code_name)
    except ValueError as exc:
        print(f"Error: Invalid model card: {exc}")
        return None
    if not code_file.is_file():
        print(f"Error: Python code file not found at {code_file}")
        return None

    interface = card.get("interface")
    if not isinstance(interface, dict):
        print("Error: Model card must define interface as an object")
        return None
    entrypoint_signature = interface.get("entrypoint")
    if (
        not isinstance(entrypoint_signature, str)
        or not entrypoint_signature.strip()
    ):
        print("Error: Model card must define interface.entrypoint")
        return None
    entrypoint_name = entrypoint_signature.split("(", 1)[0].strip()
    if not re.fullmatch(r"[A-Za-z_]\w*", entrypoint_name):
        print(
            f"Error: Invalid entrypoint function name: {entrypoint_name!r}"
        )
        return None
    capabilities = card.get("capabilities", {})
    if not isinstance(capabilities, dict):
        print("Error: Model-card capabilities must be an object when provided")
        return None
    dependencies = card.get("dependencies", [])
    if not isinstance(dependencies, list):
        print("Error: Model-card dependencies must be a list")
        return None

    request = {
        "artifact_dir": str(artifact_dir),
        "module_name": code_file.stem,
        "entrypoint_name": entrypoint_name,
        "interface": interface,
        "capabilities": capabilities,
        "dependencies": dependencies,
    }
    return request, code_file


def _child_environment(verification_dir: Path) -> dict[str, str]:
    child_env = sanitized_subprocess_env()
    child_env.update(
        {
            "HOME": str(verification_dir),
            "XDG_CACHE_HOME": str(verification_dir / "cache"),
            "TMPDIR": str(verification_dir),
            "TMP": str(verification_dir),
            "TEMP": str(verification_dir),
            "MPLCONFIGDIR": str(verification_dir / "matplotlib"),
            "PYTHONDONTWRITEBYTECODE": "1",
            "HTTP_PROXY": "http://127.0.0.1:9",
            "HTTPS_PROXY": "http://127.0.0.1:9",
            "ALL_PROXY": "http://127.0.0.1:9",
            "http_proxy": "http://127.0.0.1:9",
            "https_proxy": "http://127.0.0.1:9",
            "all_proxy": "http://127.0.0.1:9",
            "NO_PROXY": "",
            "no_proxy": "",
        }
    )
    return child_env


def _run_isolated(
    request_path: Path, verification_dir: Path
) -> tuple[subprocess.CompletedProcess[str], str]:
    runtime_path = Path(__file__).with_name("verification_runtime.py")
    command = [sys.executable, str(runtime_path), str(request_path)]
    isolation_mode = "resource-limited-subprocess"
    sandbox_exec = shutil.which("sandbox-exec")
    if sandbox_exec:
        writable = str(verification_dir).replace('"', '\\"')
        profile = (
            "(version 1)\n"
            "(deny default)\n"
            "(allow process*)\n"
            "(allow file-read*)\n"
            f'(allow file-write* (subpath "{writable}"))\n'
            '(allow file-write* (literal "/dev/null"))\n'
            "(allow sysctl-read)\n"
        )
        command = [sandbox_exec, "-p", profile, *command]
        isolation_mode = "sandbox-exec"

    run_kwargs = {
        "capture_output": True,
        "text": True,
        "timeout": 60,
        "cwd": verification_dir,
        "env": _child_environment(verification_dir),
        "preexec_fn": (
            _limit_verification_process if os.name == "posix" else None
        ),
    }
    result = subprocess.run(command, **run_kwargs)
    if (
        sandbox_exec
        and result.returncode != 0
        and "sandbox_apply: Operation not permitted" in result.stderr
    ):
        # Managed/containerized hosts may expose sandbox-exec but prohibit
        # nested profiles. Retain process limits and the isolated writable
        # directory instead of rejecting every otherwise valid artifact.
        result = subprocess.run(
            [sys.executable, str(runtime_path), str(request_path)],
            **run_kwargs,
        )
        isolation_mode = "resource-limited-subprocess"
    return result, isolation_mode


def _runtime_metadata(output: str) -> tuple[str, str]:
    modality_match = re.search(
        r"^VERIFICATION_MODALITY=([a-z_]+)$", output, re.MULTILINE
    )
    contract_match = re.search(
        r"^VERIFICATION_CONTRACT=([a-z_]+)$", output, re.MULTILINE
    )
    return (
        modality_match.group(1) if modality_match else "unknown",
        contract_match.group(1) if contract_match else "unknown",
    )


def verify_artifact(json_path: Path) -> bool:
    """Verify one artifact against its declared or inferred data contract."""
    json_path = Path(json_path).resolve()
    print(f"Verifying artifact: {json_path.name}")
    card = _load_card(json_path)
    if card is None:
        return False
    prepared = _verification_request(card, json_path)
    if prepared is None:
        return False
    request, _ = prepared
    artifact_dir = json_path.parent
    verification_dir = Path(
        tempfile.mkdtemp(prefix="aibuildai_verify_")
    )
    isolation_mode = "resource-limited-subprocess"
    try:
        request_path = verification_dir / "verification_request.json"
        with open(request_path, "w", encoding="utf-8") as stream:
            json.dump(request, stream, indent=2)
            stream.write("\n")
        result, isolation_mode = _run_isolated(
            request_path, verification_dir
        )
        success = result.returncode == 0 and "SUCCESS" in result.stdout
        log_message = redact_local_paths(
            result.stdout + "\n" + result.stderr,
            artifact_dir,
            verification_dir,
        )
    except subprocess.TimeoutExpired:
        success = False
        log_message = (
            "Verification process exceeded its isolated safety budget"
        )
    except Exception as exc:
        success = False
        log_message = f"Process launch failed: {exc}"
    finally:
        shutil.rmtree(verification_dir, ignore_errors=True)

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    if success:
        modality, contract_source = _runtime_metadata(log_message)
        print(
            f"Artifact {card['artifact_id']} passed {modality} sandbox "
            "verification!"
        )
        print(
            "WARNING: Verified against synthetic contract data only, not real "
            "task data."
        )
        card["verified"] = True
        legacy_neural_mixed = (
            modality == "tabular"
            and "input_contract" not in request["interface"]
            and any(
                str(item).lower().startswith(
                    (
                        "torch",
                        "pytorch",
                        "tensorflow",
                        "keras",
                        "jax",
                        "flax",
                    )
                )
                for item in request["dependencies"]
            )
        )
        card["verification_level"] = (
            "mixed-missing-contract-mock-data"
            if legacy_neural_mixed
            else f"{modality}-contract-synthetic-data"
        )
        card["verification_contract_source"] = contract_source
        card["verification_isolation"] = isolation_mode
        card["verification_log"] = (
            f"Passed subprocess verification at {timestamp}.\n"
            f"Output:\n{log_message.strip()}"
        )
    else:
        print(f"Artifact {card['artifact_id']} failed sandbox verification!")
        print(f"Log:\n{log_message}")
        card["verified"] = False
        card["verification_log"] = (
            f"Verification failed at {timestamp}.\n"
            f"Log:\n{redact_local_paths(log_message, artifact_dir)}"
        )
    try:
        with open(json_path, "w", encoding="utf-8") as stream:
            json.dump(card, stream, indent=2)
            stream.write("\n")
    except Exception as exc:
        print(f"Failed to update JSON metadata: {exc}")
        return False
    return success


def main() -> None:
    if len(sys.argv) > 1:
        ok = verify_artifact(Path(sys.argv[1]))
        raise SystemExit(0 if ok else 1)

    root_dir = Path(__file__).resolve().parent.parent / "l2_store"
    if not root_dir.exists():
        print(f"Store directory does not exist: {root_dir}")
        raise SystemExit(1)
    all_json_files = list(root_dir.glob("**/*.json"))
    print(f"Found {len(all_json_files)} artifacts to verify.")
    success_count = sum(verify_artifact(path) for path in all_json_files)
    print(
        f"\nVerification summary: {success_count}/"
        f"{len(all_json_files)} verified successfully."
    )
    raise SystemExit(0 if success_count == len(all_json_files) else 1)


if __name__ == "__main__":
    main()
