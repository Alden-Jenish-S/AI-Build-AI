"""Shared validation and subprocess-safety helpers."""

from __future__ import annotations

import math
import os
import queue
import re
import signal
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Iterable, Mapping, Optional, Sequence


_STORAGE_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_PATH_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SENSITIVE_ENV_PARTS = (
    "API_KEY",
    "ACCESS_KEY",
    "AUTH",
    "BEARER",
    "CREDENTIAL",
    "PASSWORD",
    "PRIVATE_KEY",
    "SECRET",
    "SESSION_TOKEN",
)
_SENSITIVE_ENV_EXACT = {
    "AWS_PROFILE",
    "GOOGLE_APPLICATION_CREDENTIALS",
    "KAGGLE_CONFIG_DIR",
    "NETRC",
}
_ACCELERATORS = {"cpu", "cuda", "mps"}
_TASK_DATA_SUFFIXES = {
    ".csv",
    ".feather",
    ".json",
    ".jsonl",
    ".npy",
    ".npz",
    ".parquet",
    ".pickle",
    ".pkl",
    ".tsv",
    ".txt",
}
_TASK_DATA_EXCLUSIONS = {
    "dataset_analysis_report.txt",
    "initial_algorithm.py",
    "initial_dataloader.py",
    "submission.csv",
    "task_config.json",
}
_TASK_TYPES = {
    "classification",
    "regression",
    "supervised",
    "unsupervised_clustering",
}


@dataclass(frozen=True)
class SupervisedProcessResult:
    """Captured result and progress-lease diagnostics for a child process."""

    args: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    elapsed_seconds: float
    stalled: bool
    hard_limit_reached: bool
    termination_reason: Optional[str]
    progress_events: int
    last_progress_source: str
    last_progress_age_seconds: float


def _parse_process_cpu_time(value: str) -> float:
    """Parse the elapsed CPU format emitted by POSIX ``ps``."""
    raw = value.strip()
    if not raw:
        raise ValueError("empty process CPU time")
    days = 0
    if "-" in raw:
        day_text, raw = raw.split("-", 1)
        days = int(day_text)
    parts = raw.split(":")
    if len(parts) == 3:
        hours, minutes, seconds = parts
    elif len(parts) == 2:
        hours = "0"
        minutes, seconds = parts
    else:
        hours = "0"
        minutes = "0"
        seconds = parts[0]
    return (
        days * 86400
        + int(hours) * 3600
        + int(minutes) * 60
        + float(seconds)
    )


def _process_group_cpu_seconds(process_group_id: int) -> Optional[float]:
    """Return aggregate CPU consumed by a POSIX process group when available."""
    if os.name != "posix":
        return None
    proc_root = Path("/proc")
    if proc_root.is_dir():
        try:
            clock_ticks = float(os.sysconf("SC_CLK_TCK"))
            total_ticks = 0
            matched = False
            for process_dir in proc_root.iterdir():
                if not process_dir.name.isdigit():
                    continue
                try:
                    stat_text = (process_dir / "stat").read_text(
                        encoding="utf-8"
                    )
                    fields = stat_text[stat_text.rfind(")") + 2 :].split()
                    process_group = int(fields[2])
                    if process_group != process_group_id:
                        continue
                    total_ticks += int(fields[11]) + int(fields[12])
                    matched = True
                except (OSError, ValueError, IndexError):
                    continue
            if matched and clock_ticks > 0:
                return total_ticks / clock_ticks
        except (OSError, ValueError):
            pass
    if shutil.which("ps") is None:
        return None
    try:
        probe = subprocess.run(
            ["ps", "-axo", "pgid=,time="],
            capture_output=True,
            text=True,
            timeout=5,
            env=sanitized_subprocess_env(),
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if probe.returncode != 0:
        return None
    total = 0.0
    matched = False
    for raw_line in probe.stdout.splitlines():
        fields = raw_line.strip().split(None, 1)
        if len(fields) != 2:
            continue
        try:
            pgid = int(fields[0])
        except ValueError:
            continue
        if pgid != process_group_id:
            continue
        try:
            total += _parse_process_cpu_time(fields[1])
        except ValueError:
            continue
        matched = True
    return total if matched else None


def _activity_signature(root: Optional[Path]) -> tuple[tuple[str, int, int], ...]:
    """Snapshot child-owned output files without traversing linked task input."""
    if root is None:
        return ()
    root = Path(root)
    if not root.is_dir():
        return ()
    entries: list[tuple[str, int, int]] = []
    try:
        candidates = root.rglob("*")
        for candidate in candidates:
            try:
                relative = candidate.relative_to(root)
                if relative.parts and relative.parts[0] == "input":
                    continue
                if candidate.is_symlink() or not candidate.is_file():
                    continue
                stat = candidate.stat()
            except (OSError, ValueError):
                continue
            entries.append((str(relative), stat.st_mtime_ns, stat.st_size))
    except OSError:
        return ()
    return tuple(sorted(entries))


def _terminate_process_tree(
    process: subprocess.Popen[bytes], grace_seconds: float
) -> None:
    """Terminate the supervised process group, escalating automatically."""
    if process.poll() is not None:
        return
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
    else:
        process.terminate()
    try:
        process.wait(timeout=grace_seconds)
        return
    except subprocess.TimeoutExpired:
        pass
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
    else:
        process.kill()
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        return


def run_supervised_process(
    command: Sequence[str | os.PathLike[str]],
    *,
    cwd: Path,
    env: Optional[Mapping[str, str]] = None,
    stall_seconds: Optional[float] = 1800.0,
    hard_limit_seconds: Optional[float] = None,
    activity_root: Optional[Path] = None,
    stdout_stream: Optional[IO[str]] = None,
    stderr_stream: Optional[IO[str]] = None,
    poll_seconds: float = 0.1,
    resource_sample_seconds: float = 5.0,
    terminate_grace_seconds: float = 5.0,
    label: str = "Process",
) -> SupervisedProcessResult:
    """Run a child under an automatically renewed progress lease.

    There is no total runtime ceiling unless an explicit ``hard_limit_seconds``
    is supplied by a focused direct caller. Normal workflow jobs may run for any
    duration. Their lease is renewed by stdout/stderr, output-file changes, or
    increasing CPU consumption anywhere in the child process group.
    """
    if not command:
        raise ValueError("command cannot be empty")
    for field_name, value in (
        ("stall_seconds", stall_seconds),
        ("hard_limit_seconds", hard_limit_seconds),
    ):
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or value <= 0
        ):
            raise ValueError(f"{field_name} must be None or a positive number")
    if poll_seconds <= 0 or resource_sample_seconds <= 0:
        raise ValueError("poll and resource sampling intervals must be positive")
    if terminate_grace_seconds <= 0:
        raise ValueError("terminate_grace_seconds must be positive")

    normalized_command = tuple(str(item) for item in command)
    working_directory = Path(cwd)
    watched_root = (
        Path(activity_root)
        if activity_root is not None
        else working_directory
    )
    child_env = dict(env) if env is not None else None
    if child_env is not None:
        child_env.setdefault("PYTHONUNBUFFERED", "1")
    process = subprocess.Popen(
        normalized_command,
        cwd=working_directory,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=child_env,
        start_new_session=(os.name == "posix"),
        bufsize=0,
    )
    if process.stdout is None or process.stderr is None:
        raise RuntimeError("supervised process pipes were not created")

    output_queue: queue.Queue[tuple[str, bytes]] = queue.Queue()

    def pump(name: str, pipe: IO[bytes]) -> None:
        try:
            while True:
                chunk = os.read(pipe.fileno(), 65536)
                if not chunk:
                    return
                output_queue.put((name, chunk))
        except (OSError, ValueError):
            return

    readers = [
        threading.Thread(
            target=pump,
            args=("stdout", process.stdout),
            daemon=True,
            name=f"{label}-stdout",
        ),
        threading.Thread(
            target=pump,
            args=("stderr", process.stderr),
            daemon=True,
            name=f"{label}-stderr",
        ),
    ]
    for reader in readers:
        reader.start()

    stdout_parts: list[str] = []
    stderr_parts: list[str] = []
    started = time.monotonic()
    last_progress = started
    last_progress_source = "process_started"
    progress_events = 0
    last_resource_sample = started
    cpu_seconds = _process_group_cpu_seconds(process.pid)
    activity_signature = _activity_signature(watched_root)
    termination_reason: Optional[str] = None

    def record_output() -> bool:
        nonlocal progress_events, last_progress, last_progress_source
        observed = False
        while True:
            try:
                stream_name, raw_chunk = output_queue.get_nowait()
            except queue.Empty:
                break
            text = raw_chunk.decode("utf-8", errors="replace")
            if stream_name == "stdout":
                stdout_parts.append(text)
                if stdout_stream is not None:
                    stdout_stream.write(text)
                    stdout_stream.flush()
            else:
                stderr_parts.append(text)
                if stderr_stream is not None:
                    stderr_stream.write(text)
                    stderr_stream.flush()
            observed = True
        if observed:
            last_progress = time.monotonic()
            last_progress_source = "process_output"
            progress_events += 1
        return observed

    try:
        while True:
            record_output()
            returncode = process.poll()
            now = time.monotonic()
            if returncode is not None:
                break

            if now - last_resource_sample >= resource_sample_seconds:
                current_signature = _activity_signature(watched_root)
                if current_signature != activity_signature:
                    activity_signature = current_signature
                    last_progress = now
                    last_progress_source = "output_artifact"
                    progress_events += 1

                current_cpu_seconds = _process_group_cpu_seconds(process.pid)
                if (
                    current_cpu_seconds is not None
                    and cpu_seconds is not None
                    and current_cpu_seconds > cpu_seconds
                ):
                    last_progress = now
                    last_progress_source = "process_cpu"
                    progress_events += 1
                cpu_seconds = current_cpu_seconds
                last_resource_sample = now

            if (
                hard_limit_seconds is not None
                and now - started >= hard_limit_seconds
            ):
                termination_reason = "explicit_hard_limit"
                _terminate_process_tree(process, terminate_grace_seconds)
                break

            if stall_seconds is not None and now - last_progress >= stall_seconds:
                termination_reason = "progress_stalled"
                message = (
                    f"\n{label}: no output, artifact changes, or process activity "
                    f"for {stall_seconds:.1f}s; automatically recycling the "
                    "stalled process.\n"
                )
                stderr_parts.append(message)
                if stderr_stream is not None:
                    stderr_stream.write(message)
                    stderr_stream.flush()
                _terminate_process_tree(process, terminate_grace_seconds)
                break

            time.sleep(poll_seconds)
    except BaseException:
        _terminate_process_tree(process, terminate_grace_seconds)
        raise
    finally:
        for reader in readers:
            reader.join(timeout=0.5)
        record_output()
        for pipe in (process.stdout, process.stderr):
            try:
                pipe.close()
            except OSError:
                pass
        for reader in readers:
            reader.join(timeout=0.5)
        record_output()

    returncode = process.poll()
    if returncode is None:
        returncode = -1
    finished = time.monotonic()
    return SupervisedProcessResult(
        args=normalized_command,
        returncode=returncode,
        stdout="".join(stdout_parts),
        stderr="".join(stderr_parts),
        elapsed_seconds=finished - started,
        stalled=termination_reason == "progress_stalled",
        hard_limit_reached=termination_reason == "explicit_hard_limit",
        termination_reason=termination_reason,
        progress_events=progress_events,
        last_progress_source=last_progress_source,
        last_progress_age_seconds=max(0.0, finished - last_progress),
    )


def absolute_path_without_symlink_resolution(path: str | Path) -> Path:
    """Return an absolute path while preserving virtualenv executable symlinks."""
    return Path(os.path.abspath(os.path.expanduser(str(path))))


def validate_storage_identifier(value: object, field_name: str) -> str:
    """Validate an L1 category or L2 artifact identifier."""
    if not isinstance(value, str) or not _STORAGE_IDENTIFIER.fullmatch(value):
        raise ValueError(
            f"{field_name} must match {_STORAGE_IDENTIFIER.pattern!r}; got {value!r}"
        )
    return value


def validate_path_component(value: object, field_name: str) -> str:
    """Validate a user-controlled directory name without allowing traversal."""
    if (
        not isinstance(value, str)
        or value in {".", ".."}
        or not _PATH_COMPONENT.fullmatch(value)
    ):
        raise ValueError(f"Invalid {field_name}: {value!r}")
    return value


def resolve_within(base_dir: Path, *parts: str) -> Path:
    """Resolve a descendant path and reject attempts to escape ``base_dir``."""
    base = Path(base_dir).resolve()
    candidate = base.joinpath(*parts).resolve()
    if candidate != base and base not in candidate.parents:
        raise ValueError(f"Path escapes allowed directory {base}: {candidate}")
    return candidate


def infer_task_type(
    description: str, configured_task_type: object = None
) -> str:
    """Resolve the broad task family used during dataset discovery."""
    if configured_task_type is not None:
        normalized = str(configured_task_type).strip().lower().replace("-", "_")
        aliases = {
            "clustering": "unsupervised_clustering",
            "unsupervised": "unsupervised_clustering",
        }
        normalized = aliases.get(normalized, normalized)
        if normalized not in _TASK_TYPES:
            raise ValueError(
                "task_type must be classification, regression, supervised, or "
                f"unsupervised_clustering; got {configured_task_type!r}"
            )
        return normalized

    lowered = str(description or "").lower()
    clustering_markers = (
        "adjusted rand index",
        "cluster label",
        "clustering",
        "unlabeled",
        "unsupervised",
    )
    if any(marker in lowered for marker in clustering_markers):
        return "unsupervised_clustering"
    if any(
        marker in lowered
        for marker in (
            "regression",
            "rmse",
            "mae",
            "root mean squared",
            "mean absolute error",
        )
    ):
        return "regression"
    if any(
        marker in lowered
        for marker in ("classification", "accuracy", "roc auc", "area under")
    ):
        return "classification"
    return "supervised"


def task_data_files(task_dir: Path) -> list[Path]:
    """Return immutable task-owned data files, never generated run artifacts."""
    task_root = Path(task_dir)
    source_root = (
        task_root / "input"
        if (task_root / "input").is_dir()
        else task_root
    )
    if not source_root.is_dir():
        return []
    return [
        path
        for path in sorted(source_root.iterdir())
        if (
            path.is_file()
            and path.suffix.lower() in _TASK_DATA_SUFFIXES
            and path.name not in _TASK_DATA_EXCLUSIONS
        )
    ]


def expose_task_data(task_dir: Path, run_dir: Path) -> list[Path]:
    """Expose task-owned input data in a run directory without copying it.

    Dataloaders execute from the run directory and expect ``./input``. The run
    always owns that directory and contains file-level links to immutable task
    data. Linking a task's entire ``input`` directory is forbidden because a
    generated cache or output below ``./input`` would then be created inside
    ``tasks/``. A failed file link is an error: silently copying a dataset per
    implementation node would make run storage grow with the search budget.
    """
    task_dir = Path(task_dir).resolve()
    run_dir = Path(run_dir)
    destination = run_dir / "input"
    source_root = (
        task_dir / "input"
        if (task_dir / "input").is_dir()
        else task_dir
    )

    def link(source: Path, target: Path) -> None:
        try:
            os.symlink(str(source.resolve()), str(target))
        except OSError as exc:
            raise RuntimeError(
                f"Could not link task data {source} into {target}; refusing to "
                "copy the dataset into the run directory."
            ) from exc

    destination.mkdir(parents=True, exist_ok=True)
    linked_files = []
    for source_file in sorted(source_root.iterdir()):
        if (
            not source_file.is_file()
            or source_file.suffix.lower() not in _TASK_DATA_SUFFIXES
            or source_file.name in _TASK_DATA_EXCLUSIONS
        ):
            continue
        target = destination / source_file.name
        if target.exists() or target.is_symlink():
            if target.is_symlink() and target.resolve() == source_file.resolve():
                linked_files.append(target)
                continue
            raise FileExistsError(
                f"Run input path already exists and is not a task-data link: {target}"
            )
        link(source_file, target)
        linked_files.append(target)
    return linked_files


def sanitized_subprocess_env(
    base_env: Optional[Mapping[str, str]] = None,
) -> dict[str, str]:
    """Copy an environment while removing credentials from untrusted children."""
    source = os.environ if base_env is None else base_env
    clean: dict[str, str] = {}
    for name, value in source.items():
        upper_name = name.upper()
        if upper_name in _SENSITIVE_ENV_EXACT:
            continue
        if any(part in upper_name for part in _SENSITIVE_ENV_PARTS):
            continue
        clean[name] = value
    return clean


def detect_available_accelerators(python_executable: str) -> set[str]:
    """Detect accelerators usable by the run without making GPU availability mandatory."""
    available = {"cpu"}
    torch_cuda_incompatible = False
    try:
        probe = subprocess.run(
            [
                python_executable,
                "-c",
                (
                    "import torch\n"
                    "if torch.cuda.is_available():\n"
                    "    major, minor = torch.cuda.get_device_capability(0)\n"
                    "    device_arch = f'sm_{major}{minor}'\n"
                    "    compiled_arches = set(torch.cuda.get_arch_list())\n"
                    "    if compiled_arches and device_arch not in compiled_arches:\n"
                    "        print(f'incompatible:{device_arch}')\n"
                    "    else:\n"
                    "        torch.ones(1, device='cuda').add_(1)\n"
                    "        print('cuda')\n"
                    "elif (hasattr(torch.backends, 'mps') and "
                    "torch.backends.mps.is_available()):\n"
                    "    print('mps')\n"
                    "else:\n"
                    "    print('none')\n"
                ),
            ],
            capture_output=True,
            text=True,
            timeout=10,
            env=sanitized_subprocess_env(),
        )
        detected = probe.stdout.strip().lower()
        if probe.returncode == 0 and detected in {"cuda", "mps"}:
            available.add(detected)
        elif probe.returncode == 0 and detected.startswith("incompatible:"):
            torch_cuda_incompatible = True
    except Exception:
        pass

    # CatBoost/LightGBM can use CUDA even when PyTorch is absent. nvidia-smi is
    # therefore a useful secondary hardware probe, unless CUDA was explicitly
    # hidden or the selected PyTorch build explicitly rejects this GPU.
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if (
        "cuda" not in available
        and not torch_cuda_incompatible
        and cuda_visible not in {"", "-1"}
        and shutil.which("nvidia-smi")
    ):
        try:
            probe = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=10,
                env=sanitized_subprocess_env(),
            )
            if probe.returncode == 0 and probe.stdout.strip():
                available.add("cuda")
        except Exception:
            pass
    return available


def select_preferred_accelerator(
    available: Iterable[str], preference: str = "auto"
) -> str:
    """Select CUDA, then MPS, then CPU while respecting an optional preference."""
    normalized = {str(item).lower() for item in available} & _ACCELERATORS
    normalized.add("cpu")
    requested = str(preference or "auto").lower()
    if requested not in {"auto", "gpu", "cpu", "cuda", "mps"}:
        raise ValueError(
            "preferred_accelerator must be one of auto, gpu, cpu, cuda, or mps"
        )
    if requested == "cpu":
        return "cpu"
    if requested in {"cuda", "mps"} and requested in normalized:
        return requested
    for candidate in ("cuda", "mps", "cpu"):
        if candidate in normalized:
            return candidate
    return "cpu"


def accelerator_subprocess_env(
    accelerator: str,
    base_env: Optional[Mapping[str, str]] = None,
) -> dict[str, str]:
    """Return a sanitized child environment carrying the selected accelerator contract."""
    selected = str(accelerator).lower()
    if selected not in _ACCELERATORS:
        raise ValueError(f"Unsupported selected accelerator: {accelerator!r}")
    clean = sanitized_subprocess_env(base_env)
    clean["AIBUILDAI_ACCELERATOR"] = selected
    clean["AIBUILDAI_ACTUAL_ACCELERATOR"] = selected
    clean["AIBUILDAI_PREFER_GPU"] = "1" if selected in {"cuda", "mps"} else "0"
    if selected == "cuda":
        clean.setdefault("AIBUILDAI_CUDA_DEVICES", "0")
        clean.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    return clean


def redact_local_paths(text: str, *additional_paths: Path) -> str:
    """Remove machine-specific absolute paths from persisted diagnostics."""
    redacted = text
    replacements = {Path.home(): "<HOME>"}
    replacements.update({Path(path).resolve(): "<WORKDIR>" for path in additional_paths})
    for path, label in sorted(
        replacements.items(), key=lambda item: len(str(item[0])), reverse=True
    ):
        redacted = redacted.replace(str(path), label)
    redacted = re.sub(
        r"/(?:private/)?var/folders/[^\s'\"]+|/tmp/[^\s'\"]+",
        "<TEMP_PATH>",
        redacted,
    )
    return redacted
