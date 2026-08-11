"""Small runtime helpers for isolated generated implementations."""

from __future__ import annotations

import logging
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
from typing import IO, Mapping, Optional, Sequence


_PATH_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SENSITIVE_ENV_PARTS = (
    "API_KEY", "ACCESS_KEY", "AUTH", "BEARER", "CREDENTIAL", "PASSWORD",
    "PRIVATE_KEY", "SECRET", "SESSION_TOKEN",
)
_SENSITIVE_ENV_EXACT = {
    "AWS_PROFILE", "GOOGLE_APPLICATION_CREDENTIALS", "KAGGLE_CONFIG_DIR", "NETRC",
}

# Bounded in-memory capture for supervised child output; a runaway generator
# must not grow the parent's memory without limit. Console streaming continues
# after capture truncation.
_OUTPUT_CAPTURE_LIMIT_CHARS = 1_000_000


@dataclass(frozen=True)
class SupervisedProcessResult:
    args: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    elapsed_seconds: float
    stalled: bool
    hard_limit_reached: bool
    termination_reason: str | None
    progress_events: int
    last_progress_source: str
    last_progress_age_seconds: float


def _parse_cpu_time(value: str) -> float:
    raw = value.strip()
    days = 0
    if "-" in raw:
        day_text, raw = raw.split("-", 1)
        days = int(day_text)
    parts = raw.split(":")
    if len(parts) == 3:
        hours, minutes, seconds = parts
    elif len(parts) == 2:
        hours, minutes, seconds = "0", parts[0], parts[1]
    else:
        hours, minutes, seconds = "0", "0", parts[0]
    return days * 86400 + int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def _process_group_cpu_seconds(process_group_id: int) -> float | None:
    if os.name != "posix":
        return None
    try:
        result = subprocess.run(
            ["ps", "-o", "time=", "-g", str(process_group_id)],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode != 0:
            return None
        values = [_parse_cpu_time(line) for line in result.stdout.splitlines() if line.strip()]
        return sum(values) if values else None
    except Exception:
        return None


def _deliverable_signature(root: Path | None) -> tuple[tuple[str, int, int], ...]:
    if root is None or not root.exists():
        return ()
    observed: list[tuple[str, int, int]] = []
    try:
        for path in root.rglob("*"):
            if not path.is_file() or "input" in path.relative_to(root).parts:
                continue
            try:
                stat = path.stat()
                observed.append((path.relative_to(root).as_posix(), stat.st_size, stat.st_mtime_ns))
            except OSError:
                continue
            if len(observed) >= 4000:
                break
    except OSError:
        pass
    return tuple(observed)


def _terminate_process_tree(process: subprocess.Popen[bytes], grace_seconds: float) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
        process.wait(timeout=grace_seconds)
        return
    except (ProcessLookupError, subprocess.TimeoutExpired):
        pass
    if process.poll() is None:
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        except ProcessLookupError:
            pass


def run_supervised_process(
    command: Sequence[str | os.PathLike[str]],
    *,
    cwd: Path,
    env: Mapping[str, str] | None = None,
    stall_seconds: float | None = 1800.0,
    hard_limit_seconds: float | None = None,
    activity_root: Path | None = None,
    stdout_stream: IO[str] | None = None,
    stderr_stream: IO[str] | None = None,
    poll_seconds: float = 0.2,
    resource_sample_seconds: float = 5.0,
    terminate_grace_seconds: float = 5.0,
    label: str = "Process",
) -> SupervisedProcessResult:
    """Run a child while renewing its lease on output, files, or CPU work."""
    if not command:
        raise ValueError("command cannot be empty")
    for name, value in (("stall_seconds", stall_seconds), ("hard_limit_seconds", hard_limit_seconds)):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(float(value)) or value <= 0
        ):
            raise ValueError(f"{name} must be None or a positive finite number")

    normalized = tuple(str(item) for item in command)
    child_env = dict(env) if env is not None else None
    if child_env is not None:
        child_env.setdefault("PYTHONUNBUFFERED", "1")
    process = subprocess.Popen(
        normalized,
        cwd=Path(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=child_env,
        start_new_session=(os.name == "posix"),
        bufsize=0,
    )
    if process.stdout is None or process.stderr is None:
        raise RuntimeError("child output streams were not created")

    messages: queue.Queue[tuple[str, bytes]] = queue.Queue()

    def pump(name: str, stream: IO[bytes]) -> None:
        try:
            while chunk := os.read(stream.fileno(), 65536):
                messages.put((name, chunk))
        except (OSError, ValueError):
            pass

    threads = [
        threading.Thread(target=pump, args=("stdout", process.stdout), daemon=True),
        threading.Thread(target=pump, args=("stderr", process.stderr), daemon=True),
    ]
    for thread in threads:
        thread.start()

    stdout_parts: list[str] = []
    stderr_parts: list[str] = []
    stdout_chars = 0
    stderr_chars = 0
    stdout_truncated = False
    stderr_truncated = False
    started = time.monotonic()
    last_progress = started
    last_source = "process_started"
    events = 0
    last_sample = started
    cpu = _process_group_cpu_seconds(process.pid)
    signature = _deliverable_signature(activity_root or Path(cwd))
    termination_reason: str | None = None

    def collect() -> None:
        nonlocal last_progress, last_source, events
        nonlocal stdout_chars, stderr_chars, stdout_truncated, stderr_truncated
        found = False
        while True:
            try:
                name, raw = messages.get_nowait()
            except queue.Empty:
                break
            text = raw.decode("utf-8", errors="replace")
            if name == "stdout":
                if not stdout_truncated:
                    keep = max(0, _OUTPUT_CAPTURE_LIMIT_CHARS - stdout_chars)
                    if keep < len(text):
                        stdout_parts.append(text[:keep])
                        stdout_parts.append(
                            "\n[stdout capture truncated; live console streaming continues]\n"
                        )
                        stdout_truncated = True
                    else:
                        stdout_parts.append(text)
                stdout_chars += len(text)
                if stdout_stream:
                    stdout_stream.write(text)
                    stdout_stream.flush()
            else:
                if not stderr_truncated:
                    keep = max(0, _OUTPUT_CAPTURE_LIMIT_CHARS - stderr_chars)
                    if keep < len(text):
                        stderr_parts.append(text[:keep])
                        stderr_parts.append(
                            "\n[stderr capture truncated; live console streaming continues]\n"
                        )
                        stderr_truncated = True
                    else:
                        stderr_parts.append(text)
                stderr_chars += len(text)
                if stderr_stream:
                    stderr_stream.write(text)
                    stderr_stream.flush()
            found = True
        if found:
            last_progress, last_source = time.monotonic(), "process_output"
            events += 1

    try:
        while process.poll() is None:
            collect()
            now = time.monotonic()
            if now - last_sample >= resource_sample_seconds:
                current_signature = _deliverable_signature(activity_root or Path(cwd))
                if current_signature != signature:
                    signature = current_signature
                    last_progress, last_source = now, "deliverable_change"
                    events += 1
                current_cpu = _process_group_cpu_seconds(process.pid)
                if current_cpu is not None and cpu is not None and current_cpu > cpu:
                    last_progress, last_source = now, "process_cpu"
                    events += 1
                cpu = current_cpu
                last_sample = now
            if hard_limit_seconds is not None and now - started >= hard_limit_seconds:
                termination_reason = "explicit_hard_limit"
                _terminate_process_tree(process, terminate_grace_seconds)
                break
            if stall_seconds is not None and now - last_progress >= stall_seconds:
                termination_reason = "progress_stalled"
                stderr_parts.append(f"\n{label}: no output or observed work for {stall_seconds:.0f}s.\n")
                _terminate_process_tree(process, terminate_grace_seconds)
                break
            time.sleep(poll_seconds)
    except BaseException:
        _terminate_process_tree(process, terminate_grace_seconds)
        raise
    finally:
        for thread in threads:
            thread.join(timeout=0.5)
        collect()
        process.stdout.close()
        process.stderr.close()

    finished = time.monotonic()
    return SupervisedProcessResult(
        args=normalized,
        returncode=process.poll() if process.poll() is not None else -1,
        stdout="".join(stdout_parts),
        stderr="".join(stderr_parts),
        elapsed_seconds=finished - started,
        stalled=termination_reason == "progress_stalled",
        hard_limit_reached=termination_reason == "explicit_hard_limit",
        termination_reason=termination_reason,
        progress_events=events,
        last_progress_source=last_source,
        last_progress_age_seconds=max(0.0, finished - last_progress),
    )


def absolute_path_without_symlink_resolution(path: str | Path) -> Path:
    return Path(os.path.abspath(os.path.expanduser(str(path))))


def validate_path_component(value: object, field_name: str) -> str:
    if not isinstance(value, str) or value in {".", ".."} or not _PATH_COMPONENT.fullmatch(value):
        raise ValueError(f"Invalid {field_name}: {value!r}")
    return value


def task_data_files(task_dir: Path) -> list[Path]:
    root = Path(task_dir)
    if not root.is_dir():
        return []
    files: list[Path] = []
    for current, directory_names, file_names in os.walk(
        root, followlinks=False, onerror=lambda _error: None
    ):
        directory_names.sort()
        for name in sorted(file_names):
            path = Path(current) / name
            try:
                if path.is_file() and path.name != ".DS_Store":
                    files.append(path)
            except OSError:
                continue
    return files


def task_dir_snapshot(task_dir: Path) -> dict[str, tuple[int, int]]:
    """Capture ``(size, mtime_ns)`` per task file for integrity verification."""
    root = Path(task_dir)
    snapshot: dict[str, tuple[int, int]] = {}
    if not root.is_dir():
        return snapshot
    for current, directory_names, file_names in os.walk(
        root, followlinks=False, onerror=lambda _error: None
    ):
        directory_names.sort()
        for name in sorted(file_names):
            if name == ".DS_Store":
                continue
            path = Path(current) / name
            try:
                stat = path.stat()
            except OSError:
                continue
            snapshot[path.relative_to(root).as_posix()] = (
                stat.st_size,
                stat.st_mtime_ns,
            )
    return snapshot


def verify_task_dir_unchanged(
    task_dir: Path, snapshot: dict[str, tuple[int, int]]
) -> list[str]:
    """Return the task files whose size or mtime changed since the snapshot."""
    changed: list[str] = []
    root = Path(task_dir)
    current = task_dir_snapshot(root)
    for relative, (size, mtime_ns) in snapshot.items():
        observed = current.get(relative)
        if observed is None or observed != (size, mtime_ns):
            changed.append(relative)
    for relative in sorted(set(current) - set(snapshot)):
        changed.append(relative)
    return changed


def _default_copy_limit_bytes() -> int | None:
    configured = os.getenv("AIBUILDAI_INPUT_COPY_LIMIT_BYTES", "").strip()
    if configured.isdigit():
        return int(configured)
    return 64 * 1024 * 1024


def expose_task_data(
    task_dir: Path,
    run_dir: Path,
    *,
    allowed_paths: Sequence[str] | None = None,
    copy_files: bool = False,
    copy_limit_bytes: int | None = None,
) -> list[Path]:
    """Expose approved task files through file-level links in ``run_dir/input``.

    Without ``allowed_paths`` this retains the historical all-files behavior.
    Council-guided runs provide an explicit allowlist so held-out labels and
    grading artifacts are not made visible to generated programs.

    Read-only copies are the default whenever the approved input set fits under
    ``copy_limit_bytes`` (default 64 MiB) so a faulty script cannot modify
    task-owned files through a writable link. Larger collections fall back to
    links to avoid multiplying benchmark storage; callers that use links should
    verify the task directory with ``task_dir_snapshot`` / ``verify_task_dir_unchanged``
    around child execution.
    """
    task_root = Path(task_dir).resolve()
    destination = Path(run_dir) / "input"
    destination.mkdir(parents=True, exist_ok=True)
    linked: list[Path] = []
    if allowed_paths is None:
        sources = task_data_files(task_root)
    else:
        sources = []
        for raw in sorted(set(str(item) for item in allowed_paths)):
            relative = Path(raw)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Invalid allowed task path: {raw!r}")
            source = task_root / relative
            try:
                resolved = source.resolve()
            except OSError:
                continue
            if resolved != task_root and task_root not in resolved.parents:
                raise ValueError(f"Allowed task path escapes task root: {raw!r}")
            if resolved.is_file():
                sources.append(resolved)
    if copy_limit_bytes is None:
        copy_limit_bytes = _default_copy_limit_bytes()
    should_copy = bool(copy_files)
    if not should_copy and copy_limit_bytes is not None and sources:
        total_bytes = 0
        for source in sources:
            try:
                total_bytes += source.stat().st_size
            except OSError:
                continue
            if total_bytes > copy_limit_bytes:
                break
        should_copy = total_bytes <= copy_limit_bytes
    for source in sources:
        target = destination / source.relative_to(task_root)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"Node input path is already occupied: {target}")
        
        if should_copy:
            shutil.copy2(source, target)
            try:
                target.chmod(0o444)
            except OSError:
                pass
        else:
            try:
                os.symlink(source, target)
            except (OSError, NotImplementedError) as exc:
                logging.warning(
                    "Symlink failed for %s -> %s: %s. Falling back to copy.",
                    source, target, exc
                )
                shutil.copy2(source, target)
                try:
                    target.chmod(0o444)
                except OSError:
                    pass
        linked.append(target)
    return linked


def sanitized_subprocess_env(base_env: Mapping[str, str] | None = None) -> dict[str, str]:
    source = os.environ if base_env is None else base_env
    clean: dict[str, str] = {}
    for name, value in source.items():
        upper = name.upper()
        if upper in _SENSITIVE_ENV_EXACT or any(part in upper for part in _SENSITIVE_ENV_PARTS):
            continue
        clean[name] = value
    return clean
