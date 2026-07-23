import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version
from runtime_utils import (
    absolute_path_without_symlink_resolution,
    sanitized_subprocess_env,
)

_PYTORCH_CUDA_126_INDEX = "https://download.pytorch.org/whl/cu126"


def _validate_requirement(requirement: object) -> str:
    """Accept registry package requirements, but reject URLs, paths, and pip flags."""
    if not isinstance(requirement, str) or not requirement.strip():
        raise ValueError(f"Invalid dependency requirement: {requirement!r}")
    requirement = requirement.strip()
    try:
        from packaging.requirements import Requirement
        parsed = Requirement(requirement)
    except (ImportError, ValueError) as exc:
        raise ValueError(f"Invalid dependency requirement: {requirement!r}") from exc
    if parsed.url is not None:
        raise ValueError(f"Direct URL/path dependencies are not allowed: {requirement!r}")
    return requirement

class SetupAgent:
    def __init__(self, venv_python_path: str | None = None):
        import sys
        if venv_python_path is None:
            self.venv_python = sys.executable
            self.pip_path = str(Path(self.venv_python).parent / "pip")
            self.log_file = None
            return
        resolved_path = str(
            absolute_path_without_symlink_resolution(venv_python_path)
        )
        # Check if the resolved venv python is fully functional
        use_fallback = True
        if Path(resolved_path).exists():
            try:
                res = subprocess.run([resolved_path, "-c", "import sys; print('ok')"], capture_output=True, text=True, timeout=5)
                if res.returncode == 0 and "ok" in res.stdout:
                    use_fallback = False
            except Exception:
                pass
                
        if use_fallback:
            print(f"SetupAgent WARNING: Specified python path '{resolved_path}' is invalid or non-functional. Falling back to active running interpreter: {sys.executable}")
            self.venv_python = sys.executable
        else:
            self.venv_python = resolved_path
            
        self.pip_path = str(Path(self.venv_python).parent / "pip")
        self.log_file = None

    def set_task_run_dir(self, run_dir: Path):
        """Set task run folder to write dependency logs."""
        self.log_file = run_dir / "dependency_log.txt"

    def _log(self, message: str):
        print(message)
        if self.log_file:
            try:
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(message + "\n")
            except Exception:
                pass

    def install_dependencies(self, artifact_cards: List[dict]):
        """
        Reads the union of the 'dependencies' fields from all imported Model Cards
        for the run, checks if they are already installed, and installs missing ones.
        """
        dependencies = set()
        for card in artifact_cards:
            if card.get("verified") is not True:
                raise ValueError(
                    f"Refusing to install dependencies for unverified artifact "
                    f"{card.get('artifact_id', '<unknown>')!r}"
                )
            deps = card.get("dependencies", [])
            if not isinstance(deps, list):
                raise ValueError("Model-card dependencies must be a list")
            for dep in deps:
                dependencies.add(_validate_requirement(dep))
                
        if not dependencies:
            self._log("SetupAgent: No dependencies to install.")
            return
            
        self._log(f"SetupAgent: Selected interpreter: {self.venv_python}")
        self._log(f"SetupAgent: Checking/Installing dependencies: {list(dependencies)}")
        
        active_dependencies = set()
        to_install = []
        for dep in dependencies:
            requirement = Requirement(dep)
            if requirement.marker is not None and not requirement.marker.evaluate():
                self._log(
                    f"SetupAgent: Skipping inactive environment marker for {dep!r}."
                )
                continue
            active_dependencies.add(dep)
            pkg_name = requirement.name

            # `pip show` alone only establishes presence. Parse its Version field
            # so stale environments cannot silently bypass an exact project pin.
            cmd_check = [self.venv_python, "-m", "pip", "show", pkg_name]
            try:
                res_check = subprocess.run(
                    cmd_check,
                    capture_output=True,
                    text=True,
                    timeout=30,
                    env=sanitized_subprocess_env(),
                )
                if res_check.returncode == 0:
                    version_text = next(
                        (
                            line.split(":", 1)[1].strip()
                            for line in res_check.stdout.splitlines()
                            if line.lower().startswith("version:")
                        ),
                        "",
                    )
                    try:
                        version_matches = bool(version_text) and (
                            not requirement.specifier
                            or requirement.specifier.contains(
                                Version(version_text), prereleases=True
                            )
                        )
                    except InvalidVersion:
                        version_matches = False
                    if version_matches:
                        self._log(
                            f"SetupAgent: Package '{pkg_name}' {version_text} "
                            "satisfies the resolved requirement. Skipping."
                        )
                    else:
                        self._log(
                            f"SetupAgent: Package '{pkg_name}' version "
                            f"{version_text or '<unknown>'} does not satisfy {dep!r}."
                        )
                        to_install.append(dep)
                else:
                    to_install.append(dep)
            except Exception:
                to_install.append(dep)

        if not to_install:
            self._verify_dependency_imports(active_dependencies)
            self._log("SetupAgent: All dependencies are installed and importable.")
            return

        free_bytes = shutil.disk_usage(Path(self.venv_python).resolve().parent).free
        minimum_free_bytes = 1024 ** 3
        if free_bytes < minimum_free_bytes:
            message = (
                "Dependency installation skipped because the selected environment "
                f"has only {free_bytes / (1024 ** 3):.2f} GiB free; at least "
                "1.00 GiB is required to avoid filling the worker disk."
            )
            self._log("SetupAgent ERROR: " + message)
            raise OSError(28, message)

        self._log(f"SetupAgent: Executing pip install for missing dependencies: {to_install}")
        cmd = [
            self.venv_python,
            "-m",
            "pip",
            "install",
            "--no-cache-dir",
        ]
        if any(
            canonicalize_name(Requirement(dep).name) == "torch"
            and "+cu126" in dep.lower()
            for dep in to_install
        ):
            cmd.extend(["--extra-index-url", _PYTORCH_CUDA_126_INDEX])
        cmd.extend(to_install)
        
        try:
            res = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=1800,
                env=sanitized_subprocess_env(),
            )
        except subprocess.CalledProcessError as e:
            self._log("SetupAgent ERROR: Failed to install dependencies.")
            self._log(e.stderr)
            raise e
        except subprocess.TimeoutExpired as e:
            self._log("SetupAgent ERROR: Dependency installation timed out after 1800s.")
            raise e
        self._log("SetupAgent: Package installation successful!")
        self._log(res.stdout)
        self._verify_dependency_imports(active_dependencies)

    def _verify_dependency_imports(self, dependencies) -> None:
        """Detect installed-but-broken dependency environments before execution."""
        import_names = {
            "scikit-learn": "sklearn",
            "pytorch-tabnet": "pytorch_tabnet",
            "pytorch-lightning": "pytorch_lightning",
            "opencv-python": "cv2",
            "imbalanced-learn": "imblearn",
        }
        failures = []
        resolved_versions = []
        for dependency in sorted(dependencies):
            package = Requirement(dependency).name
            module = import_names.get(package.lower(), package.replace("-", "_"))
            cuda_arch_check = ""
            if (
                canonicalize_name(package) == "torch"
                and "+cu126" in dependency.lower()
            ):
                cuda_arch_check = (
                    "; assert torch.version.cuda == '12.6', "
                    "f'Expected a CUDA 12.6 PyTorch wheel, got CUDA "
                    "{torch.version.cuda!r}'"
                    "; flags=torch._C._cuda_getArchFlags()"
                    "; arches=set(flags.split())"
                    "; assert 'sm_61' in arches, "
                    "f'PyTorch wheel lacks required TITAN Xp architecture sm_61: "
                    "{sorted(arches)}'"
                )
            try:
                result = subprocess.run(
                    [
                        self.venv_python,
                        "-c",
                        (
                            f"import {module}; import importlib.metadata as metadata; "
                            f"print(metadata.version({package!r})){cuda_arch_check}"
                        ),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=120,
                    env=sanitized_subprocess_env(),
                )
            except subprocess.TimeoutExpired:
                failures.append(
                    f"{package}: import validation timed out after 120 seconds"
                )
                continue
            if result.returncode != 0:
                failures.append(f"{package}: {result.stderr[-1200:]}")
            else:
                resolved_versions.append(
                    f"{package}=={result.stdout.strip() or '<unknown>'}"
                )
        if failures:
            message = "Installed dependency failed import validation:\n" + "\n".join(failures)
            self._log("SetupAgent ERROR: " + message)
            raise RuntimeError(message)
        if resolved_versions:
            self._log(
                "SetupAgent: Validated imports in selected interpreter: "
                + ", ".join(resolved_versions)
            )

    def install_allowlisted_dependencies(
        self,
        artifact_cards: List[dict],
        requirements_file: Path,
    ) -> None:
        """Install unverified-artifact dependencies only from the project allowlist.

        Generated model cards cannot choose versions or arbitrary packages here. A
        requested distribution must already exist in the human-controlled
        requirements file, and the exact requirement from that file is installed.
        """
        requested = []
        for card in artifact_cards:
            dependencies = card.get("dependencies", [])
            if not isinstance(dependencies, list):
                raise ValueError("Model-card dependencies must be a list")
            requested.extend(_validate_requirement(dep) for dep in dependencies)
        if not requested:
            return

        requirements_file = Path(requirements_file)
        if not requirements_file.is_file():
            raise FileNotFoundError(
                f"Project dependency allowlist does not exist: {requirements_file}"
            )
        allowlist = {}
        for raw_line in requirements_file.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith(("-", "--")):
                # Project-controlled pip source options are consumed by ordinary
                # `pip install -r`; generated model cards still cannot provide
                # options because `_validate_requirement` rejects them.
                continue
            validated = _validate_requirement(line)
            parsed = Requirement(validated)
            if parsed.marker is not None and not parsed.marker.evaluate():
                continue
            allowlist[canonicalize_name(parsed.name)] = validated

        approved = set()
        for dependency in requested:
            name = canonicalize_name(Requirement(dependency).name)
            if name not in allowlist:
                raise ValueError(
                    f"Generated dependency {dependency!r} is not allowlisted in "
                    f"{requirements_file.name}"
                )
            approved.add(allowlist[name])

        if not approved:
            return
        # Reuse the verified-card installation path after replacing generated
        # requirements with the exact project-controlled specifications.
        self.install_dependencies(
            [{"verified": True, "dependencies": sorted(approved)}]
        )
