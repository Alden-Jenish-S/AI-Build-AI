import json
import subprocess
import sys
from pathlib import Path
from typing import List
from runtime_utils import sanitized_subprocess_env


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
    def __init__(self, venv_python_path: str = "./.venv/bin/python"):
        import sys
        resolved_path = str(Path(venv_python_path).resolve())
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
            
        self._log(f"SetupAgent: Checking/Installing dependencies: {list(dependencies)}")
        
        to_install = []
        for dep in dependencies:
            # Basic parsing of requirement specifier to get package name
            # e.g., 'catboost==1.2.3' -> 'catboost'
            pkg_name = dep
            for op in ['==', '>=', '<=', '>', '<', '~=']:
                if op in dep:
                    pkg_name = dep.split(op)[0].strip()
                    break
            
            # Check if package is already installed
            cmd_check = [self.venv_python, "-m", "pip", "show", pkg_name]
            try:
                res_check = subprocess.run(
                    cmd_check,
                    capture_output=True,
                    text=True,
                    env=sanitized_subprocess_env(),
                )
                if res_check.returncode == 0:
                    self._log(f"SetupAgent: Package '{pkg_name}' is already installed. Skipping.")
                else:
                    to_install.append(dep)
            except Exception:
                to_install.append(dep)

        if not to_install:
            self._log("SetupAgent: All dependencies are already satisfied.")
            return

        self._log(f"SetupAgent: Executing pip install for missing dependencies: {to_install}")
        cmd = [self.venv_python, "-m", "pip", "install"] + to_install
        
        try:
            res = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                env=sanitized_subprocess_env(),
            )
            self._log("SetupAgent: Package installation successful!")
            self._log(res.stdout)
        except subprocess.CalledProcessError as e:
            self._log("SetupAgent ERROR: Failed to install dependencies.")
            self._log(e.stderr)
            raise e
