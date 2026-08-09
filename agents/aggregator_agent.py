"""Final-output materialization for the selected implementation."""

from __future__ import annotations

import shutil
from pathlib import Path


class AggregatorAgent:
    """Copy the strongest node's native deliverable into the run root."""

    def materialize(self, source: str | Path, run_root: str | Path) -> Path:
        source_path = Path(source)
        destination_root = Path(run_root)
        destination_root.mkdir(parents=True, exist_ok=True)

        if source_path.is_file() and source_path.suffix.lower() == ".csv":
            destination = destination_root / "submission.csv"
            shutil.copy2(source_path, destination)
            return destination

        final_root = destination_root / "final_output"
        final_root.mkdir(parents=True, exist_ok=True)
        if source_path.is_file():
            destination = final_root / source_path.name
            shutil.copy2(source_path, destination)
            return destination
        if source_path.is_dir():
            destination = final_root / source_path.name
            shutil.copytree(source_path, destination, dirs_exist_ok=True)
            return destination
        raise FileNotFoundError(f"Implementation output is missing: {source_path}")
