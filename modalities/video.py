"""Video task adapter with metadata-only discovery."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from core.contracts import TaskSpec
from .common import resolve_task_source
from .media_base import ManifestMediaAdapter


class VideoAdapter(ManifestMediaAdapter):
    name = "video"
    extensions = frozenset(
        {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".webm"}
    )
    path_field_candidates = (
        "video_path",
        "clip_path",
        "file_path",
        "filename",
        "path",
    )

    def profile(
        self, task_dir: Path, task_spec: TaskSpec
    ) -> dict[str, object]:
        bundle = self.build_bundle(task_dir, task_spec)
        extensions: Counter[str] = Counter()
        sizes = []
        missing = []
        for record in bundle.train_records[:128]:
            reference = next(iter(record.inputs.values()))
            try:
                path = resolve_task_source(task_dir, str(reference))
            except FileNotFoundError:
                path = Path(task_dir) / str(reference)
            if not path.is_file():
                missing.append(record.sample_id)
                continue
            extensions[path.suffix.lower()] += 1
            sizes.append(path.stat().st_size)
        return {
            **bundle.to_index_dict(),
            "sampled_extensions": dict(extensions),
            "sampled_size_bytes": {
                "min": min(sizes) if sizes else None,
                "max": max(sizes) if sizes else None,
                "mean": sum(sizes) / len(sizes) if sizes else None,
            },
            "sampled_missing_ids": missing,
            "target_distribution": dict(
                Counter(
                    str(record.target)
                    for record in bundle.train_records
                )
            ),
            "decode_policy": "lazy_deterministic_clips",
        }
