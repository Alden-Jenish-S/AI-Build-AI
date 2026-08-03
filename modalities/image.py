"""Image task adapter."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from core.contracts import TaskSpec
from core.runtime_contracts import DatasetBundle
from .common import resolve_task_source
from .media_base import ManifestMediaAdapter


class ImageAdapter(ManifestMediaAdapter):
    name = "image"
    extensions = frozenset(
        {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
    )
    path_field_candidates = (
        "image_path",
        "image",
        "file_path",
        "filename",
        "path",
    )

    def profile(
        self, task_dir: Path, task_spec: TaskSpec
    ) -> dict[str, object]:
        bundle = self.build_bundle(task_dir, task_spec)
        sizes: Counter[str] = Counter()
        modes: Counter[str] = Counter()
        corrupt: list[str] = []
        try:
            from PIL import Image

            for record in bundle.train_records[:64]:
                reference = next(iter(record.inputs.values()))
                try:
                    path = resolve_task_source(task_dir, str(reference))
                    with Image.open(path) as image:
                        sizes[f"{image.width}x{image.height}"] += 1
                        modes[str(image.mode)] += 1
                        image.verify()
                except Exception:
                    corrupt.append(record.sample_id)
        except ImportError:
            modes["metadata_unavailable_without_pillow"] = 1
        profile = {
            **bundle.to_index_dict(),
            "sampled_dimensions": dict(sizes),
            "sampled_color_modes": dict(modes),
            "sampled_corrupt_ids": corrupt,
        }
        if task_spec.target and str(task_spec.target.type or "").endswith("_path"):
            profile["structured_target_references"] = len(bundle.train_records)
            profile["target_storage"] = task_spec.target.type
        else:
            profile["target_distribution"] = dict(
                Counter(str(record.target) for record in bundle.train_records)
            )
        return profile
