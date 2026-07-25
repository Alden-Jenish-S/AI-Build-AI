"""Audio task adapter."""

from __future__ import annotations

import wave
from collections import Counter
from pathlib import Path

from core.contracts import TaskSpec
from .common import resolve_task_source
from .media_base import ManifestMediaAdapter


class AudioAdapter(ManifestMediaAdapter):
    name = "audio"
    extensions = frozenset(
        {".aac", ".flac", ".m4a", ".mp3", ".ogg", ".wav"}
    )
    path_field_candidates = (
        "audio_path",
        "recording_path",
        "file_path",
        "filename",
        "path",
    )

    def profile(
        self, task_dir: Path, task_spec: TaskSpec
    ) -> dict[str, object]:
        bundle = self.build_bundle(task_dir, task_spec)
        sample_rates: Counter[str] = Counter()
        channels: Counter[str] = Counter()
        durations: list[float] = []
        unreadable: list[str] = []
        for record in bundle.train_records[:64]:
            reference = next(iter(record.inputs.values()))
            try:
                path = resolve_task_source(task_dir, str(reference))
            except FileNotFoundError:
                path = Path(task_dir) / str(reference)
            if path.suffix.lower() != ".wav":
                continue
            try:
                with wave.open(str(path), "rb") as stream:
                    rate = stream.getframerate()
                    sample_rates[str(rate)] += 1
                    channels[str(stream.getnchannels())] += 1
                    durations.append(
                        stream.getnframes() / max(rate, 1)
                    )
            except (OSError, wave.Error):
                unreadable.append(record.sample_id)
        return {
            **bundle.to_index_dict(),
            "sampled_sample_rates": dict(sample_rates),
            "sampled_channels": dict(channels),
            "sampled_duration_seconds": {
                "min": min(durations) if durations else None,
                "max": max(durations) if durations else None,
                "mean": (
                    sum(durations) / len(durations)
                    if durations
                    else None
                ),
            },
            "sampled_unreadable_ids": unreadable,
            "target_distribution": dict(
                Counter(
                    str(record.target)
                    for record in bundle.train_records
                )
            ),
        }
