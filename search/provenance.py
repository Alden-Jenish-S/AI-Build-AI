"""Append-only artifact provenance DAG, separate from execution ancestry."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class ArtifactRecord:
    artifact_id: str
    artifact_type: str
    produced_by_node_id: str
    dataset_fingerprint: str | None = None
    code_hash: str | None = None
    parameter_hash: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ProvenanceGraph:
    """Persist multi-source artifact relationships without changing the tree."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def _load(self) -> dict:
        if not self.path.is_file():
            return {"version": 1, "artifacts": {}, "edges": []}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"version": 1, "artifacts": {}, "edges": []}
        if not isinstance(payload, dict):
            return {"version": 1, "artifacts": {}, "edges": []}
        payload.setdefault("version", 1)
        payload.setdefault("artifacts", {})
        payload.setdefault("edges", [])
        return payload

    def record(
        self,
        artifact: ArtifactRecord,
        *,
        sources: Iterable[tuple[str, str]] = (),
    ) -> None:
        payload = self._load()
        artifact_payload = asdict(artifact)
        artifact_payload["recorded_at_utc"] = datetime.now(
            timezone.utc
        ).isoformat()
        payload["artifacts"][artifact.artifact_id] = artifact_payload
        existing = {
            (
                edge.get("source_artifact_id"),
                edge.get("target_artifact_id"),
                edge.get("relation"),
            )
            for edge in payload["edges"]
        }
        for source_artifact_id, relation in sources:
            key = (source_artifact_id, artifact.artifact_id, relation)
            if key in existing:
                continue
            payload["edges"].append(
                {
                    "source_artifact_id": source_artifact_id,
                    "target_artifact_id": artifact.artifact_id,
                    "relation": relation,
                }
            )
            existing.add(key)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(payload, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
