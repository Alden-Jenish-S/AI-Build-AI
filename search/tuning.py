"""Persistent, compatibility-aware fine-tuning history and trial reuse."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class TrialRecord:
    trial_id: str
    task_name: str
    model_family: str
    search_space_version: str
    parameters: dict[str, Any]
    score: float
    normalized_improvement: float
    metric_name: str
    metric_direction: str
    fidelity: str
    status: str
    dataset_fingerprint: str | None = None
    library_version: str | None = None
    uncertainty: float | None = None
    elapsed_seconds: float | None = None
    trial_count: int = 1
    dataset_metafeatures: dict[str, Any] = field(default_factory=dict)


class TuningKnowledgeBase:
    """JSONL-backed global store with strict semantic compatibility checks."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def records(self) -> list[dict]:
        if not self.path.is_file():
            return []
        records: list[dict] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                records.append(payload)
        return records

    def append(self, record: TrialRecord) -> None:
        if not math.isfinite(float(record.score)):
            raise ValueError("tuning score must be finite")
        payload = asdict(record)
        payload["parameter_hash"] = _stable_hash(record.parameters)
        payload["recorded_at_utc"] = datetime.now(timezone.utc).isoformat()
        existing_ids = {
            item.get("trial_id") for item in self.records()
        }
        if record.trial_id in existing_ids:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, default=str) + "\n")

    def compatible(
        self,
        *,
        model_family: str,
        search_space_version: str,
        metric_name: str,
        metric_direction: str,
        task_name: str,
        dataset_fingerprint: str | None = None,
        limit: int = 8,
    ) -> list[dict]:
        candidates = []
        for record in self.records():
            if (
                record.get("status") != "completed"
                or record.get("model_family") != model_family
                or record.get("search_space_version") != search_space_version
                or record.get("metric_name") != metric_name
                or record.get("metric_direction") != metric_direction
                or not isinstance(record.get("parameters"), dict)
            ):
                continue
            same_dataset = bool(
                dataset_fingerprint
                and record.get("dataset_fingerprint") == dataset_fingerprint
            )
            same_task = record.get("task_name") == task_name
            compatibility_rank = 2 if same_dataset else (1 if same_task else 0)
            candidates.append(
                (
                    compatibility_rank,
                    float(record.get("normalized_improvement", -math.inf)),
                    record,
                )
            )
        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [item[2] for item in candidates[: max(0, int(limit))]]


class TuningCoordinator:
    """Construct knowledge-guided tuning context and record completed studies."""

    def __init__(self, knowledge_base: TuningKnowledgeBase):
        self.knowledge_base = knowledge_base

    @staticmethod
    def search_space_version(tunable_parameters: Iterable[Any]) -> str:
        return _stable_hash(sorted(str(item) for item in tunable_parameters))

    def build_context(
        self,
        *,
        task_name: str,
        model_family: str,
        tunable_parameters: list[Any],
        metric_name: str,
        metric_direction: str,
        dataset_fingerprint: str | None,
    ) -> dict:
        version = self.search_space_version(tunable_parameters)
        reused = self.knowledge_base.compatible(
            model_family=model_family,
            search_space_version=version,
            metric_name=metric_name,
            metric_direction=metric_direction,
            task_name=task_name,
            dataset_fingerprint=dataset_fingerprint,
        )
        compact = [
            {
                "task_name": record.get("task_name"),
                "fidelity": record.get("fidelity"),
                "parameters": record.get("parameters"),
                "normalized_improvement": record.get(
                    "normalized_improvement"
                ),
                "uncertainty": record.get("uncertainty"),
                "same_dataset": bool(
                    dataset_fingerprint
                    and record.get("dataset_fingerprint")
                    == dataset_fingerprint
                ),
            }
            for record in reused
        ]
        return {
            "search_space_version": version,
            "global_trial_reuse": True,
            "reused_trials": compact,
            "suggested_initial_parameters": [
                item["parameters"] for item in compact if item.get("parameters")
            ],
            "avoid_duplicate_parameter_hashes": [
                _stable_hash(item["parameters"])
                for item in compact
                if item.get("parameters")
            ],
        }

    def record(
        self,
        *,
        trial_id: str,
        task_name: str,
        model_family: str,
        search_space_version: str,
        parameters: dict,
        score: float,
        normalized_improvement: float,
        metric_name: str,
        metric_direction: str,
        fidelity: str,
        dataset_fingerprint: str | None,
        uncertainty: float | None,
        elapsed_seconds: float | None,
        trial_count: int,
    ) -> None:
        self.knowledge_base.append(
            TrialRecord(
                trial_id=trial_id,
                task_name=task_name,
                model_family=model_family,
                search_space_version=search_space_version,
                parameters=parameters,
                score=score,
                normalized_improvement=normalized_improvement,
                metric_name=metric_name,
                metric_direction=metric_direction,
                fidelity=fidelity,
                status="completed",
                dataset_fingerprint=dataset_fingerprint,
                uncertainty=uncertainty,
                elapsed_seconds=elapsed_seconds,
                trial_count=max(1, int(trial_count or 1)),
            )
        )
