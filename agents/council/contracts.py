"""Stable data contracts shared by the research council and search tree."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def content_hash(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _clean_string(value: object, *, fallback: str = "") -> str:
    text = " ".join(str(value or "").split())
    return text or fallback


@dataclass(frozen=True)
class EvaluationProtocol:
    """One immutable validation specification used by every search node."""

    metric: str
    direction: str
    mode: str
    split_strategy: str
    folds: int
    seed: int
    leakage_unit: str
    rationale: str
    required_result_fields: tuple[str, ...] = (
        "score",
        "metric",
        "direction",
        "evaluation_protocol_hash",
        "fold_scores",
        "validation_sample_count",
    )

    def __post_init__(self) -> None:
        if self.direction not in {"maximize", "minimize"}:
            raise ValueError("evaluation direction must be maximize or minimize")
        if self.mode not in {"cross_validation", "holdout", "task_native"}:
            raise ValueError("unsupported evaluation mode")
        if isinstance(self.folds, bool) or int(self.folds) < 1:
            raise ValueError("folds must be a positive integer")
        if self.mode == "cross_validation" and int(self.folds) < 2:
            raise ValueError("cross-validation requires at least two folds")

    @property
    def protocol_hash(self) -> str:
        return content_hash(asdict(self))

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "protocol_hash": self.protocol_hash}

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, object] | None,
        *,
        metric: str,
        direction: str,
    ) -> "EvaluationProtocol":
        value = dict(raw or {})
        mode = str(value.get("mode") or "cross_validation").strip().lower()
        if mode not in {"cross_validation", "holdout", "task_native"}:
            mode = "cross_validation"
        default_folds = 5 if mode == "cross_validation" else 1
        try:
            folds = int(str(value.get("folds", default_folds)))
        except (TypeError, ValueError):
            folds = default_folds
        folds = max(2, min(10, folds)) if mode == "cross_validation" else 1
        try:
            seed = int(str(value.get("seed", 42)))
        except (TypeError, ValueError):
            seed = 42
        return cls(
            # Metric semantics come from the task contract, never from an LLM
            # synthesis field that could silently make node scores incomparable.
            metric=_clean_string(metric, fallback="task score"),
            direction=direction,
            mode=mode,
            split_strategy=_clean_string(
                value.get("split_strategy"),
                fallback="deterministic stratified folds"
                if mode == "cross_validation"
                else "deterministic holdout"
                if mode == "holdout"
                else "task-native deterministic evaluation",
            ),
            folds=folds,
            seed=seed,
            leakage_unit=_clean_string(
                value.get("leakage_unit"), fallback="sample"
            ),
            rationale=_clean_string(
                value.get("rationale"),
                fallback="Use one reproducible protocol for comparable tree scores.",
            ),
        )


@dataclass
class CouncilBrief:
    """Validated, persisted output of one pre-search council session."""

    task_name: str
    status: str
    problem_fingerprint: dict[str, Any]
    allowed_input_paths: tuple[str, ...]
    prohibited_inputs: tuple[dict[str, str], ...]
    evaluation_protocol: EvaluationProtocol
    evidence: list[dict[str, Any]] = field(default_factory=list)
    sources: list[dict[str, Any]] = field(default_factory=list)
    member_reports: list[dict[str, Any]] = field(default_factory=list)
    hypotheses: list[dict[str, Any]] = field(default_factory=list)
    selected_portfolio: list[dict[str, Any]] = field(default_factory=list)
    rejected_hypotheses: list[dict[str, Any]] = field(default_factory=list)
    unresolved_questions: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    recommended_root_count: int = 1
    created_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def __post_init__(self) -> None:
        if self.status not in {"completed", "degraded"}:
            raise ValueError("council status must be completed or degraded")
        allowed = tuple(dict.fromkeys(str(path) for path in self.allowed_input_paths))
        object.__setattr__(self, "allowed_input_paths", allowed)
        self.recommended_root_count = max(
            1,
            min(
                max(1, len(self.selected_portfolio)),
                int(self.recommended_root_count or 1),
            ),
        )

    @property
    def brief_hash(self) -> str:
        payload = self.to_dict(include_hash=False)
        return content_hash(payload)

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload = {
            "schema_version": 1,
            "task_name": self.task_name,
            "status": self.status,
            "created_at_utc": self.created_at_utc,
            "problem_fingerprint": self.problem_fingerprint,
            "allowed_input_paths": list(self.allowed_input_paths),
            "prohibited_inputs": list(self.prohibited_inputs),
            "evaluation_protocol": self.evaluation_protocol.to_dict(),
            "evidence": self.evidence,
            "sources": self.sources,
            "member_reports": self.member_reports,
            "hypotheses": self.hypotheses,
            "selected_portfolio": self.selected_portfolio,
            "rejected_hypotheses": self.rejected_hypotheses,
            "unresolved_questions": self.unresolved_questions,
            "warnings": self.warnings,
            "recommended_root_count": self.recommended_root_count,
        }
        if include_hash:
            payload["brief_hash"] = content_hash(payload)
        return payload

    def prompt_context(self, max_chars: int = 18000) -> str:
        """Compact council evidence for downstream planning and implementation."""
        compact = {
            "brief_hash": self.brief_hash,
            "status": self.status,
            "problem_fingerprint": self.problem_fingerprint,
            "evaluation_protocol": self.evaluation_protocol.to_dict(),
            "prohibited_inputs": list(self.prohibited_inputs),
            "verified_evidence": self.evidence[:40],
            "research_sources": [
                {
                    "source_id": item.get("source_id"),
                    "title": item.get("title"),
                    "url": item.get("url"),
                    "claim": item.get("claim"),
                }
                for item in self.sources[:30]
            ],
            "selected_portfolio": self.selected_portfolio,
            "unresolved_questions": self.unresolved_questions,
        }
        text = json.dumps(compact, indent=2, ensure_ascii=False, default=str)
        return text if len(text) <= max_chars else text[:max_chars] + "\n[Council context truncated]"

    def write(self, council_dir: Path) -> tuple[Path, Path]:
        council_dir = Path(council_dir)
        council_dir.mkdir(parents=True, exist_ok=True)
        json_path = council_dir / "council_brief.json"
        temporary = json_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False, default=str)
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(json_path)

        report_path = council_dir / "council_report.md"
        lines = [
            "# ML Research Council",
            "",
            f"Status: {self.status}",
            f"Council brief hash: `{self.brief_hash}`",
            f"Independent member reports: {len(self.member_reports)}",
            f"Verified evidence records: {len(self.evidence)}",
            f"Primary/authoritative sources: {len(self.sources)}",
            "",
            "## Evaluation protocol",
            "",
            f"- Metric: {self.evaluation_protocol.metric} ({self.evaluation_protocol.direction})",
            f"- Mode: {self.evaluation_protocol.mode}",
            f"- Split: {self.evaluation_protocol.split_strategy}",
            f"- Folds: {self.evaluation_protocol.folds}",
            f"- Leakage unit: {self.evaluation_protocol.leakage_unit}",
            f"- Protocol hash: `{self.evaluation_protocol.protocol_hash}`",
            "",
            "## Selected research portfolio",
            "",
        ]
        for item in self.selected_portfolio:
            lines.extend(
                (
                    f"### {item.get('hypothesis_id', 'hypothesis')}: {item.get('title', 'Untitled')}",
                    "",
                    str(item.get("rationale") or item.get("experiment") or ""),
                    "",
                    f"First experiment: {item.get('experiment', 'not specified')}",
                    f"Stopping rule: {item.get('stopping_rule', 'not specified')}",
                    "",
                )
            )
        if self.rejected_hypotheses:
            lines.extend(("## Rejected hypotheses", ""))
            for item in self.rejected_hypotheses:
                lines.append(
                    f"- `{item.get('hypothesis_id', 'unknown')}` — "
                    f"{item.get('reason', 'not selected by the council')}"
                )
            lines.append("")
        if self.member_reports:
            lines.extend(("## Independent member conclusions", ""))
            for report in self.member_reports:
                lines.append(
                    f"- `{report.get('member_id', 'member')}` — "
                    f"{report.get('summary', 'no summary')}"
                )
            lines.append("")
        if self.sources:
            lines.extend(("## Research provenance", ""))
            for source in self.sources[:40]:
                lines.append(
                    f"- `{source.get('source_id', 'source')}` "
                    f"[{source.get('title', 'Untitled')}]({source.get('url', '')})"
                )
            lines.append("")
        if self.prohibited_inputs:
            lines.extend(("## Prohibited inputs", ""))
            for item in self.prohibited_inputs:
                lines.append(f"- `{item.get('path')}` — {item.get('reason')}")
            lines.append("")
        if self.unresolved_questions:
            lines.extend(("## Unresolved questions", ""))
            lines.extend(f"- {question}" for question in self.unresolved_questions)
            lines.append("")
        if self.warnings:
            lines.extend(("## Warnings", ""))
            lines.extend(f"- {warning}" for warning in self.warnings)
            lines.append("")
        report_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
        return json_path, report_path
