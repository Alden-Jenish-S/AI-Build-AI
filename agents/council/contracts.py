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


def _short(value: object, limit: int) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: max(0, limit - 1)] + "…"


def _valid_bounded_context(payload: dict[str, Any], max_chars: int) -> str:
    """Render valid JSON and degrade optional detail instead of slicing JSON."""
    limit = max(1000, int(max_chars))
    value = dict(payload)
    text = json.dumps(value, indent=2, ensure_ascii=False, default=str)
    if len(text) <= limit:
        return text

    value.pop("research_sources", None)
    value["context_note"] = "Research-source detail omitted from this role-specific view."
    text = json.dumps(value, indent=2, ensure_ascii=False, default=str)
    if len(text) <= limit:
        return text

    fingerprint = value.get("problem_fingerprint")
    if isinstance(fingerprint, Mapping):
        retained_keys = (
            "data_kinds", "modalities", "is_multimodal", "table_shapes",
            "target_structure", "resource_inventory", "train_test_shift",
        )
        value["problem_fingerprint"] = {
            key: fingerprint[key] for key in retained_keys if key in fingerprint
        }
        value["problem_fingerprint_hash"] = content_hash(fingerprint)
    text = json.dumps(value, indent=2, ensure_ascii=False, default=str)
    if len(text) <= limit:
        return text

    for key in ("relevant_evidence", "measured_baselines", "selected_portfolio"):
        items = value.get(key)
        if not isinstance(items, list):
            continue
        compacted = []
        for item in items:
            if not isinstance(item, Mapping):
                continue
            compacted.append(
                {
                    name: (
                        _short(field, 240)
                        if isinstance(field, str)
                        else field
                    )
                    for name, field in item.items()
                    if name not in {"findings", "diagnostic_method", "limitations"}
                }
            )
        value[key] = compacted
    text = json.dumps(value, indent=2, ensure_ascii=False, default=str)
    if len(text) <= limit:
        return text

    # The immutable protocol and active IDs are never discarded. This final
    # valid-JSON digest is preferable to an invalid character-sliced document.
    minimal = {
        key: value[key]
        for key in (
            "brief_hash", "status", "evaluation_protocol", "active_hypothesis_id",
            "allowed_input_count", "prohibited_inputs",
        )
        if key in value
    }
    minimal["context_digest"] = content_hash(payload)
    minimal["context_note"] = "Optional evidence exceeded the context budget; use the active plan."
    return json.dumps(minimal, indent=2, ensure_ascii=False, default=str)


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
    measured_baselines: list[dict[str, Any]] = field(default_factory=list)
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
            "measured_baselines": self.measured_baselines,
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

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "CouncilBrief":
        """Restore a persisted brief without trusting derived hash fields.

        Older artifacts did not serialize ``measured_baselines``; accepting a
        missing field keeps those runs resumable while new artifacts preserve
        the evidence needed to execute the strongest measured control.
        """
        value = dict(raw)
        protocol_raw = value.get("evaluation_protocol")
        if not isinstance(protocol_raw, Mapping):
            raise ValueError("council brief is missing its evaluation protocol")
        direction = str(protocol_raw.get("direction") or "").strip().lower()
        metric = _clean_string(protocol_raw.get("metric"), fallback="task score")
        protocol = EvaluationProtocol.from_mapping(
            protocol_raw,
            metric=metric,
            direction=direction,
        )
        evidence = [
            dict(item)
            for item in value.get("evidence") or []
            if isinstance(item, Mapping)
        ]
        measured_baselines = [
            dict(item)
            for item in value.get("measured_baselines") or []
            if isinstance(item, Mapping)
        ]
        # Schema-v1 artifacts wrote focused diagnostic payloads into evidence
        # but accidentally omitted the convenience baseline list. Recover it
        # so an interrupted older run resumes with the same measured controls.
        if not measured_baselines:
            for record in evidence:
                summary = record.get("summary")
                payload = summary.get("result") if isinstance(summary, Mapping) else None
                baseline = (
                    payload.get("measured_baseline")
                    if isinstance(payload, Mapping)
                    else None
                )
                if isinstance(baseline, Mapping):
                    restored = dict(baseline)
                    restored.setdefault("diagnostic_method", payload.get("method", ""))
                    restored.setdefault(
                        "source", str(record.get("evidence_id") or "diagnostic")
                    )
                    restored.setdefault("metric", protocol.metric)
                    restored.setdefault("direction", protocol.direction)
                    measured_baselines.append(restored)
        def baseline_key(item: Mapping[str, Any]) -> float:
            try:
                return float(item.get("score"))
            except (TypeError, ValueError):
                return float("inf") if protocol.direction == "minimize" else float("-inf")
        measured_baselines.sort(
            key=baseline_key,
            reverse=protocol.direction != "minimize",
        )
        return cls(
            task_name=_clean_string(value.get("task_name"), fallback="task"),
            status=str(value.get("status") or "degraded"),
            problem_fingerprint=dict(value.get("problem_fingerprint") or {}),
            allowed_input_paths=tuple(value.get("allowed_input_paths") or ()),
            prohibited_inputs=tuple(
                dict(item)
                for item in value.get("prohibited_inputs") or ()
                if isinstance(item, Mapping)
            ),
            evaluation_protocol=protocol,
            evidence=evidence,
            measured_baselines=measured_baselines,
            sources=[
                dict(item)
                for item in value.get("sources") or []
                if isinstance(item, Mapping)
            ],
            member_reports=[
                dict(item)
                for item in value.get("member_reports") or []
                if isinstance(item, Mapping)
            ],
            hypotheses=[
                dict(item)
                for item in value.get("hypotheses") or []
                if isinstance(item, Mapping)
            ],
            selected_portfolio=[
                dict(item)
                for item in value.get("selected_portfolio") or []
                if isinstance(item, Mapping)
            ],
            rejected_hypotheses=[
                dict(item)
                for item in value.get("rejected_hypotheses") or []
                if isinstance(item, Mapping)
            ],
            unresolved_questions=[
                str(item) for item in value.get("unresolved_questions") or []
            ],
            warnings=[str(item) for item in value.get("warnings") or []],
            recommended_root_count=int(value.get("recommended_root_count") or 1),
            created_at_utc=str(
                value.get("created_at_utc")
                or datetime.now(timezone.utc).isoformat()
            ),
        )

    def search_context(self, max_chars: int = 6000) -> str:
        """Small immutable context for node ideation and search decisions."""
        selected = [
            {
                "hypothesis_id": item.get("hypothesis_id"),
                "title": _short(item.get("title"), 260),
                "model_family": item.get("model_family"),
                "architecture_track": item.get("architecture_track"),
                "modality_scope": item.get("modality_scope"),
            }
            for item in self.selected_portfolio
        ]
        compact = {
            "brief_hash": self.brief_hash,
            "status": self.status,
            "problem_fingerprint": self.problem_fingerprint,
            "evaluation_protocol": self.evaluation_protocol.to_dict(),
            "prohibited_inputs": list(self.prohibited_inputs),
            "allowed_input_count": len(self.allowed_input_paths),
            "measured_baselines": [
                {
                    "model": item.get("model"),
                    "model_family": item.get("model_family"),
                    "score": item.get("score"),
                    "metric": item.get("metric"),
                    "direction": item.get("direction"),
                    "diagnostic_method": _short(item.get("diagnostic_method"), 500),
                    "limitations": _short(item.get("limitations"), 300),
                }
                for item in self.measured_baselines[:4]
            ],
            "selected_portfolio": selected,
            "unresolved_questions": self.unresolved_questions,
        }
        return _valid_bounded_context(compact, max_chars)

    def implementation_context(
        self,
        hypothesis_id: str | None,
        max_chars: int = 6000,
    ) -> str:
        """Evidence scoped to the single experiment being implemented."""
        active = next(
            (
                item for item in self.selected_portfolio
                if str(item.get("hypothesis_id")) == str(hypothesis_id)
            ),
            None,
        )
        evidence_ids = {
            str(item) for item in (active or {}).get("evidence_ids", [])
        }
        relevant_evidence: list[dict[str, Any]] = []
        for record in self.evidence:
            evidence_id = str(record.get("evidence_id") or "")
            if evidence_id not in evidence_ids:
                continue
            summary = record.get("summary")
            result = summary.get("result") if isinstance(summary, Mapping) else None
            payload = result if isinstance(result, Mapping) else summary
            relevant_evidence.append(
                {
                    "evidence_id": evidence_id,
                    "kind": record.get("kind"),
                    "status": (
                        summary.get("status") if isinstance(summary, Mapping) else None
                    ),
                    "method": _short(
                        payload.get("method") if isinstance(payload, Mapping) else "", 700
                    ),
                    "findings": _short(
                        json.dumps(payload.get("findings"), default=str)
                        if isinstance(payload, Mapping) else "",
                        1200,
                    ),
                }
            )
        relevant_sources = [
            {
                "source_id": item.get("source_id"),
                "title": _short(item.get("title"), 240),
                "claim": _short(item.get("claim") or item.get("snippet"), 500),
            }
            for item in self.sources
            if str(item.get("source_id")) in evidence_ids
        ]
        compact = {
            "brief_hash": self.brief_hash,
            "status": self.status,
            "active_hypothesis_id": hypothesis_id,
            "evaluation_protocol": self.evaluation_protocol.to_dict(),
            "allowed_input_count": len(self.allowed_input_paths),
            "prohibited_inputs": list(self.prohibited_inputs),
            "active_hypothesis": active,
            "measured_baselines": self.measured_baselines[:3],
            "relevant_evidence": relevant_evidence,
            "research_sources": relevant_sources,
        }
        return _valid_bounded_context(compact, max_chars)

    def prompt_context(self, max_chars: int = 6000) -> str:
        """Compatibility alias for the search-scoped context view."""
        return self.search_context(max_chars)

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
        ]
        
        if self.measured_baselines:
            lines.extend(("## Measured baselines", ""))
            for item in self.measured_baselines:
                lines.append(
                    f"- **{item.get('model_family', 'unknown')}** ({item.get('model', 'unknown')}): "
                    f"`{item.get('score')}` ({item.get('metric')}) by {item.get('source', 'unknown')}"
                )
            lines.append("")

        lines.extend([
            "## Selected research portfolio",
            "",
        ])
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
