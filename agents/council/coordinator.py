"""Adaptive evidence gathering, peer review, and pre-search synthesis."""

from __future__ import annotations

import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from ..architecture_policy import annotate_hypothesis
from ..llm_utils import call_llm_json
from ..task_analyzer import TaskAnalysis
from .contracts import CouncilBrief, EvaluationProtocol, content_hash
from .diagnostics import (
    DiagnosticScriptRunner,
    build_problem_fingerprint,
    classify_input_access,
    collect_base_diagnostics,
)
from .research import ResearchRetriever


def _object_schema(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": required,
        "properties": properties,
    }


_MEMBER_SCHEMA = _object_schema(
    {
        "member_id": {"type": "string"},
        "mandate": {"type": "string"},
        "research_focus": {"type": "string"},
        "key_uncertainties": {"type": "array", "items": {"type": "string"}, "minItems": 1},
    },
    ["member_id", "mandate", "research_focus", "key_uncertainties"],
)
_COUNCIL_DESIGN_SCHEMA = _object_schema(
    {
        "coverage_rationale": {"type": "string"},
        "members": {"type": "array", "minItems": 2, "maxItems": 5, "items": _MEMBER_SCHEMA},
    },
    ["coverage_rationale", "members"],
)
_RESEARCH_REQUEST_SCHEMA = _object_schema(
    {
        "question": {"type": "string"},
        "queries": {
            "type": "array",
            "minItems": 2,
            "maxItems": 4,
            "items": {"type": "string"},
        },
    },
    ["question", "queries"],
)
_RESEARCH_PLAN_SCHEMA = _object_schema(
    {
        "requests": {
            "type": "array",
            "minItems": 2,
            "maxItems": 12,
            "items": _RESEARCH_REQUEST_SCHEMA,
        }
    },
    ["requests"],
)
_FINDING_SCHEMA = _object_schema(
    {
        "claim": {"type": "string"},
        "evidence_ids": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "number"},
    },
    ["claim", "evidence_ids", "confidence"],
)
_HYPOTHESIS_SCHEMA = _object_schema(
    {
        "hypothesis_id": {"type": "string"},
        "title": {"type": "string"},
        "model_family": {"type": "string"},
        "rationale": {"type": "string"},
        "evidence_ids": {"type": "array", "items": {"type": "string"}},
        "experiment": {"type": "string"},
        "expected_signal": {"type": "string"},
        "estimated_cost": {"type": "string"},
        "risks": {"type": "array", "items": {"type": "string"}},
        "stopping_rule": {"type": "string"},
        "compatible_with": {"type": "array", "items": {"type": "string"}},
        "architecture_track": {
            "type": "string",
            "enum": [
                "conventional",
                "established_neural",
                "custom_neural",
                "representation",
                "hybrid",
                "other",
            ],
        },
        "architecture_spec": {"type": "string"},
        "novelty_test": {"type": "string"},
        "modality_scope": {
            "type": "string",
            "enum": [
                "not_applicable",
                "single_modality",
                "fused_multimodal",
                "modality_ablation",
            ],
        },
        "modality_ablation": {"type": "string"},
    },
    [
        "hypothesis_id",
        "title",
        "model_family",
        "rationale",
        "evidence_ids",
        "experiment",
        "expected_signal",
        "estimated_cost",
        "risks",
        "stopping_rule",
        "compatible_with",
        "architecture_track",
        "architecture_spec",
        "novelty_test",
        "modality_scope",
        "modality_ablation",
    ],
)
_MEMBER_REPORT_SCHEMA = _object_schema(
    {
        "member_id": {"type": "string"},
        "summary": {"type": "string"},
        "findings": {"type": "array", "items": _FINDING_SCHEMA},
        "hypotheses": {"type": "array", "minItems": 1, "maxItems": 4, "items": _HYPOTHESIS_SCHEMA},
        "assumptions": {"type": "array", "items": {"type": "string"}},
        "unresolved_questions": {"type": "array", "items": {"type": "string"}},
    },
    ["member_id", "summary", "findings", "hypotheses", "assumptions", "unresolved_questions"],
)
_CROSS_REVIEW_SCHEMA = _object_schema(
    {
        "target_hypothesis_id": {"type": "string"},
        "strongest_objection": {"type": "string"},
        "fatal": {"type": "boolean"},
        "required_test": {"type": "string"},
    },
    ["target_hypothesis_id", "strongest_objection", "fatal", "required_test"],
)
_CRITIQUE_SCHEMA = _object_schema(
    {
        "summary": {"type": "string"},
        "cross_reviews": {"type": "array", "items": _CROSS_REVIEW_SCHEMA},
        "missing_evidence": {"type": "array", "items": {"type": "string"}},
        "requires_additional_member": {"type": "boolean"},
        "additional_mandate": {"type": "string"},
        "additional_research_focus": {"type": "string"},
    },
    [
        "summary",
        "cross_reviews",
        "missing_evidence",
        "requires_additional_member",
        "additional_mandate",
        "additional_research_focus",
    ],
)
_EVALUATION_SCHEMA = _object_schema(
    {
        "metric": {"type": "string"},
        "direction": {"type": "string", "enum": ["maximize", "minimize"]},
        "mode": {
            "type": "string",
            "enum": ["cross_validation", "holdout", "task_native"],
        },
        "split_strategy": {"type": "string"},
        "folds": {"type": "integer"},
        "seed": {"type": "integer"},
        "leakage_unit": {"type": "string"},
        "rationale": {"type": "string"},
    },
    ["metric", "direction", "mode", "split_strategy", "folds", "seed", "leakage_unit", "rationale"],
)
_REJECTED_SCHEMA = _object_schema(
    {
        "hypothesis_id": {"type": "string"},
        "reason": {"type": "string"},
    },
    ["hypothesis_id", "reason"],
)
_SYNTHESIS_SCHEMA = _object_schema(
    {
        "decision_summary": {"type": "string"},
        "evaluation_protocol": _EVALUATION_SCHEMA,
        "hypotheses": {"type": "array", "minItems": 1, "maxItems": 10, "items": _HYPOTHESIS_SCHEMA},
        "selected_hypothesis_ids": {
            "type": "array",
            "minItems": 1,
            "maxItems": 4,
            "items": {"type": "string"},
        },
        "rejected_hypotheses": {"type": "array", "items": _REJECTED_SCHEMA},
        "unresolved_questions": {"type": "array", "items": {"type": "string"}},
        "recommended_root_count": {"type": "integer"},
    },
    [
        "decision_summary",
        "evaluation_protocol",
        "hypotheses",
        "selected_hypothesis_ids",
        "rejected_hypotheses",
        "unresolved_questions",
        "recommended_root_count",
    ],
)


def _bounded_json(value: object, limit: int) -> str:
    text = json.dumps(value, indent=2, ensure_ascii=False, default=str)
    return text if len(text) <= limit else text[:limit] + "\n[truncated]"


def _clean_member_id(value: object, index: int) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(value or "").casefold()).strip("_")
    return normalized[:40] or f"member_{index}"


class CouncilCoordinator:
    """Coordinate independent investigators around a shared evidence ledger."""

    def __init__(
        self,
        *,
        python: str | None = None,
        model_name: str | None = None,
        max_members: int = 5,
        enable_web: bool = True,
        enable_generated_diagnostics: bool = True,
    ) -> None:
        self.python = str(python or sys.executable)
        self.model_name = model_name
        self.max_members = max(2, min(5, int(max_members)))
        self.enable_web = bool(enable_web)
        self.enable_generated_diagnostics = bool(enable_generated_diagnostics)

    @staticmethod
    def _log(message: str) -> None:
        print(f"MLResearchCouncil: {message}", flush=True)

    def _design_members(
        self, analysis: TaskAnalysis, diagnostics: Mapping[str, Any]
    ) -> dict[str, Any]:
        prompt = f"""
Design an adaptive council for the observed ML task. Do not choose members from a
fixed modality template. Create the smallest set of independent senior-engineer
mandates that collectively resolve the highest-impact uncertainties. Every mandate
must name a concrete question that can be answered by bounded local measurements.
Across the council, explicitly investigate the counterfactual that learned neural
representations or a task-invented trainable computation graph could outperform the
usual library models. This is a feasibility and information-gain question, not a
requirement to recommend a neural network. If it is unsuitable, identify which
measured sample-size, structure, compute, or validation evidence rules it out.
If the inventory contains more than one predictive modality, assign enough coverage
to determine whether fusion is actually beneficial. Treat full fusion, credible
single-modality controls, and leave-one-modality-out comparisons as competing
hypotheses; never assume that every provided modality is necessary.
The research_focus must be de-identified and suitable for general academic literature
search: no task name, competition name, file name, column name, or unique dataset phrase.

Maximum members: {self.max_members}

Task evidence:
{analysis.prompt_context(12000)}

Bounded preflight:
{_bounded_json(diagnostics, 16000)}
""".strip()
        payload = call_llm_json(
            "You are the chair of a senior ML research lab. Allocate experts according to evidence gaps, not canned roles.",
            prompt,
            schema=_COUNCIL_DESIGN_SCHEMA,
            schema_name="ml_research_council_design",
            model=self.model_name,
            temperature=0.1,
        )
        assert isinstance(payload, dict)
        members = []
        seen: set[str] = set()
        for index, raw in enumerate(payload.get("members", []), start=1):
            if not isinstance(raw, dict):
                continue
            member = dict(raw)
            member_id = _clean_member_id(member.get("member_id"), index)
            if member_id in seen:
                member_id = f"{member_id}_{index}"
            seen.add(member_id)
            member["member_id"] = member_id
            members.append(member)
        if len(members) < 2:
            raise ValueError("council design returned fewer than two unique members")
        payload["members"] = members[: self.max_members]
        return payload

    def _research_plan(
        self, fingerprint: Mapping[str, Any], members: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        current_year = datetime.now(timezone.utc).year
        focuses = [
            {
                "member_id": member["member_id"],
                "research_focus": member["research_focus"],
            }
            for member in members
        ]
        prompt = f"""
Create a focused academic-search plan from this de-identified ML problem. Decompose
each focus into answerable research questions. Give 2-4 materially different queries
per question. Every query must target a primary source using site:arxiv.org,
site:openreview.net, site:proceedings.mlr.press, site:jmlr.org,
site:proceedings.neurips.cc, site:openaccess.thecvf.com, site:aclanthology.org,
or another authoritative research domain. Use precise technical
concepts, evaluation conditions, and resource constraints. Never search for a dataset,
competition, notebook, winner, leaderboard, or exact solution. Never request code.
Prioritize work from {current_year - 3}–{current_year}; include older work only when it
is foundational and use a separate query so recency and foundations remain auditable.
Include precise searches for learned representations, neural inductive biases, and
component-level architecture mechanisms that match the measured data geometry and
resource regime. Search the actual uncertainty (for example, a specific interaction,
encoding, loss, or optimization mechanism under a stated sample/compute condition),
not a vague phrase such as "deep learning for tabular classification."
For a multimodal fingerprint, also search precise evidence about fusion versus
unimodal dominance, modality redundancy, missing-modality robustness, and controlled
modality ablation under the observed sample and compute regime.

Problem fingerprint:
{_bounded_json(fingerprint, 12000)}

Research focuses:
{_bounded_json(focuses, 6000)}
""".strip()
        payload = call_llm_json(
            "You are a research librarian for an ML lab. Optimize queries for precise primary literature retrieval.",
            prompt,
            schema=_RESEARCH_PLAN_SCHEMA,
            schema_name="ml_council_research_plan",
            model=self.model_name,
            temperature=0.0,
        )
        assert isinstance(payload, dict)
        return [dict(item) for item in payload.get("requests", []) if isinstance(item, dict)]

    @staticmethod
    def _protected_terms(analysis: TaskAnalysis) -> set[str]:
        common = {
            "train",
            "test",
            "sample",
            "submission",
            "target",
            "label",
            "data",
            "image",
            "audio",
            "classification",
            "regression",
        }
        terms: set[str] = set()
        for item in analysis.files:
            terms.update(
                token.casefold()
                for token in re.findall(r"[A-Za-z0-9_]+", Path(str(item.get("path") or "")).stem)
                if len(token) >= 4 and token.casefold() not in common
            )
            profile = item.get("profile", {})
            if isinstance(profile, Mapping):
                for column in profile.get("columns", []) or []:
                    token = str(column).strip().casefold()
                    if len(token) >= 4 and token not in common:
                        terms.add(token)
        return terms

    def _member_report(
        self,
        member: Mapping[str, Any],
        fingerprint: Mapping[str, Any],
        diagnostics: Mapping[str, Any],
        investigation: Mapping[str, Any],
        sources: list[dict[str, Any]],
    ) -> dict[str, Any]:
        research_context = [
            {
                "source_id": source.get("source_id"),
                "title": source.get("title"),
                "url": source.get("url"),
                "snippet": source.get("snippet"),
                "text": str(source.get("retrieved_text") or "")[:3500],
            }
            for source in sources[:24]
        ]
        local_evidence_id = f"diag_{member['member_id']}"
        prompt = f"""
Act as one independent senior ML engineer. Build task-specific research hypotheses
from measurements and cited literature. Do not imitate a known competition solution.
Do not recommend a model because it is fashionable. Every hypothesis needs a cheap,
discriminating first experiment, expected signal, cost, risks, and stopping rule.
Explicitly evaluate three counterfactuals: an established non-neural method, a
resource-bounded neural representation, and a custom neural architecture assembled
from primitive trainable operations around the observed inductive biases. A custom
architecture must specify its computation graph and an ablation against a simpler
control; do not merely rename an MLP, TabNet, transformer, or another library model.
If neural approaches are implausible, record the measured reason instead of silently
omitting them. Do not claim an architecture is unprecedented; state how a primary-
literature search and component ablations would test novelty and value.
When the fingerprint is multimodal, every relevant hypothesis must declare
`modality_scope` and a concrete `modality_ablation`. Compare all modalities, each
credible single modality, and leave-one-modality-out variants on identical validation
indices. Account for the extra parameters of fusion so added capacity is not mistaken
for modality value. The final recommendation may intentionally use only one modality.
Separate verified facts from assumptions. Evidence IDs may reference
`local_preflight`, `{local_evidence_id}`, and supplied source IDs.

Mandate:
{_bounded_json(member, 5000)}

De-identified fingerprint:
{_bounded_json(fingerprint, 10000)}

Local preflight evidence:
{_bounded_json(diagnostics, 14000)}

Focused diagnostic:
{_bounded_json(investigation, 7000)}

Primary literature evidence:
{_bounded_json(research_context, 22000)}
""".strip()
        payload = call_llm_json(
            "You are an independent senior ML research engineer. Prefer falsifiable, resource-feasible hypotheses.",
            prompt,
            schema=_MEMBER_REPORT_SCHEMA,
            schema_name=f"council_report_{member['member_id']}",
            model=self.model_name,
            temperature=0.2,
        )
        assert isinstance(payload, dict)
        payload["member_id"] = str(member["member_id"])
        valid_evidence_ids = {
            "local_preflight",
            local_evidence_id,
            *(str(source.get("source_id")) for source in sources),
        }
        removed_evidence: set[str] = set()
        for collection_name in ("findings", "hypotheses"):
            for item in payload.get(collection_name, []):
                if not isinstance(item, dict):
                    continue
                claimed = [str(value) for value in item.get("evidence_ids", [])]
                verified = [value for value in claimed if value in valid_evidence_ids]
                removed_evidence.update(set(claimed) - set(verified))
                item["evidence_ids"] = verified
        payload["evidence_validation"] = {
            "removed_unknown_ids": sorted(removed_evidence),
            "valid_ids": sorted(valid_evidence_ids),
        }
        return payload

    def _critique(self, reports: list[dict[str, Any]]) -> dict[str, Any]:
        prompt = f"""
Adversarially peer-review these independent ML research reports. Find leakage,
invalid validation comparisons, unsupported literature claims, resource failures,
duplicated model families, and experiments that cannot falsify their hypothesis.
Also flag any report set that never seriously evaluates neural representations or
task-invented architectures, or that calls a standard named architecture "custom."
For multimodal tasks, flag unsupported fusion claims, missing single-modality
controls, unequal validation splits, and conclusions based only on training loss.
For each important proposal, state its strongest objection and required test.
Request one additional council member only if a material question remains that the
current evidence cannot answer; otherwise set requires_additional_member=false and
return empty additional strings.

Reports:
{_bounded_json(reports, 36000)}
""".strip()
        payload = call_llm_json(
            "You are the skeptical principal scientist responsible for stopping weak or contaminated ML research directions.",
            prompt,
            schema=_CRITIQUE_SCHEMA,
            schema_name="ml_council_cross_review",
            model=self.model_name,
            temperature=0.1,
        )
        assert isinstance(payload, dict)
        return payload

    def _synthesize(
        self,
        analysis: TaskAnalysis,
        fingerprint: Mapping[str, Any],
        reports: list[dict[str, Any]],
        critique: Mapping[str, Any],
        sources: list[dict[str, Any]],
    ) -> dict[str, Any]:
        source_index = [
            {
                "source_id": item.get("source_id"),
                "title": item.get("title"),
                "url": item.get("url"),
                "snippet": item.get("snippet"),
            }
            for item in sources
        ]
        prompt = f"""
Synthesize a pre-search ML research brief. Select a Pareto portfolio rather than
voting: maximize expected score and information gain while controlling compute,
implementation risk, leakage, and redundancy. A cheap baseline may be selected when
it is a useful control, but there is no mandatory model progression. Prefer hypotheses
supported by both local evidence and primary literature. Preserve alternative
representations that could later ensemble well.

The full hypothesis ledger must explicitly resolve conventional, established-neural,
and custom-neural counterfactuals. When resources make training feasible, include at
least one falsifiable neural or custom-architecture hypothesis, even if it is not
selected. Set `architecture_track`; for a neural proposal, provide `architecture_spec`
with inputs, learned transformations, interaction mechanism, output head, loss,
regularization, optimizer, and stopping rule. For task-invented networks, provide a
`novelty_test` based on component-level literature searches and ablations. Never claim
global novelty from an incomplete search. If the selected portfolio contains only
conventional library models, give concrete measured evidence for rejecting both
neural tracks rather than relying on task simplicity.

When `is_multimodal` is true, the portfolio must resolve whether all modalities are
useful. Include a modality-ablation hypothesis with the full model, credible
single-modality models, and leave-one-modality-out variants evaluated on the same
folds. Prefer the smallest modality subset within score uncertainty; select fusion
only when its repeatable gain justifies its compute and failure modes. Populate
`modality_scope` and `modality_ablation` for every hypothesis.

Define one fixed validation protocol for every candidate. The split must respect any
observed entity, group, temporal, spatial, or source dependency. Choose cross-validation
only when its cost and independence assumptions are credible. Use metric
{analysis.metric!r} with direction {analysis.direction!r}.

Problem fingerprint:
{_bounded_json(fingerprint, 10000)}

Member reports:
{_bounded_json(reports, 38000)}

Adversarial review:
{_bounded_json(critique, 14000)}

Source index:
{_bounded_json(source_index, 12000)}
""".strip()
        payload = call_llm_json(
            "You chair an ML research council. You may synthesize claims but may not invent facts or uncited results.",
            prompt,
            schema=_SYNTHESIS_SCHEMA,
            schema_name="ml_council_synthesis",
            model=self.model_name,
            temperature=0.1,
        )
        assert isinstance(payload, dict)
        return payload

    @staticmethod
    def _fallback_brief(
        analysis: TaskAnalysis,
        fingerprint: dict[str, Any],
        diagnostics: dict[str, Any],
        allowed: tuple[str, ...],
        prohibited: tuple[dict[str, str], ...],
        warnings: list[str],
    ) -> CouncilBrief:
        protocol = EvaluationProtocol.from_mapping(
            {
                "mode": "cross_validation" if analysis.metric != "task score" else "holdout",
                "folds": 5,
                "seed": 42,
                "split_strategy": "deterministic stratified or dependency-aware split after inspecting labels",
                "leakage_unit": "sample or discovered group/entity",
                "rationale": "Council synthesis was degraded; use a conservative shared protocol.",
            },
            metric=analysis.metric,
            direction=analysis.direction,
        )
        hypothesis = {
            "hypothesis_id": "H_fallback_control",
            "title": "Evidence-compatible measured control",
            "model_family": "resource-compatible task-native baseline",
            "rationale": (
                "Establish a valid score using the observed inputs, available dependencies, "
                "and shared evaluation protocol before spending budget on refinements."
            ),
            "evidence_ids": ["local_preflight"],
            "experiment": (
                "Inspect the allowed inputs, implement the strongest dependency-available baseline "
                "that matches their actual representation, and record fold-level measurements."
            ),
            "expected_signal": "A reproducible runnable control and evidence about the dominant error mode.",
            "estimated_cost": "low to moderate",
            "risks": ["Council LLM or web research was unavailable."],
            "stopping_rule": "Stop after one valid shared-protocol result; do not repeatedly tune this control.",
            "compatible_with": [],
            "architecture_track": "other",
            "architecture_spec": "",
            "novelty_test": (
                "After the control is measured, test a resource-bounded task-tailored neural "
                "architecture if the main search has enough idea budget."
            ),
            "modality_scope": (
                "modality_ablation"
                if fingerprint.get("is_multimodal")
                else "not_applicable"
            ),
            "modality_ablation": (
                "If multiple predictive modalities are discovered during implementation, "
                "compare full fusion, single-modality controls, and leave-one-out variants "
                "on identical validation folds."
            ),
        }
        return CouncilBrief(
            task_name=analysis.task_name,
            status="degraded",
            problem_fingerprint=fingerprint,
            allowed_input_paths=allowed,
            prohibited_inputs=prohibited,
            evaluation_protocol=protocol,
            evidence=[
                {
                    "evidence_id": "local_preflight",
                    "kind": "local_diagnostics",
                    "hash": diagnostics.get("diagnostics_hash"),
                    "summary": diagnostics,
                }
            ],
            hypotheses=[hypothesis],
            selected_portfolio=[hypothesis],
            unresolved_questions=[
                "Would a learned representation or task-tailored neural computation graph "
                "outperform the measured control under the available compute budget?"
            ],
            warnings=warnings,
            recommended_root_count=1,
        )

    def run(
        self,
        analysis: TaskAnalysis,
        run_root: Path,
    ) -> CouncilBrief:
        council_dir = Path(run_root) / "council"
        diagnostics_dir = council_dir / "diagnostics"
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
        research_sources_path = council_dir / "research_sources.jsonl"
        query_audit_path = council_dir / "query_audit.jsonl"
        research_sources_path.write_text("", encoding="utf-8")
        query_audit_path.write_text("", encoding="utf-8")
        warnings: list[str] = []
        try:
            diagnostics, allowed, prohibited = collect_base_diagnostics(analysis)
        except Exception as exc:
            allowed, prohibited = classify_input_access(analysis)
            diagnostics = {
                "schema_version": 1,
                "analysis_kind": "minimal preflight fallback",
                "preflight_error": f"{type(exc).__name__}: {exc}",
                "prohibited_inputs": list(prohibited),
            }
            diagnostics["diagnostics_hash"] = content_hash(diagnostics)
            warnings.append(
                f"Bounded preflight failed: {type(exc).__name__}: {exc}"
            )
        fingerprint = build_problem_fingerprint(analysis, diagnostics)
        (diagnostics_dir / "preflight.json").write_text(
            json.dumps(diagnostics, indent=2, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8",
        )
        (council_dir / "problem_fingerprint.json").write_text(
            json.dumps(fingerprint, indent=2, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8",
        )
        try:
            self._log("Designing evidence-gap-specific member mandates.")
            design = self._design_members(analysis, diagnostics)
            members = list(design["members"])
            (council_dir / "council_design.json").write_text(
                json.dumps(design, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
        except Exception as exc:
            warnings.append(f"Council design failed: {type(exc).__name__}: {exc}")
            query_audit_path.write_text(
                json.dumps(
                    {
                        "accepted": False,
                        "reason": "research did not start because council design failed",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            brief = self._fallback_brief(
                analysis, fingerprint, diagnostics, allowed, prohibited, warnings
            )
            brief.write(council_dir)
            return brief

        sources: list[dict[str, Any]] = []
        audit: list[dict[str, Any]] = []
        if self.enable_web:
            try:
                self._log("Planning multiple de-identified primary-literature searches.")
                research_requests = self._research_plan(fingerprint, members)
                (council_dir / "research_plan.json").write_text(
                    json.dumps(
                        {"requests": research_requests},
                        indent=2,
                        ensure_ascii=False,
                        default=str,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                retriever = ResearchRetriever(
                    analysis.task_name,
                    council_dir,
                    forbidden_terms=self._protected_terms(analysis),
                )
                sources, audit = retriever.collect(research_requests)
                if not sources:
                    warnings.append("No primary literature source survived query and provenance checks.")
            except Exception as exc:
                warnings.append(f"Literature research failed: {type(exc).__name__}: {exc}")
                query_audit_path.write_text(
                    json.dumps(
                        {
                            "accepted": False,
                            "reason": f"literature research failed: {type(exc).__name__}: {exc}",
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
        else:
            warnings.append("Web research was disabled; council used local evidence only.")
            query_audit_path.write_text(
                json.dumps(
                    {"accepted": False, "reason": "web research was disabled"}
                )
                + "\n",
                encoding="utf-8",
            )

        investigations: dict[str, dict[str, Any]] = {}
        if self.enable_generated_diagnostics:
            self._log(f"Running {len(members)} focused local diagnostic investigations.")
            runner = DiagnosticScriptRunner(self.python, self.model_name)
            with ThreadPoolExecutor(max_workers=min(4, len(members))) as pool:
                future_map = {
                    pool.submit(
                        runner.run,
                        str(member["member_id"]),
                        str(member["mandate"]),
                        analysis,
                        diagnostics,
                        council_dir,
                        allowed,
                    ): str(member["member_id"])
                    for member in members
                }
                for future in as_completed(future_map):
                    member_id = future_map[future]
                    try:
                        investigations[member_id] = future.result()
                    except Exception as exc:
                        investigations[member_id] = {
                            "member_id": member_id,
                            "status": "worker_failed",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
        else:
            warnings.append("Generated diagnostics were disabled; only the bounded preflight was used.")

        reports: list[dict[str, Any]] = []
        self._log("Collecting independent evidence-backed research proposals.")
        with ThreadPoolExecutor(max_workers=min(4, len(members))) as pool:
            future_map = {
                pool.submit(
                    self._member_report,
                    member,
                    fingerprint,
                    diagnostics,
                    investigations.get(str(member["member_id"]), {}),
                    sources,
                ): str(member["member_id"])
                for member in members
            }
            for future in as_completed(future_map):
                member_id = future_map[future]
                try:
                    reports.append(future.result())
                except Exception as exc:
                    warnings.append(
                        f"Member {member_id} failed: {type(exc).__name__}: {exc}"
                    )
        reports.sort(key=lambda item: str(item.get("member_id")))
        if not reports:
            warnings.append("All council member reports failed.")
            brief = self._fallback_brief(
                analysis, fingerprint, diagnostics, allowed, prohibited, warnings
            )
            brief.sources = sources
            brief.write(council_dir)
            return brief

        try:
            self._log("Running adversarial cross-review.")
            critique = self._critique(reports)
        except Exception as exc:
            warnings.append(f"Cross-review failed: {type(exc).__name__}: {exc}")
            critique = {
                "summary": "Cross-review unavailable.",
                "cross_reviews": [],
                "missing_evidence": [],
                "requires_additional_member": False,
                "additional_mandate": "",
                "additional_research_focus": "",
            }

        if (
            critique.get("requires_additional_member")
            and len(members) < self.max_members
            and str(critique.get("additional_mandate") or "").strip()
        ):
            extra_id = f"gap_specialist_{len(members) + 1}"
            extra = {
                "member_id": extra_id,
                "mandate": str(critique["additional_mandate"]),
                "research_focus": str(critique.get("additional_research_focus") or ""),
                "key_uncertainties": list(critique.get("missing_evidence") or []),
            }
            self._log(f"Spawning one additional member for unresolved evidence: {extra_id}.")
            investigation: dict[str, Any] = {}
            if self.enable_generated_diagnostics:
                try:
                    investigation = DiagnosticScriptRunner(
                        self.python, self.model_name
                    ).run(
                        extra_id,
                        str(extra["mandate"]),
                        analysis,
                        diagnostics,
                        council_dir,
                        allowed,
                    )
                except Exception as exc:
                    investigation = {"status": "worker_failed", "error": str(exc)}
            try:
                reports.append(
                    self._member_report(
                        extra, fingerprint, diagnostics, investigation, sources
                    )
                )
                members.append(extra)
            except Exception as exc:
                warnings.append(f"Additional member failed: {type(exc).__name__}: {exc}")

        try:
            self._log("Synthesizing a Pareto-ranked initial experiment portfolio.")
            synthesis = self._synthesize(
                analysis, fingerprint, reports, critique, sources
            )
            protocol = EvaluationProtocol.from_mapping(
                synthesis.get("evaluation_protocol"),
                metric=analysis.metric,
                direction=analysis.direction,
            )
            hypotheses = [
                annotate_hypothesis(dict(item))
                for item in synthesis.get("hypotheses", [])
                if isinstance(item, dict)
            ]
            valid_evidence_ids = {
                "local_preflight",
                *(f"diag_{member_id}" for member_id in investigations),
                *(str(source.get("source_id")) for source in sources),
            }
            for hypothesis in hypotheses:
                claimed = [str(item) for item in hypothesis.get("evidence_ids", [])]
                hypothesis["evidence_ids"] = [
                    item for item in claimed if item in valid_evidence_ids
                ]
                removed = sorted(set(claimed) - valid_evidence_ids)
                if removed:
                    warnings.append(
                        f"Hypothesis {hypothesis.get('hypothesis_id')} cited unknown evidence IDs: "
                        + ", ".join(removed)
                    )
            if fingerprint.get("is_multimodal") and not any(
                item.get("modality_scope") == "modality_ablation"
                for item in hypotheses
            ):
                modalities = [
                    str(item) for item in fingerprint.get("predictive_modalities", [])
                ]
                hypotheses.append(
                    {
                        "hypothesis_id": "H_modality_contribution_audit",
                        "title": "Measured modality contribution and fusion audit",
                        "model_family": (
                            "task-selected per-modality controls with validation-selected fusion"
                        ),
                        "rationale": (
                            "Providing multiple modalities does not establish that each contributes "
                            "independent predictive signal. Controlled ablations can select a simpler, "
                            "stronger, or more robust modality subset."
                        ),
                        "evidence_ids": ["local_preflight"],
                        "experiment": (
                            "Using identical validation indices, compare the full fusion model, "
                            "credible models using each modality alone, and leave-one-modality-out "
                            f"variants for: {', '.join(modalities)}. Match capacity where practical."
                        ),
                        "expected_signal": (
                            "A ranked modality subset showing whether fusion adds repeatable value "
                            "beyond the strongest single modality."
                        ),
                        "estimated_cost": "moderate; reuse cached preprocessing and embeddings",
                        "risks": [
                            "Fusion capacity can confound modality contribution.",
                            "Unequal folds can make ablation scores incomparable.",
                        ],
                        "stopping_rule": (
                            "Prefer the smallest modality subset whose score is within validation "
                            "uncertainty of the best variant; retain fusion only for repeatable gain."
                        ),
                        "compatible_with": [],
                        "architecture_track": "representation",
                        "architecture_spec": "",
                        "novelty_test": "Not a novelty claim; this is a contribution ablation.",
                        "modality_scope": "modality_ablation",
                        "modality_ablation": (
                            "Full fusion, each credible single modality, and every leave-one-out "
                            "variant on the exact shared folds."
                        ),
                        "selection_policy": (
                            "Injected because multiple predictive modalities require measured "
                            "necessity rather than an assumption of useful fusion."
                        ),
                    }
                )
            selected_ids = {
                str(item) for item in synthesis.get("selected_hypothesis_ids", [])
            }
            selected = [
                item
                for item in hypotheses
                if str(item.get("hypothesis_id")) in selected_ids
                and item.get("evidence_ids")
            ]
            if not selected:
                selected = [item for item in hypotheses if item.get("evidence_ids")][:1]
                if selected:
                    warnings.append(
                        "Chair selected no traceable hypothesis; retained the first evidence-linked proposal."
                    )
            if not selected:
                raise ValueError("Council synthesis produced no evidence-linked hypothesis")
            selected_tracks = {
                str(item.get("architecture_track") or "") for item in selected
            }
            packages = (
                fingerprint.get("resource_inventory", {}).get("packages_available", {})
                if isinstance(fingerprint.get("resource_inventory"), Mapping)
                else {}
            )
            torch_available = bool(
                packages.get("torch") if isinstance(packages, Mapping) else False
            )
            promoted_architecture = False
            if (
                torch_available
                and not selected_tracks.intersection(
                    {"established_neural", "custom_neural"}
                )
            ):
                architecture_candidate = next(
                    (
                        item
                        for item in hypotheses
                        if item not in selected
                        and item.get("evidence_ids")
                        and item.get("architecture_track")
                        in {"established_neural", "custom_neural"}
                    ),
                    None,
                )
                if architecture_candidate is not None:
                    architecture_candidate["selection_policy"] = (
                        "Promoted as a bounded architecture counterfactual so the initial "
                        "portfolio is not restricted to conventional library models."
                    )
                    insertion = min(1, len(selected))
                    selected.insert(insertion, architecture_candidate)
                    if len(selected) > 4:
                        selected.pop()
                    promoted_architecture = True
            promoted_modality = False
            if fingerprint.get("is_multimodal") and not any(
                item.get("modality_scope") == "modality_ablation"
                for item in selected
            ):
                modality_candidate = next(
                    (
                        item
                        for item in hypotheses
                        if item.get("modality_scope") == "modality_ablation"
                        and item.get("evidence_ids")
                    ),
                    None,
                )
                if modality_candidate is not None:
                    insertion = min(1, len(selected))
                    selected.insert(insertion, modality_candidate)
                    if len(selected) > 4:
                        selected.pop()
                    promoted_modality = True
            recommended_roots = int(
                synthesis.get("recommended_root_count", len(selected)) or 1
            )
            if promoted_architecture or promoted_modality:
                recommended_roots = max(2, recommended_roots)
            brief = CouncilBrief(
                task_name=analysis.task_name,
                status="degraded" if warnings else "completed",
                problem_fingerprint=fingerprint,
                allowed_input_paths=allowed,
                prohibited_inputs=prohibited,
                evaluation_protocol=protocol,
                evidence=[
                    {
                        "evidence_id": "local_preflight",
                        "kind": "local_diagnostics",
                        "hash": diagnostics.get("diagnostics_hash"),
                        "summary": diagnostics,
                    },
                    *[
                        {
                            "evidence_id": f"diag_{member_id}",
                            "kind": "focused_diagnostic",
                            "hash": content_hash(result),
                            "summary": result,
                        }
                        for member_id, result in sorted(investigations.items())
                    ],
                ],
                sources=sources,
                member_reports=reports,
                hypotheses=hypotheses,
                selected_portfolio=selected,
                rejected_hypotheses=[
                    dict(item)
                    for item in synthesis.get("rejected_hypotheses", [])
                    if isinstance(item, dict)
                ],
                unresolved_questions=[
                    str(item) for item in synthesis.get("unresolved_questions", [])
                ],
                warnings=warnings,
                recommended_root_count=recommended_roots,
            )
        except Exception as exc:
            warnings.append(f"Council synthesis failed: {type(exc).__name__}: {exc}")
            brief = self._fallback_brief(
                analysis, fingerprint, diagnostics, allowed, prohibited, warnings
            )
            brief.sources = sources
            brief.member_reports = reports

        (council_dir / "member_reports.json").write_text(
            json.dumps(reports, indent=2, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8",
        )
        (council_dir / "cross_review.json").write_text(
            json.dumps(critique, indent=2, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8",
        )
        (council_dir / "evidence.jsonl").write_text(
            "".join(
                json.dumps(item, ensure_ascii=False, default=str) + "\n"
                for item in brief.evidence
            ),
            encoding="utf-8",
        )
        brief.write(council_dir)
        self._log(
            f"Council completed with {len(reports)} reports, {len(sources)} sources, "
            f"and {len(brief.selected_portfolio)} selected hypotheses."
        )
        return brief
