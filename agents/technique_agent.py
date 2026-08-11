"""LLM planning for direct task implementations."""

from __future__ import annotations

import re
import json
from typing import Any, TYPE_CHECKING

from .architecture_policy import classify_architecture
from .llm_utils import call_llm, call_llm_json
from .modality_policy import predictive_modality_inventory
from .task_analyzer import TaskAnalysis

if TYPE_CHECKING:
    from .council.contracts import CouncilBrief


def _plan_requests_modality_ablation(plan: str | None) -> bool:
    """Detect a modality-contribution request without relying on literal markers."""
    normalized = re.sub(r"[^a-z]+", " ", str(plan or "").casefold())
    normalized = " ".join(normalized.split())
    if "modality scope modality ablation" in normalized:
        return True
    if "modality ablation" in normalized:
        return True
    if "leave one modality out" in normalized:
        return True
    return False


_DECISION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["action", "target_node_ids", "reasoning"],
    "additionalProperties": False,
    "properties": {
        "action": {
            "type": "string",
            "enum": ["merge", "tune", "refine", "diversify", "architect", "transfer", "finalize"],
        },
        "target_node_ids": {"type": "array", "items": {"type": "string"}},
        "thinking": {"type": "string"},
        "reasoning": {"type": "string"},
        "plan": {"type": "string"},
    },
}


class TechniqueAgent:
    def __init__(
        self,
        model_name: str | None = None,
        council_brief: "CouncilBrief | None" = None,
    ) -> None:
        self.model_name = model_name
        self.council_brief = council_brief

    def set_council_brief(self, brief: "CouncilBrief | None") -> None:
        self.council_brief = brief

    def _council_context(self, max_chars: int = 12000) -> str:
        if self.council_brief is None:
            return ""
        return (
            "\n\n# ML Research Council brief (authoritative pre-search evidence)\n"
            + self.council_brief.prompt_context(max_chars)
        )

    def _portfolio_plans(self, count: int) -> list[str]:
        if self.council_brief is None:
            return []
        plans: list[str] = []
        for hypothesis in self.council_brief.selected_portfolio[:count]:
            plan = "\n".join(
                (
                    f"Hypothesis ID: {hypothesis.get('hypothesis_id', 'unknown')}",
                    f"Research direction: {hypothesis.get('title', 'Untitled hypothesis')}",
                    f"Model family or representation: {hypothesis.get('model_family', 'task-selected')}",
                    "Architecture track: "
                    f"{hypothesis.get('architecture_track') or classify_architecture(json.dumps(hypothesis, default=str))}",
                    f"Architecture specification: {hypothesis.get('architecture_spec', '')}",
                    f"Novelty/value test: {hypothesis.get('novelty_test', '')}",
                    f"Modality scope: {hypothesis.get('modality_scope', 'not_applicable')}",
                    f"Modality ablation: {hypothesis.get('modality_ablation', '')}",
                    f"Evidence-backed rationale: {hypothesis.get('rationale', '')}",
                    f"First experiment: {hypothesis.get('experiment', '')}",
                    f"Expected signal: {hypothesis.get('expected_signal', '')}",
                    f"Estimated cost: {hypothesis.get('estimated_cost', '')}",
                    "Risks: " + "; ".join(str(item) for item in hypothesis.get("risks", [])),
                    f"Stopping rule: {hypothesis.get('stopping_rule', '')}",
                    f"Council brief hash: {self.council_brief.brief_hash}",
                    f"Evaluation protocol hash: {self.council_brief.evaluation_protocol.protocol_hash}",
                    "Implement this discriminating experiment using only council-approved inputs. "
                    "Preserve the fixed evaluation protocol and report fold-level evidence.",
                )
            )
            plans.append(plan[:8000])
        return plans

    @staticmethod
    def _parse_plans(response: str, count: int) -> list[str]:
        cleaned = re.sub(r"(?s)<thinking>.*?</thinking>", "", response).strip()
        chunks = re.split(r"(?im)^\s*(?:PLAN\s*\d*|\d+[.)])\s*[:.-]?\s*", cleaned)
        plans = [" ".join(chunk.split()) for chunk in chunks if len(chunk.split()) >= 8]
        if not plans and cleaned.strip():
            plans = [" ".join(cleaned.split())]
        unique: list[str] = []
        for plan in plans:
            if plan not in unique:
                unique.append(plan[:4000])
        return unique[:count]

    def search_for_new_ideas(self, analysis: TaskAnalysis) -> str:
        """Query web search for competitive ML strategies and novel literature ideas tailored to this task."""
        if self.council_brief is not None:
            return "\n\n".join(
                f"Title: {item.get('title', '')}\nURL: {item.get('url', '')}\n"
                f"Council claim: {item.get('claim') or item.get('snippet', '')}"
                for item in self.council_brief.sources[:12]
            )
        try:
            from .web_search import search_web
            prompt = (
                f"{analysis.prompt_context(4000)}\n\n"
                "Based on the task description and data properties above, formulate a highly effective "
                "web search query to find state-of-the-art machine learning architectures, novel literature, "
                f"or general ML strategies specific to this problem type. The target metric is {analysis.metric}.\n"
                "CRITICAL ANTI-PLAGIARISM RULE: Do NOT search for 'Kaggle winning solutions', 'notebooks', or exact answers for this specific dataset. "
                "Search ONLY for general ML techniques, architectures, or strategies that apply to this modality.\n\n"
                "Return ONLY the exact search query string to use, without quotes, explanations, or introductory text. Keep it concise (under 150 characters)."
            )
            query = call_llm(
                "You are an expert Machine Learning Researcher formulating web search queries.",
                prompt,
                model=self.model_name,
                temperature=0.2,
            ).strip().strip('"\',.')
            
            query = query[:150].strip()
            print(f"TechniqueAgent: LLM generated web search query: '{query}'...", flush=True)
            results = search_web(query)
            if not results:
                fallback = f"{analysis.task_name} machine learning competitive approaches {analysis.metric}"[:150].strip()
                print(f"TechniqueAgent: Fallback web search: '{fallback}'...", flush=True)
                results = search_web(fallback)
            return results
        except Exception as exc:
            print(f"TechniqueAgent: Web search call failed: {exc}", flush=True)
            return ""

    def generate_initial_approaches(
        self,
        analysis: TaskAnalysis,
        count: int = 3,
    ) -> list[str]:
        """Ask once for several materially different implementation plans with distinct model families."""
        count = max(1, int(count))
        council_plans = self._portfolio_plans(count)
        if council_plans:
            print(
                f"TechniqueAgent: Using {len(council_plans)} evidence-ranked council hypotheses.",
                flush=True,
            )
            return council_plans
        print(f"TechniqueAgent: Requesting {count} initial implementation plans (distinct model families).", flush=True)
        
        web_insights = self.search_for_new_ideas(analysis)
        web_section = f"\nWeb Search Literature & Solution Insights:\n{web_insights[:3000]}\n" if web_insights else ""

        prompt = (
            f"{analysis.prompt_context(14000)}\n"
            f"{web_section}\n"
            "CRITICAL REASONING & PLANNING INSTRUCTIONS:\n\n"
            "1. THINK BEFORE YOU DECIDE:\n"
            "   Begin your response with a <thinking> ... </thinking> block where you step-by-step:\n"
            f"   a. Task & Resource Inspection: Analyze the task goal, score metric ({analysis.metric} {analysis.direction}), expected output deliverable, and observed files. Do NOT assume hardcoded file names, columns, or schemas—inspect actual resources dynamically from `input/`.\n"
            "   b. Data Modality & Architecture Suitability: Inspect the observed data modalities (tabular, vision, text, audio, etc.). Consider BOTH simple foundational algorithms (e.g., Logistic Regression, SVM, Naive Bayes) and complex SOTA architectures (e.g., custom PyTorch, GBDT). Simple baselines can often outperform complex models.\n"
            "   c. Novel & Task-Tailored Techniques: Where appropriate, think through domain-specific data augmentations, metric-aligned loss functions, and specialized feature representations.\n"
            "   d. Neural Counterfactual: Explicitly assess whether learned representations could capture signal that tree or linear libraries cannot. Distinguish an established neural baseline from a task-invented computation graph built from primitive trainable operations. If rejecting both, cite concrete data-size, structure, compute, or validation evidence.\n"
            "   e. Modality Necessity: If more than one predictive modality is present, do not assume fusion helps. Plan identical-fold comparisons for full fusion, credible single-modality controls, and leave-one-modality-out variants; allow a single modality to win.\n"
            "   f. Model Size & Complexity vs Customization: Compare lightweight custom architectures with tailored modules against fine-tuning pretrained standard backbones. Reason through the right tradeoff for this dataset.\n"
            "   g. Portfolio Design: Select the approaches with the strongest expected value and information gain under the observed resource budget. Include a simple baseline only when it is a useful measured control; do not force an architecture progression.\n\n"
            "2. PROPOSAL REQUIREMENTS:\n"
            f"   Propose {count} executable approaches for this task. Each approach must explain how to load observed paths dynamically from input/, build an honest local validation score matching {analysis.metric} ({analysis.direction}), fit/train, tune, and write the requested deliverable.\n"
            "   STRICT REQUIREMENT: Use a materially DIFFERENT model family / algorithm in each plan. "
            "Start each plan with a `Model family: <name>` line naming the family you select. "
            "When two or more plans are requested and neural training is resource-feasible, at least one plan must measure a neural counterfactual. A neural plan can either describe a custom computation graph, or measure pretrained-fine-tune as a first-class counterfactual. "
            "DO NOT repeat the same model family across plans. Return sections labelled "
            "PLAN 1:, PLAN 2:, and so on, preceded by your <thinking> ... </thinking> block."
        )
        try:
            response = call_llm(
                "You are an expert AI Machine Learning Architect. You inspect task resources thoroughly, reason step-by-step, and design competitive SOTA implementations tailored to observed data modalities.",
                prompt,
                model=self.model_name,
                temperature=0.25,
            )
            plans = self._parse_plans(response, count)
            print(f"TechniqueAgent: Parsed {len(plans)} plans from the LLM response.", flush=True)
        except Exception as exc:
            print(f"TechniqueAgent: planning call failed; using resilient defaults: {exc}")
            plans = []

        defaults = [
            (
                "Build a dependable baseline after inspecting the listed inputs at runtime. "
                "Use a task-appropriate preprocessing pipeline, a robust conventional model "
                "or algorithm, honest local validation/proxy scoring, and emit the exact sample format."
            ),
            (
                "Run an architecture counterfactual suitable for the observed data. Build a compact "
                "custom PyTorch nn.Module from primitive layers after inspecting modality, shapes, "
                "sample count, feature types, and available hardware. Derive its trainable interactions "
                "from those observations instead of instantiating a named tabular architecture. Compare "
                "it honestly with the conventional control, use early stopping, and retain it only when "
                "the fixed local protocol improves."
            ),
            (
                "Use an efficient representation or feature pipeline tailored to the actual file "
                "contents, compare a compact set of model settings, retain the strongest local result, "
                "and generate the submission directly from that implementation."
            ),
        ]
        for default in defaults:
            if len(plans) >= count:
                break
            plans.append(
                default
                + f" Use one deterministic local estimate of {analysis.metric} ({analysis.direction}) "
                "across every approach; if the official score is hidden, use the same deterministic "
                "stability/holdout proxy in every node."
            )
        modality_inventory = predictive_modality_inventory(analysis.files)
        if modality_inventory["is_multimodal"] and not any(
            _plan_requests_modality_ablation(plan) for plan in plans
        ):
            modalities = ", ".join(modality_inventory["modalities"])
            audit_plan = (
                "Modality scope: modality_ablation\n"
                f"Predictive modalities: {modalities}. Determine rather than assume which inputs "
                "are necessary. On identical validation indices, compare a resource-matched full "
                "fusion model, credible models using each modality alone, and every leave-one-"
                "modality-out variant. Cache reusable representations, report all ablation scores, "
                "and select the smallest modality subset within validation uncertainty of the best "
                "score. Generate the requested output using that measured winner."
            )
            if len(plans) >= count:
                plans[-1] = audit_plan
            else:
                plans.append(audit_plan)
        return plans[:count]

    def propose_tuning(
        self,
        analysis: TaskAnalysis,
        plan: str,
        score: float,
        diagnostics: str = "",
    ) -> str:
        """Turn a successful implementation into a bounded hyperparameter search."""
        print(f"TechniqueAgent: Requesting focused tuning search for score {score}.", flush=True)
        prompt = (
            f"{analysis.prompt_context(10000)}\n\n"
            f"{self._council_context(9000)}\n\n"
            f"Current plan:\n{plan[:5000]}\n\n"
            f"Current local score: {score} ({analysis.direction}).\n"
            f"Run notes:\n{diagnostics[-3000:]}\n\n"
            "CRITICAL REASONING INSTRUCTIONS:\n\n"
            "1. THINK BEFORE YOU DECIDE:\n"
            "   Begin your response with a <thinking> ... </thinking> block where you step-by-step:\n"
            "   a. Evaluate the current plan, local score, and execution diagnostics.\n"
            "   b. Model Suitability & Performance Assessment: Verify if this current model architecture actually suits the task and data properties. Assess whether a bounded hyperparameter search (rather than structural changes) is the highest-leverage next step.\n"
            "   c. Choose the 2-6 hyperparameters most likely to move the metric. Prefer axes that matter for this data size and estimator: e.g. regularization, learning rate, tree depth/leaves, feature subsampling, embedding dim, batch size, weight decay, augmentation strength.\n\n"
            "2. SEARCH SPACE SPECIFICATION (mandatory):\n"
            "   Return ONE executable plan that:\n"
            "   a. Preserves the core estimator family, the working data paths, validation protocol, and output behavior of the current plan.\n"
            "   b. Defines a `Search space:` section listing every selected hyperparameter with concrete ranges or choices (uniform/int/log ranges, or discrete candidates).\n"
            "   c. Defines a `Search budget:` section: a bounded configuration count (at most 40, scale down for expensive/deep models or small data) and an early-stopping rule per configuration.\n"
            "   d. Evaluates every candidate configuration on the IDENTICAL validation protocol, keeps the best configuration honestly, reports the final score and the winning configuration.\n"
            "   Do not settle for a single hand-picked configuration; the plan must actually search."
        )
        try:
            response = call_llm(
                "You are an expert AI ML Optimization Engineer defining bounded hyperparameter searches over working models without destabilizing them.",
                prompt,
                model=self.model_name,
                temperature=0.15,
            )
            cleaned = re.sub(r"(?s)<thinking>.*?</thinking>", "", response).strip()
            return cleaned[:6000]
        except Exception as exc:
            return (
                f"Keep the working implementation described here: {plan}. Perform a bounded "
                f"internal hyperparameter search around its current values: define a `Search space:` "
                f"of 2-5 hyperparameters with concrete ranges, run 20-40 configurations evaluated on "
                f"the identical validation protocol with per-config early stopping, keep the best "
                f"configuration, and report the final score and winning settings. Planning service note: {exc}"
            )[:6000]

    def propose_architecture_exploration(
        self,
        analysis: TaskAnalysis,
        parent_plan: str,
        score: float,
        *,
        measured_alternatives: str = "",
        plateau_evidence: str = "",
        require_custom: bool = True,
        residual_evidence: str = "",
    ) -> str:
        """Design one measured neural counterfactual from task evidence, not a template."""
        mode = "custom, task-invented neural architecture" if require_custom else "neural architecture"
        track = "custom_neural" if require_custom else "established_neural"
        revision = (
            "\n# Residual error analysis from the previous custom-network measurement\n"
            + residual_evidence[-4000:]
            + "\nExplain which validation samples/behaviors the previous custom network solved "
            "or failed on relative to the conventional control, and design THIS revision around "
            "the residual errors rather than repeating the same computation graph.\n"
            if residual_evidence
            else ""
        )
        print(
            f"TechniqueAgent: Designing a {mode} experiment after measured model-family saturation.",
            flush=True,
        )
        prompt = (
            f"{analysis.prompt_context(11000)}\n\n"
            f"{self._council_context(10000)}\n\n"
            f"Measured control plan:\n{parent_plan[:4500]}\n\n"
            f"Control score: {score} ({analysis.direction}).\n"
            f"Plateau evidence:\n{plateau_evidence[-3000:]}\n\n"
            f"Previously measured alternatives:\n{measured_alternatives[-5000:]}\n\n"
            f"{revision}"
            "ARCHITECTURE-LAB INSTRUCTIONS:\n"
            "1. Re-inspect the observed modality, sample count, feature geometry, missingness, "
            "cardinality, dependency structure, metric, CPU/GPU inventory, and time budget.\n"
            "2. Explain which residual error or interaction the conventional control cannot model "
            "well and turn that into a falsifiable neural inductive bias.\n"
            "3. Design the computation graph explicitly: input encoders, tensor shapes, learned "
            "transformations, interaction/routing mechanism, residual paths, output head, loss, "
            "regularization, optimizer, batching, early stopping, and inference.\n"
            "4. Build the primary predictor as a self-contained PyTorch `nn.Module` from primitive "
            "layers and tensor operations. Do not use TabNet, FT-Transformer, TabTransformer, or a "
            "renamed plain MLP as the proposal. It may use a plain MLP as an ablation/control only.\n"
            "5. The architecture must be derived adaptively from the observed evidence; do not copy "
            "a canned modality template. Keep parameter count and training bounded, work on CPU, and "
            "optionally accelerate on an available GPU.\n"
            "6. Use the exact shared split and metric. Compare the custom network against the measured "
            "parent and at least one architecture ablation. Stop if the custom mechanism does not add "
            "repeatable validation value.\n"
            "7. You may combine known primitive operations in a new task-specific way, but do not claim "
            "the design is globally unprecedented. State component-level prior-art searches and "
            "ablations that would be needed to support a novelty claim.\n\n"
            f"Return one concrete executable plan for a {mode}. Include an `Architecture specification:` "
            "section and an `Ablation and stopping rule:` section."
        )
        try:
            response = call_llm(
                "You are the architecture-invention lead in a senior ML lab. You derive compact trainable computation graphs from measured task evidence and test them skeptically.",
                prompt,
                model=self.model_name,
                temperature=0.3,
            )
            cleaned = re.sub(r"(?s)<thinking>.*?</thinking>", "", response).strip()
            if len(cleaned.split()) < 20:
                raise ValueError("architecture plan was empty or too short")
            return (
                f"Architecture exploration track: {track}\n"
                + cleaned[:7000]
            )
        except Exception as exc:
            return (
                f"Architecture exploration track: {track}\n"
                "Construct a resource-bounded, task-tailored PyTorch `nn.Module` after inspecting "
                "the actual input modality, tensor geometry, sample count, feature types, and hardware. "
                "Use primitive layers and tensor operations to encode the strongest observed inductive "
                "bias, with explicit input projections, a learned interaction or gating module, residual "
                "paths where justified, and a metric-appropriate output head. Do not instantiate TabNet, "
                "FT-Transformer, TabTransformer, or a renamed plain MLP as the "
                "primary model. Keep a plain MLP and the measured parent as controls. Train with bounded "
                "batches, regularization, deterministic seeds, early stopping, and CPU fallback under the "
                "same validation protocol. Ablate the custom interaction module and retain the architecture "
                "only if it produces a repeatable score improvement. Do not claim global novelty; record "
                "which component-level literature searches would be needed. "
                f"Planning service note: {exc}"
            )[:7500]

    def propose_transfer_exploration(
        self,
        analysis: TaskAnalysis,
        parent_plan: str,
        score: float,
        *,
        measured_alternatives: str = "",
        plateau_evidence: str = "",
        residual_evidence: str = "",
    ) -> str:
        """Design one measured transfer-learning experiment using a pretrained backbone."""
        track = "established_neural"
        revision = (
            "\n# Residual error analysis from the previous measurement\n"
            + residual_evidence[-4000:]
            + "\nExplain which validation samples/behaviors the previous network solved "
            "or failed on, and design THIS revision around the residual errors.\n"
            if residual_evidence
            else ""
        )
        print(
            f"TechniqueAgent: Designing a transfer-learning experiment after measured model-family saturation.",
            flush=True,
        )
        prompt = (
            f"{analysis.prompt_context(11000)}\n\n"
            f"{self._council_context(10000)}\n\n"
            f"Measured control plan:\n{parent_plan[:4500]}\n\n"
            f"Control score: {score} ({analysis.direction}).\n"
            f"Plateau evidence:\n{plateau_evidence[-3000:]}\n\n"
            f"Previously measured alternatives:\n{measured_alternatives[-5000:]}\n\n"
            f"{revision}"
            "TRANSFER-LAB INSTRUCTIONS:\n"
            "1. Re-inspect the observed modality (image, text, audio), sample count, metric, CPU/GPU inventory, and time budget.\n"
            "2. Select a suitable pretrained foundation backbone (e.g., EfficientNet, ResNet, ViT, BERT, Wav2Vec).\n"
            "3. Design the computation graph explicitly: load the pretrained backbone, add a custom metric-appropriate output head or attention pooling layer, define the loss, optimizer, batching, early stopping, and inference.\n"
            "4. Build the primary predictor as a self-contained PyTorch `nn.Module` that wraps the pretrained backbone and the custom head. Do not just call a high-level library without explicit custom head/interaction code.\n"
            "5. Ensure the plan includes a fallback if external downloads fail (e.g., attempt fetch, then gracefully skip or fallback to a bundled cache/dummy network without crashing).\n"
            "6. Use the exact shared split and metric. Compare the transfer network against the measured parent.\n"
            "7. Do not plagiarize known competition solutions; design an honest transfer counterfactual.\n\n"
            f"Return one concrete executable plan for a pretrained-fine-tune transfer architecture. Include an `Architecture specification:` "
            "section and an `Ablation and stopping rule:` section."
        )
        try:
            response = call_llm(
                "You are a transfer-learning lead in a senior ML lab. You fine-tune pretrained backbones on measured task evidence and test them skeptically.",
                prompt,
                model=self.model_name,
                temperature=0.3,
            )
            cleaned = re.sub(r"(?s)<thinking>.*?</thinking>", "", response).strip()
            if len(cleaned.split()) < 20:
                raise ValueError("architecture plan was empty or too short")
            return (
                f"Architecture exploration track: {track}\n"
                + cleaned[:7000]
            )
        except Exception as exc:
            return (
                f"Architecture exploration track: {track}\n"
                "Construct a resource-bounded transfer-learning PyTorch `nn.Module`. Wrap a suitable "
                "pretrained backbone (e.g. EfficientNet, BERT) and add a custom output head or attention "
                "layer. Train with bounded batches, regularization, deterministic seeds, early stopping, and CPU fallback under the "
                "same validation protocol. Compare the transfer network against the measured parent. "
                "Include a graceful fallback if pretrained weights cannot be downloaded. "
                f"Planning service note: {exc}"
            )[:7500]

    def propose_follow_up(
        self,
        analysis: TaskAnalysis,
        operator: str,
        parent_plan: str,
        score: float,
        diagnostics: str = "",
        search_context: str = "",
        avoid_families: str = "",
    ) -> str:
        """Materialize one score-selected refinement or diversity action."""
        if operator not in {"refine", "diversify"}:
            raise ValueError(f"unsupported follow-up operator: {operator!r}")

        web_insights = ""
        if operator == "diversify":
            web_insights = self.search_for_new_ideas(analysis)

        web_section = f"\n\nWeb Search Discovery Insights:\n{web_insights}" if web_insights else ""
        avoid_section = (
            f"\n\nFAMILY GUARD (hard constraint):\n{avoid_families[:2500]}\n"
            "Your proposal MUST choose a different model family. Start the returned plan with a "
            "`Model family: <name>` line naming the family you select."
            if avoid_families
            else ""
        )

        guidance = {
            "refine": (
                "Preserve the working pipeline and change only the highest-impact feature, "
                "representation, calibration, regularization, or model module suggested by "
                "the measured run. Include an explicit local comparison with the parent."
            ),
            "diversify": (
                "Build a genuinely NEW and complementary representation or model family that HAS NOT "
                "been attempted in previous runs. Use web search literature insights if available. Do NOT repeat "
                "already executed model families. Preserve the same split, metric, input paths, and "
                "output format so the result is directly comparable and mergeable."
            ),
        }
        print(
            f"TechniqueAgent: Materializing {operator} follow-up for score {score}.",
            flush=True,
        )
        prompt = (
            f"{analysis.prompt_context(10000)}\n\n"
            f"{self._council_context(9000)}\n\n"
            f"Measured parent plan:\n{parent_plan[:5000]}\n\n"
            f"Parent local score: {score} ({analysis.direction}).\n"
            f"Bounded execution notes:\n{diagnostics[-3000:]}\n\n"
            f"Recent measured alternatives to avoid repeating:\n{search_context[-2000:]}"
            f"{web_section}"
            f"{avoid_section}\n\n"
            "CRITICAL REASONING INSTRUCTIONS:\n\n"
            "1. THINK BEFORE YOU DECIDE:\n"
            "   Begin your response with a <thinking> ... </thinking> block where you step-by-step:\n"
            "   a. Analyze the parent plan's performance and diagnostics.\n"
            "   b. For `refine`: Identify the highest-impact feature representation, calibration, or model tuning improvement.\n"
            "   c. For `diversify`: Inspect web search literature insights for architectures suitable for this modality. Assess whether proposed new model families actually suit the task and will push validation score beyond current plateaus. Consider BOTH complex SOTA models and simple foundational models (like Logistic Regression or SVM) if they haven't been tried.\n"
            "   d. Verify that no previously attempted model family (from the measured alternatives context) is repeated.\n\n"
            f"2. Selected action: {operator}. {guidance[operator]}\n"
            "   STRICT REQUIREMENT: DO NOT repeat model algorithms already tried. Return only one concrete revised implementation plan."
        )
        try:
            response = call_llm(
                "You are an expert AI ML Scientist materializing selected research experiments around measured baselines.",
                prompt,
                model=self.model_name,
                temperature=0.15,
            )
            cleaned = re.sub(r"(?s)<thinking>.*?</thinking>", "", response).strip()
            if len(cleaned.split()) < 8:
                raise ValueError("follow-up plan was empty or too short")
            return cleaned[:6500]
        except Exception as exc:
            return (
                f"{guidance[operator]} Keep this exact measured parent as the control: "
                f"{parent_plan}. Use a bounded deterministic comparison and retain the parent "
                f"whenever the {operator} candidate does not improve. Planning note: {exc}"
            )[:6500]

    def propose_merge(
        self,
        analysis: TaskAnalysis,
        first_plan: str,
        second_plan: str,
    ) -> str:
        print("TechniqueAgent: Requesting a merge of two strong branches.", flush=True)
        prompt = (
            f"{analysis.prompt_context(9000)}\n\n"
            f"{self._council_context(8000)}\n\n"
            f"Strong approach A:\n{first_plan[:3500]}\n\n"
            f"Strong approach B:\n{second_plan[:3500]}\n\n"
            "CRITICAL REASONING INSTRUCTIONS:\n\n"
            "1. THINK BEFORE YOU DECIDE:\n"
            "   Begin your response with a <thinking> ... </thinking> block where you step-by-step:\n"
            "   a. Analyze the architectures, predictions, and validation scores of Parent A and Parent B.\n"
            "   b. Assess whether combining them (via stacking, weighted blending, rank averaging, or feature concatenation) will yield a performance boost over the stronger parent.\n"
            "   c. Design the exact executable merge pipeline.\n\n"
            "2. PROPOSAL:\n"
            "   Design one executable merge that combines complementary strengths. Include a local comparison against both parents and preserve exact output format."
        )
        try:
            response = call_llm(
                "You combine strong methods only when the merge has a credible performance benefit.",
                prompt,
                model=self.model_name,
                temperature=0.15,
            )
            cleaned = re.sub(r"(?s)<thinking>.*?</thinking>", "", response).strip()
            return cleaned[:7000]
        except Exception:
            return (
                "Combine the two working approaches with a validation-selected blend or consensus, "
                "falling back to the stronger parent if the merged local score does not improve."
            )

    def decide_next_step(
        self,
        analysis: TaskAnalysis,
        nodes_history: list[dict[str, Any]],
        best_node_id: str | None,
        best_score: float | None,
        experiments_remaining: int,
        plateau_state: dict[str, Any] | None = None,
        architecture_coverage: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Analyze current tree search state and executed results, then use LLM to decide
        the optimal next strategic step (merge, tune, refine, diversify, or finalize).
        """
        print(
            f"TechniqueAgent: Evaluating {len(nodes_history)} executed nodes to decide next strategic action.",
            flush=True,
        )

        history_lines = []
        for n in nodes_history:
            nid = n.get("node_id", "")
            op = n.get("operator", "root")
            ntype = n.get("node_type", "")
            status = n.get("status", "")
            score = n.get("score")
            architecture_track = n.get("architecture_track", "other")
            modality_evidence = n.get("modality_ablation_scores")
            plan_summary = n.get("plan_summary", "")
            is_best = " [BEST SCORE SO FAR]" if nid == best_node_id else ""
            score_str = f"{score:.5f}" if score is not None else "failed"
            history_lines.append(
                f"- Node '{nid}' ({ntype}, operator='{op}', architecture_track='{architecture_track}', status='{status}'): score={score_str}{is_best}. Modality evidence: {json.dumps(modality_evidence, default=str)[:900] if modality_evidence else 'none'}. Plan summary: {plan_summary}"
            )

        history_text = "\n".join(history_lines) if history_lines else "No executed nodes yet."

        # The manager normally supplies a tolerance-based plateau analysis. Keep
        # a conservative fallback for direct callers rather than comparing floats
        # for exact equality.
        recent_scores = [
            float(n["score"])
            for n in nodes_history[-4:]
            if n.get("score") is not None
        ]
        if plateau_state is None:
            spread = max(recent_scores) - min(recent_scores) if recent_scores else 0.0
            scale = max((abs(value) for value in recent_scores), default=1.0)
            plateaued = len(recent_scores) >= 2 and spread <= max(1e-6, scale * 5e-4)
            plateau_state = {
                "plateaued": plateaued,
                "recent_score_spread": spread,
                "reason": "local tolerance-based fallback",
            }
        else:
            plateaued = bool(plateau_state.get("plateaued"))
        architecture_coverage = dict(architecture_coverage or {})

        web_insights = ""
        if plateaued or experiments_remaining <= 2:
            web_insights = self.search_for_new_ideas(analysis)

        web_section = f"\n# Web Search Discovery Insights for Task\n{web_insights}\n" if web_insights else ""

        prompt = (
            f"{analysis.prompt_context(10000)}\n\n"
            f"{self._council_context(8000)}\n\n"
            f"# Execution History\n"
            f"Metric: {analysis.metric} ({analysis.direction})\n"
            f"Best Score so far: {best_score if best_score is not None else 'N/A'} (Node: {best_node_id})\n"
            f"Remaining Main Ideas Budget: {experiments_remaining} (Tuning is FREE and does not consume main budget)\n\n"
            f"Tolerance-based Plateau State: {json.dumps(plateau_state, default=str)}\n"
            f"Measured Architecture Coverage: {json.dumps(architecture_coverage, default=str)}\n\n"
            f"Executed Nodes History:\n{history_text}\n"
            f"{web_section}\n"
            "# Manager Agent Decision Rules\n"
            "1. THINK BEFORE YOU DECIDE: Analyze tolerance-based score progression across all executed nodes. Check if current model family has plateaued and if switching architectures (including a measured custom neural computation graph) will boost score.\n"
            "2. Model Suitability & Performance Assessment: Verify if fixing/tuning the current model actually suits the task or if switching architectures will improve performance.\n"
            "3. STRICT ANTI-REPETITION: DO NOT repeat any model family, algorithm, or pipeline component that has already been attempted in previous nodes. Review the 'plan_summary' in Executed Nodes History carefully to identify what models have been used.\n"
            "4. Tuning a node (hyperparameters, extra layers, or preprocessing) is FREE and does NOT deduct from the main budget.\n"
            "5. If a tuned node fails to improve upon its parent score, it will be automatically stopped and pruned immediately to save budget for high-performing nodes.\n"
            "6. If conventional tuning/refining plateaus and no neural architecture has been measured, prefer `transfer` (if applicable for image/text/audio) or `architect` (for tabular) before another merge. A merge cannot reveal whether a different representation learns missing interactions.\n"
            "7. `architect` means a bounded custom PyTorch nn.Module derived from observed inductive biases. `transfer` means fine-tuning a pretrained backbone (e.g. EfficientNet). Do not claim global novelty without prior-art evidence.\n"
            "8. On multimodal tasks, use measured modality-ablation scores: keep full fusion only when it beats credible single and leave-one-out controls on identical folds. Do not merge modalities merely because they are available.\n"
            "9. Use `merge` only when at least two independently strong, genuinely complementary representations exist and architecture coverage is not the larger evidence gap.\n\n"
            "Select the single best strategic action to execute next:\n"
            "1. `merge`: Combine/blend/ensemble two or more top or complementary nodes (e.g. Node A and Node B) mid-process or at the end to achieve a score breakthrough.\n"
            "2. `tune`: Perform focused tuning (hyperparameters, layers, preprocessing) on a specific promising node.\n"
            "3. `refine`: Perform targeted structural/feature refinement on a specific node.\n"
            "4. `diversify`: Propose a fundamentally NEW model family or SOTA architecture not tried before.\n"
            "5. `architect`: Design and measure a task-tailored neural computation graph from primitive trainable operations.\n"
            "6. `transfer`: Design and measure a pretrained-fine-tune transfer network (if image/text/audio modality).\n"
            "7. `finalize`: Stop search and build final submission/ensemble if scores are saturated.\n\n"
            "Respond strictly with a JSON object containing:\n"
            "{\n"
            '  "thinking": "step-by-step reasoning analysis",\n'
            '  "action": "merge" | "tune" | "refine" | "diversify" | "architect" | "transfer" | "finalize",\n'
            '  "target_node_ids": ["node_id_1", "node_id_2"],\n'
            '  "reasoning": "concise explanation of why this step is chosen over others",\n'
            '  "plan": "concise execution plan for this action"\n'
            "}\n"
        )

        try:
            decision = call_llm_json(
                "You are an expert AI Search Manager orchestrating an adaptive machine learning search tree.",
                prompt,
                model=self.model_name,
                temperature=0.2,
                schema=_DECISION_SCHEMA,
                schema_name="search_decision",
            )
            if isinstance(decision, dict) and "action" in decision:
                action = str(decision.get("action", "")).lower()
                if action in {"merge", "tune", "refine", "diversify", "architect", "transfer", "finalize"}:
                    decision["action"] = action
                    if not isinstance(decision.get("target_node_ids"), list):
                        decision["target_node_ids"] = [best_node_id] if best_node_id else []
                    return dict(decision)
        except Exception as exc:
            print(f"TechniqueAgent: decision call failed; using resilient default: {exc}", flush=True)

        # A parse failure must not silently burn the budget on repeated diversify
        # plans; pick the least costly action that addresses the evidence gap.
        fallback_action = "finalize"
        if experiments_remaining > 0:
            if best_node_id is None:
                fallback_action = "diversify"
            elif not architecture_coverage.get("custom_neural_attempted"):
                fallback_action = "architect"
            else:
                fallback_action = "refine"
        return {
            "action": fallback_action,
            "target_node_ids": [best_node_id] if best_node_id else [],
            "reasoning": "Fallback search decision.",
            "plan": "Continue adaptive search or finalize.",
        }
