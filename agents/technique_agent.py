"""LLM planning for direct task implementations."""

from __future__ import annotations

import re
import json
from typing import Any
from .llm_utils import call_llm
from .task_analyzer import TaskAnalysis


class TechniqueAgent:
    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name

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

    def search_for_new_ideas(self, analysis: TaskAnalysis, query_extra: str = "") -> str:
        """Query web search for competitive ML strategies and novel literature ideas tailored to this task."""
        try:
            from .web_search import search_web
            prompt = (
                f"{analysis.prompt_context(4000)}\n\n"
                "Based on the task description and data properties above, formulate a highly effective "
                "web search query to find state-of-the-art machine learning architectures, novel literature, "
                f"or general ML strategies specific to this problem type. The target metric is {analysis.metric}.\n"
                f"Additional context: {query_extra}\n\n"
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
            "   d. Model Size & Complexity vs Customization: Note that smaller/lighter custom architectures with tailored modules, custom augmentations, or domain features often outperform heavy generic off-the-shelf pre-trained models. Reason through the right tradeoff for this dataset.\n"
            f"   e. Architecture Progression: PLAN 1 MUST be a fast, simple foundational baseline model that are applicable to the given task to establish a strong baseline score. Subsequent plans (PLAN 2, etc.) should progressively move to more complex or SOTA architectures (e.g., tree-based ensembles, neural networks) if applicable.\n\n"
            "2. PROPOSAL REQUIREMENTS:\n"
            f"   Propose {count} executable approaches for this task. Each approach must explain how to load observed paths dynamically from input/, build an honest local validation score matching {analysis.metric} ({analysis.direction}), fit/train, tune, and write the requested deliverable.\n"
            "   STRICT REQUIREMENT: Use a materially DIFFERENT model family / algorithm in each plan. "
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
                "Implement a second, materially different algorithm suitable for the observed data. "
                "Tune a small high-impact parameter set with deterministic splits or a transparent "
                "unsupervised proxy, refit on all useful data, and write the exact requested output."
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
        return plans[:count]

    def propose_tuning(
        self,
        analysis: TaskAnalysis,
        plan: str,
        score: float,
        diagnostics: str = "",
    ) -> str:
        """Turn a successful implementation into one focused improvement plan."""
        print(f"TechniqueAgent: Requesting focused tuning for score {score}.", flush=True)
        prompt = (
            f"{analysis.prompt_context(10000)}\n\n"
            f"Current plan:\n{plan[:5000]}\n\n"
            f"Current local score: {score} ({analysis.direction}).\n"
            f"Run notes:\n{diagnostics[-3000:]}\n\n"
            "CRITICAL REASONING INSTRUCTIONS:\n\n"
            "1. THINK BEFORE YOU DECIDE:\n"
            "   Begin your response with a <thinking> ... </thinking> block where you step-by-step:\n"
            "   a. Evaluate the current plan, local score, and execution diagnostics.\n"
            "   b. Model Suitability & Performance Assessment: Verify if this current model architecture actually suits the task given and data properties. Assess whether focused tuning (hyperparameters, layer depth, feature engineering/scaling, regularization) will realistically improve validation performance.\n"
            "   c. Determine the highest-leverage tuning changes without switching the core algorithm family.\n\n"
            "2. PROPOSAL:\n"
            "   Propose one focused improvement to this working model: tune hyperparameter settings, add feature engineering/scaling modules, or tune layer structure. Preserve the core algorithm, working data paths, and output behavior. Return the revised implementation plan after your <thinking> block."
        )
        try:
            response = call_llm(
                "You are an expert AI ML Optimization Engineer improving working models without destabilizing them.",
                prompt,
                model=self.model_name,
                temperature=0.15,
            )
            cleaned = re.sub(r"(?s)<thinking>.*?</thinking>", "", response).strip()
            return cleaned[:6000]
        except Exception as exc:
            return (
                f"Keep the working implementation described here: {plan}. Perform a small, "
                f"bounded hyperparameter search around its current values and retain the best "
                f"local setting. Planning service note: {exc}"
            )[:6000]

    def propose_follow_up(
        self,
        analysis: TaskAnalysis,
        operator: str,
        parent_plan: str,
        score: float,
        diagnostics: str = "",
        search_context: str = "",
    ) -> str:
        """Materialize one score-selected refinement or diversity action."""
        if operator not in {"refine", "diversify"}:
            raise ValueError(f"unsupported follow-up operator: {operator!r}")

        web_insights = ""
        if operator == "diversify":
            web_insights = self.search_for_new_ideas(analysis)

        web_section = f"\n\nWeb Search Discovery Insights:\n{web_insights}" if web_insights else ""

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
            f"Measured parent plan:\n{parent_plan[:5000]}\n\n"
            f"Parent local score: {score} ({analysis.direction}).\n"
            f"Bounded execution notes:\n{diagnostics[-3000:]}\n\n"
            f"Recent measured alternatives to avoid repeating:\n{search_context[-2000:]}"
            f"{web_section}\n\n"
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
            plan_summary = n.get("plan_summary", "")
            is_best = " [BEST SCORE SO FAR]" if nid == best_node_id else ""
            score_str = f"{score:.5f}" if score is not None else "failed"
            history_lines.append(
                f"- Node '{nid}' ({ntype}, operator='{op}', status='{status}'): score={score_str}{is_best}. Plan summary: {plan_summary}"
            )

        history_text = "\n".join(history_lines) if history_lines else "No executed nodes yet."

        # Check if score progress has plateaued or if we need web search for fresh competitive ideas
        recent_scores = [n.get("score") for n in nodes_history[-4:] if n.get("score") is not None]
        plateaued = len(recent_scores) >= 2 and len(set(recent_scores)) <= 1

        web_insights = ""
        if plateaued or experiments_remaining <= 2:
            web_insights = self.search_for_new_ideas(analysis)

        web_section = f"\n# Web Search Discovery Insights for Task\n{web_insights}\n" if web_insights else ""

        prompt = (
            f"{analysis.prompt_context(10000)}\n\n"
            f"# Execution History\n"
            f"Metric: {analysis.metric} ({analysis.direction})\n"
            f"Best Score so far: {best_score if best_score is not None else 'N/A'} (Node: {best_node_id})\n"
            f"Remaining Main Ideas Budget: {experiments_remaining} (Tuning is FREE and does not consume main budget)\n\n"
            f"Executed Nodes History:\n{history_text}\n"
            f"{web_section}\n"
            "# Manager Agent Decision Rules\n"
            "1. THINK BEFORE YOU DECIDE: Analyze score progression across all executed nodes. Check if current model family has plateaued and if switching architectures (SOTA models, or even simple foundational baselines) or merging will boost score.\n"
            "2. Model Suitability & Performance Assessment: Verify if fixing/tuning the current model actually suits the task or if switching architectures will improve performance.\n"
            "3. STRICT ANTI-REPETITION: DO NOT repeat any model family, algorithm, or pipeline component that has already been attempted in previous nodes. Review the 'plan_summary' in Executed Nodes History carefully to identify what models have been used.\n"
            "4. Tuning a node (hyperparameters, extra layers, or preprocessing) is FREE and does NOT deduct from the main budget.\n"
            "5. If a tuned node fails to improve upon its parent score, it will be automatically stopped and pruned immediately to save budget for high-performing nodes.\n"
            "6. If single-model tuning/refining plateaus or doesn't improve scores, select `merge` to combine two or more complementary strong nodes, or `diversify` using web search insights to try a genuinely novel model family (ranging from simple to complex).\n\n"
            "Select the single best strategic action to execute next:\n"
            "1. `merge`: Combine/blend/ensemble two or more top or complementary nodes (e.g. Node A and Node B) mid-process or at the end to achieve a score breakthrough.\n"
            "2. `tune`: Perform focused tuning (hyperparameters, layers, preprocessing) on a specific promising node.\n"
            "3. `refine`: Perform targeted structural/feature refinement on a specific node.\n"
            "4. `diversify`: Propose a fundamentally NEW model family or SOTA architecture not tried before.\n"
            "5. `finalize`: Stop search and build final submission/ensemble if scores are saturated.\n\n"
            "Respond strictly with a JSON object containing:\n"
            "{\n"
            '  "thinking": "step-by-step reasoning analysis",\n'
            '  "action": "merge" | "tune" | "refine" | "diversify" | "finalize",\n'
            '  "target_node_ids": ["node_id_1", "node_id_2"],\n'
            '  "reasoning": "concise explanation of why this step is chosen over others",\n'
            '  "plan": "concise execution plan for this action"\n'
            "}\n"
        )

        try:
            response = call_llm(
                "You are an expert AI Search Manager orchestrating an adaptive machine learning search tree.",
                prompt,
                model=self.model_name,
                temperature=0.2,
            )
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                decision = json.loads(json_match.group(0))
                if isinstance(decision, dict) and "action" in decision:
                    action = str(decision.get("action", "")).lower()
                    if action in {"merge", "tune", "refine", "diversify", "finalize"}:
                        decision["action"] = action
                        if not isinstance(decision.get("target_node_ids"), list):
                            decision["target_node_ids"] = [best_node_id] if best_node_id else []
                        return decision
        except Exception as exc:
            print(f"TechniqueAgent: decision call failed; using resilient default: {exc}", flush=True)

        return {
            "action": "diversify" if experiments_remaining > 0 else "finalize",
            "target_node_ids": [best_node_id] if best_node_id else [],
            "reasoning": "Fallback search decision.",
            "plan": "Continue adaptive search or finalize.",
        }
