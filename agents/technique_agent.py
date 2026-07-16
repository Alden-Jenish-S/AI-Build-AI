import json
import math
import os
import re
from pathlib import Path
from typing import List, Dict, Any
from .llm_utils import call_llm
from .web_search import search_web
from memory_pool.query_tool import query

class TechniqueAgent:
    def __init__(
        self,
        model_name: str = None,
        max_l1_categories: int = 8,
        max_artifact_candidates: int = 5,
    ):
        self.model_name = model_name
        self.max_l1_categories = max(1, int(max_l1_categories))
        self.max_artifact_candidates = max(1, int(max_artifact_candidates))

    def generate_initial_approaches(
        self, task_description: str, count: int = 3
    ) -> List[Dict[str, str]]:
        """
        Suggests a budget-aware number of distinct full modeling approaches.
        Returns a list of dicts: [{'name': '...', 'plan': '...'}]
        """
        count = max(1, min(3, int(count)))
        print(
            f"TechniqueAgent: Generating {count} dynamic initial approaches "
            "based on task description..."
        )
        system_prompt = (
            "You are the Technique Agent. Given a machine learning task description, think from first principles "
            f"and propose {count} distinct, promising, and tailored full modeling approaches (branches) "
            "that make sense for this specific task.\n"
            f"Respond ONLY with a valid JSON list of {count} dictionaries. Each dictionary must have two keys:\n"
            "1. 'name': A short camel_case or snake_case name for the branch (e.g., 'xgboost_with_target_encoding', 'feature_interactions_rf', 'mlp_tabular').\n"
            "2. 'plan': A detailed description of the approach, serving as a strategic direction for the technique selection.\n"
            "Do not reuse stock branch labels like Branch_A_Ensembling, Branch_B_Features, or Branch_C_DeepLearning. "
            "Each root branch must specify a complete pipeline or model family, not a standalone scaler, encoder, imputer, or CV utility. "
            "The branches must be task-specific, complementary, and directly implementable.\n"
            "Do not include any explanation or markdown formatting, just the raw JSON list."
        )
        user_prompt = (
            f"Task Description:\n{task_description}\n\n"
            f"Propose the {count} best approaches in JSON format."
        )
        response = call_llm(system_prompt, user_prompt, model=self.model_name)

        try:
            return self._parse_initial_approaches(response, count)
        except Exception as parse_error:
            print(f"TechniqueAgent WARNING: Failed to parse initial approaches JSON: {parse_error}. Asking LLM to repair the response.")

        repair_system = (
            "You repair malformed JSON for an ML technique planner. "
            f"Return ONLY a valid JSON list of exactly {count} dictionaries, each with non-empty 'name' and 'plan' strings. "
            "Do not invent generic fallback branches; preserve and clean the task-specific ideas from the draft."
        )
        repair_user = f"""
Task Description:
{task_description}

Malformed Draft:
{response}

Return repaired JSON only.
"""
        repaired_response = call_llm(repair_system, repair_user, model=self.model_name)
        return self._parse_initial_approaches(repaired_response, count)

    def _parse_initial_approaches(
        self, response: str, expected_count: int = 3
    ) -> List[Dict[str, str]]:
        """Parse and validate the LLM-authored initial branch ideas."""
        if "```json" in response:
            json_str = response.split("```json", 1)[1].split("```", 1)[0]
        elif "```" in response:
            json_str = response.split("```", 1)[1].split("```", 1)[0]
        else:
            json_str = response

        approaches = json.loads(json_str.strip())
        if not isinstance(approaches, list) or len(approaches) != expected_count:
            raise ValueError(
                f"expected a JSON list with exactly {expected_count} approaches"
            )

        valid_approaches = []
        seen_names = set()
        for idx, app in enumerate(approaches, start=1):
            if not isinstance(app, dict):
                raise ValueError(f"approach {idx} is not a dictionary")
            name = str(app.get("name", "")).strip()
            plan = str(app.get("plan", "")).strip()
            if not name or not plan:
                raise ValueError(f"approach {idx} must include non-empty name and plan")

            normalized_name = re.sub(r"[^a-zA-Z0-9_]+", "_", name).strip("_").lower()
            if not normalized_name:
                normalized_name = f"llm_branch_{idx}"
            if normalized_name in seen_names:
                raise ValueError(f"duplicate branch name: {normalized_name}")
            seen_names.add(normalized_name)

            valid_approaches.append({"name": normalized_name, "plan": plan})

        return valid_approaches

    def generate_follow_up_approach(
        self,
        operator: str,
        task_description: str,
        parent_code: str,
        parent_result: dict,
        global_memory_context: dict,
    ) -> Dict[str, Any]:
        """Materialize one previously scheduled operator slot on demand."""
        if operator not in {"refine", "tune", "diversify"}:
            raise ValueError(f"unsupported lazy operator: {operator!r}")
        instructions = {
            "refine": "Change only the highest-impact code block and preserve the rest.",
            "tune": "Keep the architecture and define a compact, pruned tuning experiment.",
            "diversify": "Create a sound candidate whose prediction errors should be less correlated with the parent.",
        }
        user_prompt = (
            f"Task:\n{task_description}\n\n"
            f"Selected operator: {operator}\n"
            f"Operator contract: {instructions[operator]}\n\n"
            f"Parent result:\n{json.dumps(parent_result or {}, indent=2, default=str)}\n\n"
            f"Relevant prior experiments:\n"
            f"{json.dumps(global_memory_context or {}, indent=2, default=str)}\n\n"
            f"Parent code:\n```python\n{parent_code}\n```\n"
        )
        response = call_llm(
            "You are an ML search-policy agent. Materialize exactly one already-selected "
            "operator into a concrete task-specific experiment. Return ONLY one JSON object "
            "with fields name, plan, operator, priority. Priority must be numeric in [-0.1, 0.1].",
            user_prompt,
            model=self.model_name,
        )
        if "```json" in response:
            response = response.split("```json", 1)[1].split("```", 1)[0]
        elif "```" in response:
            response = response.split("```", 1)[1].split("```", 1)[0]
        item = json.loads(response.strip())
        if not isinstance(item, dict) or item.get("operator") != operator:
            raise ValueError("lazy proposal did not preserve the selected operator")
        name = re.sub(
            r"[^a-zA-Z0-9_]+", "_", str(item.get("name", "")).strip()
        ).strip("_").lower()
        plan = str(item.get("plan", "")).strip()
        if not name or not plan:
            raise ValueError("lazy proposal is incomplete")
        try:
            priority = max(-0.1, min(0.1, float(item.get("priority", 0.0))))
        except (TypeError, ValueError):
            priority = 0.0
        return {"name": name, "plan": plan, "operator": operator, "priority": priority}

    @staticmethod
    def _terms(text: str) -> set[str]:
        return {
            token
            for token in re.findall(r"[a-z0-9]+", str(text).lower())
            if len(token) > 2 or token.isdigit()
        }

    def _prefilter_l1(self, l1_index: dict, query_text: str) -> dict:
        """Bound L1 prompt growth using deterministic lexical relevance."""
        query_terms = self._terms(query_text)
        ranked = []
        for position, (category, details) in enumerate(l1_index.items()):
            category_terms = self._terms(
                category.replace("_", " ") + " " + details.get("description", "")
            )
            overlap = len(query_terms & category_terms)
            ranked.append((overlap, -position, category, details))
        ranked.sort(reverse=True)
        return {
            category: details
            for _, _, category, details in ranked[: self.max_l1_categories]
        }

    def _artifact_prior(
        self,
        candidate: dict,
        branch_plan: str,
        total_runs: int,
        preferred_accelerator: str = "cpu",
    ) -> float:
        """Combine lexical fit, empirical reward, and an exploration bonus."""
        branch_terms = self._terms(branch_plan)
        artifact_terms = self._terms(
            " ".join(
                [
                    candidate.get("artifact_id", ""),
                    candidate.get("category", ""),
                    candidate.get("description", ""),
                    json.dumps(candidate.get("interface", {})),
                ]
            ).replace("_", " ")
        )
        lexical = len(branch_terms & artifact_terms) / max(len(branch_terms), 1)
        stats = candidate.get("validation_summary", {})
        runs = max(0, int(stats.get("runs", 0) or 0))
        mean_reward = float(stats.get("mean_reward", 0.0) or 0.0)
        improvement_rate = float(stats.get("improvement_rate", 0.0) or 0.0)
        exploration = math.sqrt(math.log(max(total_runs, 0) + 2.0) / (runs + 1.0))
        capabilities = candidate.get("capabilities") or {}
        raw_supported_accelerators = capabilities.get("supported_accelerators") or []
        if not isinstance(raw_supported_accelerators, (list, tuple, set)):
            raw_supported_accelerators = []
        supported_accelerators = {
            str(item).lower()
            for item in raw_supported_accelerators
        }
        gpu_bonus = 0.0
        if (
            preferred_accelerator in {"cuda", "mps"}
            and capabilities.get("gpu_accelerated") is True
            and (
                not supported_accelerators
                or preferred_accelerator in supported_accelerators
            )
        ):
            gpu_bonus = 0.12
        return (
            lexical
            + 0.35 * mean_reward
            + 0.15 * improvement_rate
            + 0.08 * exploration
            + gpu_bonus
        )

    def generate_follow_up_approaches(
        self,
        task_description: str,
        parent_code: str,
        parent_result: dict,
        global_memory_context: dict,
    ) -> List[Dict[str, Any]]:
        """Generate a small operator portfolio for measured candidate evolution."""
        system_prompt = (
            "You are an ML search-policy agent. Given a working parent pipeline and its measured result, "
            "propose exactly three complementary child experiments. Return ONLY a JSON list. The three "
            "operators must be exactly: 'refine', 'tune', and 'diversify'.\n"
            "- refine: change only the highest-impact code block, preserving the rest.\n"
            "- tune: keep the architecture and use Optuna-style pruning or a compact search space.\n"
            "- diversify: produce predictions likely to be less correlated with the parent for ensembling.\n"
            "Each item must have non-empty 'name', 'plan', 'operator', and numeric 'priority' in [-0.1, 0.1]. "
            "Plans must be concrete, task-specific, leakage-safe, and directly executable."
        )
        user_prompt = f"""
Task:
{task_description}

Parent result:
{json.dumps(parent_result or {}, indent=2, default=str)}

Relevant prior experiments:
{json.dumps(global_memory_context or {}, indent=2, default=str)}

Parent code:
```python
{parent_code}
```

Return the three child experiments as JSON only.
"""
        response = call_llm(system_prompt, user_prompt, model=self.model_name)
        try:
            return self._parse_follow_up_approaches(response)
        except Exception as parse_error:
            print(
                "TechniqueAgent WARNING: Failed to parse follow-up portfolio: "
                f"{parse_error}. Asking LLM to repair it."
            )
        repair = call_llm(
            "Repair the draft into ONLY a JSON list of exactly three dictionaries with unique "
            "operators refine, tune, diversify and fields name, plan, operator, priority.",
            response,
            model=self.model_name,
        )
        return self._parse_follow_up_approaches(repair)

    def _parse_follow_up_approaches(self, response: str) -> List[Dict[str, Any]]:
        if "```json" in response:
            payload = response.split("```json", 1)[1].split("```", 1)[0]
        elif "```" in response:
            payload = response.split("```", 1)[1].split("```", 1)[0]
        else:
            payload = response
        approaches = json.loads(payload.strip())
        if not isinstance(approaches, list) or len(approaches) != 3:
            raise ValueError("expected exactly three follow-up approaches")
        required_operators = {"refine", "tune", "diversify"}
        actual_operators = {str(item.get("operator", "")).strip() for item in approaches}
        if actual_operators != required_operators:
            raise ValueError("follow-up operators must be refine, tune, and diversify")
        result = []
        for index, item in enumerate(approaches, start=1):
            name = re.sub(
                r"[^a-zA-Z0-9_]+", "_", str(item.get("name", "")).strip()
            ).strip("_").lower()
            plan = str(item.get("plan", "")).strip()
            operator = str(item["operator"]).strip()
            if not name or not plan:
                raise ValueError(f"follow-up approach {index} is incomplete")
            try:
                priority = max(-0.1, min(0.1, float(item.get("priority", 0.0))))
            except (TypeError, ValueError):
                priority = 0.0
            result.append(
                {"name": name, "plan": plan, "operator": operator, "priority": priority}
            )
        return result

    def run(
        self,
        task_description: str,
        branch_plan: str,
        global_memory_context: dict,
        l1_index: dict,
        allowed_scopes: set[str] | None = None,
        available_accelerators: set[str] | None = None,
        preferred_accelerator: str = "cpu",
    ) -> Dict[str, Any]:
        """
        1. Decides if relevant techniques are in the Memory Pool.
        2. If not, searches the web, pulls new technique info, and flags for L2 build.
        
        Args:
            task_description: Clean task description for web search queries (no internal bias)
            branch_plan: Internal branch strategy/bias for L1 category selection only
            global_memory_context: Sibling/parent context from GlobalMemory
            l1_index: The L1 category index
        """
        use_pool = os.environ.get("ABLATION_USE_POOL", "1") != "0"
        if not use_pool or not l1_index:
            print("TechniqueAgent: Memory pool disabled or empty. Generating a new technique outline.")
            return self._bootstrap_from_web(task_description, branch_plan)

        # Use both the branch direction and measured lineage context. The latter
        # prevents repeating failed sibling experiments and enables evidence-based reuse.
        system_prompt = (
            "You are the Technique Agent. Given the strategic direction below, select the 1-2 MOST RELEVANT "
            "categories from the list that align with this direction.\n"
            "CONSTRAINTS:\n"
            "- Select AT MOST 2 categories.\n"
            "- Do NOT select all categories. Focus only on what the strategic direction calls for.\n"
            "- Output ONLY a comma-separated list of category names from the available list. No explanation."
        )
        
        visible_l1 = self._prefilter_l1(
            l1_index, task_description + "\n" + branch_plan
        )
        l1_summary = ""
        for cat, details in visible_l1.items():
            l1_summary += f"- {cat}: {details['description']}\n"
            
        user_prompt = f"""
Strategic Direction:
{branch_plan}

Measured Parent/Sibling Context:
{json.dumps(global_memory_context or {}, indent=2, default=str)}

Execution Resources:
Available accelerators: {sorted(available_accelerators or {'cpu'})}
Preferred accelerator: {preferred_accelerator}

Available Categories:
{l1_summary}

Which 1-2 categories best match the strategic direction? Output a comma-separated list.
"""
        response = call_llm(system_prompt, user_prompt, model=self.model_name).strip()
        selected_categories = [
            c.strip() for c in response.split(",") if c.strip() in visible_l1
        ]
        
        # Bug 4: Limit to max 2 categories even if LLM returns more
        selected_categories = selected_categories[:2]
        
        if not selected_categories:
            print("TechniqueAgent: No valid L1 category matched the branch plan. Initiating web search...")
            return self._bootstrap_from_web(task_description, branch_plan)
            
        print(f"TechniqueAgent: Selected L1 categories: {selected_categories} (for branch: {branch_plan[:60]}...)")
        
        # Query L1 categories to get L2 candidates
        l2_candidates = []
        for cat in selected_categories:
            res = query(cat)
            verified_artifacts = [
                artifact for artifact in res.get("artifacts", [])
                if artifact.get("verified") is True
                and (
                    not allowed_scopes
                    or artifact.get("scope") in allowed_scopes
                )
            ]
            l2_candidates.extend(verified_artifacts)

        total_runs = sum(
            int(item.get("validation_summary", {}).get("runs", 0) or 0)
            for item in l2_candidates
        )
        for candidate in l2_candidates:
            candidate["retrieval_prior"] = self._artifact_prior(
                candidate,
                branch_plan,
                total_runs,
                preferred_accelerator=preferred_accelerator,
            )
        l2_candidates.sort(
            key=lambda item: item.get("retrieval_prior", 0.0), reverse=True
        )
        l2_candidates = l2_candidates[: self.max_artifact_candidates]
            
        system_prompt = (
            "You are the Technique Agent. Evaluate if any candidate artifact is a high-quality match for the task requirements. "
            "Default to novelty unless a pool artifact is a strong, direct fit.\n"
            "Choose an artifact ONLY if it satisfies ALL checks:\n"
            "1. It directly implements the branch direction rather than merely belonging to the same broad category.\n"
            "2. Its interface can plausibly be wired into the current tabular task without redesign.\n"
            "3. It is more useful than generating a tailored technique for this branch.\n"
            "If any check is uncertain, output ONLY 'web_search'.\n"
            "If one artifact clearly passes all checks, output ONLY its artifact_id."
        )
        
        candidates_str = ""
        for cand in l2_candidates:
            if "error" not in cand:
                empirical = cand.get("validation_summary", {})
                candidates_str += (
                    f"- ID: {cand['artifact_id']} (Category: {cand['category']}): "
                    f"{cand['description']} | scope={cand.get('scope')} | "
                    f"retrieval_prior={cand.get('retrieval_prior', 0.0):.4f} | "
                    f"resources={json.dumps(cand.get('resource_profile', {}))} | "
                    f"capabilities={json.dumps(cand.get('capabilities', {}))} | "
                    f"empirical_validation={json.dumps(empirical)} | "
                    f"pitfalls={cand.get('known_pitfalls', [])}\n"
                )
        
        # If no candidates exist at all, skip LLM call and go straight to web search
        if not candidates_str.strip():
            decision = "web_search"
        else:
            user_prompt = f"""
Task Description:
{task_description}

Strategic Direction:
{branch_plan}

Measured Parent/Sibling Context:
{json.dumps(global_memory_context or {}, indent=2, default=str)}

Candidate Artifacts in Memory Pool:
{candidates_str}

Decide: Output either one artifact_id from the list, or 'web_search'.
"""
            decision = call_llm(system_prompt, user_prompt, model=self.model_name).strip()
        
        # Check if decision matches an existing candidate
        matching_cand = [c for c in l2_candidates if c.get("artifact_id") == decision]
        
        if matching_cand:
            # Memory Pool Hit
            selected_candidate = matching_cand[0]
            artifact_id = selected_candidate["artifact_id"]
            cat = selected_candidate["category"]
            card = query(cat, artifact_id)
            print(f"TechniqueAgent: Memory Pool Hit: {artifact_id}")
            return {
                "status": "pool_hit",
                "artifact_id": artifact_id,
                "category": cat,
                "scope": selected_candidate.get("scope"),
                "retrieval_prior": selected_candidate.get("retrieval_prior"),
                "plan": f"Use memory pool artifact {artifact_id} from category {cat}.",
                "model_card": card
            }
        else:
            # Memory Pool Miss -> Web Search
            print("TechniqueAgent: Memory Pool Miss. Initiating web search...")
            return self._bootstrap_from_web(task_description, branch_plan)

    def _bootstrap_from_web(self, task_description: str, branch_plan: str) -> Dict[str, Any]:
        """Searches or synthesizes a new technique outline for later L2 building."""
        # Bug 3 fix: Use clean task_description for query generation, NOT branch_plan alone
        query_prompt_sys = (
            "You are an ML research scientist. Write a short, precise web search query "
            "to find state-of-the-art tabular machine learning techniques, custom loss functions, "
            "or advanced neural architectures suitable for the given task and planned technique.\n"
            "CRITICAL: Do NOT include Kaggle file names (like 'train.csv', 'test.csv', 'sample_submission.csv') or generic words like 'notebook'. "
            "Query for scientific methodologies, architectures, or specific algorithms.\n"
            "Output ONLY the search query string, nothing else."
        )
        query_prompt_user = (
            f"Task Description:\n{task_description}\n\n"
            f"Planned Technique to Implement:\n{branch_plan}\n\n"
            "Write only the query string."
        )
        search_query = call_llm(query_prompt_sys, query_prompt_user, model=self.model_name).strip().replace('"', '')

        print(f"TechniqueAgent: Running web search for query: '{search_query}'")
        search_results = search_web(search_query)

        if not search_results or "error" in search_results:
            print("Web search failed or blocked. Using LLM internal knowledge fallback...")
            search_results = (
                "No web pages returned. Fallback to LLM internal knowledge base. "
                "Provide a robust, original tabular ML technique script tailored to the planned technique."
            )

        build_prompt_sys = (
            "You are the Technique Agent. Review the search results and outline ONE new reusable tabular ML technique "
            "we should implement. It should be meaningfully different from generic memory-pool defaults when possible. "
            "Provide: 1) Chosen Category, 2) a unique artifact_id (lowercase_underscores), "
            "3) a description, 4) the raw python code implementing it, 5) the interface entrypoint."
        )
        build_prompt_user = f"""
Search Results:
{search_results}

Task Description:
{task_description}

Planned Technique:
{branch_plan}

Outline the new technique and return the raw source logic for our builder.
"""
        new_technique_outline = call_llm(build_prompt_sys, build_prompt_user, model=self.model_name)

        return {
            "status": "pool_miss",
            "artifact_id": None,
            "category": "web_search_fallback",
            "plan": f"Bootstrap new technique from web search: {search_query}",
            "search_results": search_results,
            "raw_outline": new_technique_outline
        }
