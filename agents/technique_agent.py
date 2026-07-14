import json
import os
import re
from pathlib import Path
from typing import List, Dict, Any
from .llm_utils import call_llm
from .web_search import search_web
from memory_pool.query_tool import query

class TechniqueAgent:
    def __init__(self, model_name: str = None):
        self.model_name = model_name

    def generate_initial_approaches(self, task_description: str) -> List[Dict[str, str]]:
        """
        Suggests 3 distinct machine learning approaches tailored to the task description.
        Returns a list of dicts: [{'name': '...', 'plan': '...'}]
        """
        print("TechniqueAgent: Generating 3 dynamic initial approaches based on task description...")
        system_prompt = (
            "You are the Technique Agent. Given a machine learning task description, think from first principles "
            "and propose 3 distinct, promising, and tailored modeling or preprocessing approaches (branches) "
            "that make sense for this specific task.\n"
            "Respond ONLY with a valid JSON list of 3 dictionaries. Each dictionary must have two keys:\n"
            "1. 'name': A short camel_case or snake_case name for the branch (e.g., 'xgboost_with_target_encoding', 'feature_interactions_rf', 'mlp_tabular').\n"
            "2. 'plan': A detailed description of the approach, serving as a strategic direction for the technique selection.\n"
            "Do not reuse stock branch labels like Branch_A_Ensembling, Branch_B_Features, or Branch_C_DeepLearning. "
            "The branches must be task-specific, complementary, and directly implementable.\n"
            "Do not include any explanation or markdown formatting, just the raw JSON list."
        )
        user_prompt = f"Task Description:\n{task_description}\n\nPropose the 3 best approaches in JSON format."
        response = call_llm(system_prompt, user_prompt, model=self.model_name)

        try:
            return self._parse_initial_approaches(response)
        except Exception as parse_error:
            print(f"TechniqueAgent WARNING: Failed to parse initial approaches JSON: {parse_error}. Asking LLM to repair the response.")

        repair_system = (
            "You repair malformed JSON for an ML technique planner. "
            "Return ONLY a valid JSON list of exactly 3 dictionaries, each with non-empty 'name' and 'plan' strings. "
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
        return self._parse_initial_approaches(repaired_response)

    def _parse_initial_approaches(self, response: str) -> List[Dict[str, str]]:
        """Parse and validate the LLM-authored initial branch ideas."""
        if "```json" in response:
            json_str = response.split("```json", 1)[1].split("```", 1)[0]
        elif "```" in response:
            json_str = response.split("```", 1)[1].split("```", 1)[0]
        else:
            json_str = response

        approaches = json.loads(json_str.strip())
        if not isinstance(approaches, list) or len(approaches) != 3:
            raise ValueError("expected a JSON list with exactly 3 approaches")

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

    def run(self, task_description: str, branch_plan: str, global_memory_context: dict, l1_index: dict) -> Dict[str, Any]:
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

        # Bug 4 fix: Use branch_plan for category selection with tighter constraints
        system_prompt = (
            "You are the Technique Agent. Given the strategic direction below, select the 1-2 MOST RELEVANT "
            "categories from the list that align with this direction.\n"
            "CONSTRAINTS:\n"
            "- Select AT MOST 2 categories.\n"
            "- Do NOT select all categories. Focus only on what the strategic direction calls for.\n"
            "- Output ONLY a comma-separated list of category names from the available list. No explanation."
        )
        
        l1_summary = ""
        for cat, details in l1_index.items():
            l1_summary += f"- {cat}: {details['description']}\n"
            
        user_prompt = f"""
Strategic Direction:
{branch_plan}

Available Categories:
{l1_summary}

Which 1-2 categories best match the strategic direction? Output a comma-separated list.
"""
        response = call_llm(system_prompt, user_prompt, model=self.model_name).strip()
        selected_categories = [c.strip() for c in response.split(",") if c.strip() in l1_index]
        
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
            ]
            l2_candidates.extend(verified_artifacts)
            
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
                candidates_str += f"- ID: {cand['artifact_id']} (Category: {cand['category']}): {cand['description']}\n"
        
        # If no candidates exist at all, skip LLM call and go straight to web search
        if not candidates_str.strip():
            decision = "web_search"
        else:
            user_prompt = f"""
Task Description:
{task_description}

Strategic Direction:
{branch_plan}

Candidate Artifacts in Memory Pool:
{candidates_str}

Decide: Output either one artifact_id from the list, or 'web_search'.
"""
            decision = call_llm(system_prompt, user_prompt, model=self.model_name).strip()
        
        # Check if decision matches an existing candidate
        matching_cand = [c for c in l2_candidates if c.get("artifact_id") == decision]
        
        if matching_cand:
            # Memory Pool Hit
            artifact_id = matching_cand[0]["artifact_id"]
            cat = matching_cand[0]["category"]
            card = query(cat, artifact_id)
            print(f"TechniqueAgent: Memory Pool Hit: {artifact_id}")
            return {
                "status": "pool_hit",
                "artifact_id": artifact_id,
                "category": cat,
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
