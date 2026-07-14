import os
import json
import sys
import subprocess
from pathlib import Path
from typing import Dict, Any
from agents.llm_utils import call_llm

class L2Builder:
    def __init__(self, project_root: Path, model_name: str = None, venv_path: str = None):
        self.project_root = project_root
        self.model_name = model_name
        self.l1_path = project_root / "memory_pool" / "l1_index.json"
        self.l2_store = project_root / "memory_pool" / "l2_store"
        self.verifier_path = project_root / "memory_pool" / "builder" / "sandbox_verifier.py"
        
        import sys
        resolved_path = str(Path(venv_path or "./.venv/bin/python").resolve())
        # Check if the resolved venv python is fully functional
        use_fallback = True
        if Path(resolved_path).exists():
            try:
                res = subprocess.run([resolved_path, "-c", "import sys; print('ok')"], capture_output=True, text=True, timeout=5)
                if res.returncode == 0 and "ok" in res.stdout:
                    use_fallback = False
            except Exception:
                pass
                
        if use_fallback:
            print(f"L2Builder WARNING: Specified python path '{resolved_path}' is invalid or non-functional. Falling back to active running interpreter: {sys.executable}")
            self.venv_python = sys.executable
        else:
            self.venv_python = resolved_path

    def build_from_source(self, source_name: str, source_content: str, commit: bool = True, target_dir: Path = None) -> tuple:
        """
        Takes raw source content, asks LLM to classify it, extract model card + code,
        saves them, and runs sandbox_verifier.
        """
        # Load current L1 index
        with open(self.l1_path, 'r', encoding='utf-8') as f:
            l1_index = json.load(f)
            
        l1_list_str = "\n".join([f"- {cat}: {details['description']}" for cat, details in l1_index.items()])
        
        system_prompt = (
            "You are the L2 Artifact Builder. Your goal is to convert a raw source snippet "
            "into a single, clean, self-contained tabular ML utility function (.py file) and its metadata "
            "Model Card (.json file) matching the required schema.\n"
            "Output must be a valid JSON dictionary containing keys:\n"
            "1. 'category': one of the existing L1 categories, or a new category name if none fit (with lowercase_underscores).\n"
            "2. 'category_description': description of the new category (only if a new category is created).\n"
            "3. 'model_card': the Model Card dictionary following this exact schema:\n"
            "   {\n"
            "     \"artifact_id\": \"<unique_artifact_name_lowercase_underscores>\",\n"
            "     \"category\": \"<category_name>\",\n"
            "     \"description\": \"<brief description of what this code does>\",\n"
            "     \"interface\": {\n"
            "       \"entrypoint\": \"<entrypoint_function_name>(X_train, X_test, y_train=None, ...)\"\n"
            "     },\n"
            "     \"dependencies\": [\"numpy\", \"pandas\", ...],\n"
            "     \"verified\": false,\n"
            "     \"verification_log\": \"\"\n"
            "   }\n"
            "4. 'code': the self-contained importable python code containing the entrypoint function."
        )
        
        user_prompt = f"""
        Raw Source Title: {source_name}
        Raw Source Content:
        {source_content}
        
        Existing L1 Categories:
        {l1_list_str}
        
        Analyze this raw source. Extract ONE modular, reusable tabular ML technique/model/preprocessing method.
        Format the response strictly as a JSON object.
        """
        response_str = call_llm(system_prompt, user_prompt, model=self.model_name)
        
        # Clean response string to parse JSON
        try:
            if "```json" in response_str:
                json_str = response_str.split("```json")[1].split("```")[0]
            elif "```" in response_str:
                json_str = response_str.split("```")[1].split("```")[0]
            else:
                json_str = response_str
            data = json.loads(json_str.strip())
        except Exception as e:
            print(f"L2Builder ERROR: Failed to parse LLM response as JSON: {e}")
            print(f"Raw response: {response_str}")
            return False, None, None, None
 
        category = data.get("category")
        model_card = data.get("model_card")
        code = data.get("code")
        
        if not category or not model_card or not code:
            print("L2Builder ERROR: Missing category, model_card or code in LLM response.")
            return False, None, None, None
            
        artifact_id = model_card.get("artifact_id")
        if not artifact_id:
            print("L2Builder ERROR: Missing artifact_id inside model_card.")
            return False, None, None, None
            
        # Handle new category creation
        if commit and category not in l1_index:
            print(f"L2Builder: Creating new category '{category}'")
            l1_index[category] = {
                "description": data.get("category_description", "Custom category."),
                "l2_pointers": []
            }
            # Save updated L1 index
            with open(self.l1_path, 'w', encoding='utf-8') as f:
                json.dump(l1_index, f, indent=2)
                
        # Prepare L2 store paths
        if commit:
            cat_dir = self.l2_store / category
        else:
            if not target_dir:
                raise ValueError("target_dir must be provided if commit=False")
            cat_dir = Path(target_dir)
            
        cat_dir.mkdir(parents=True, exist_ok=True)
        
        card_file = cat_dir / f"{artifact_id}.json"
        code_file = cat_dir / f"{artifact_id}.py"
        
        # Write files
        with open(code_file, 'w', encoding='utf-8') as f:
            f.write(code.strip())
            
        model_card["verified"] = False
        model_card.pop("code_path", None)  # Remove any LLM-provided code_path
        model_card["code_path"] = f"{artifact_id}.py"
        model_card["category"] = category
        with open(card_file, 'w', encoding='utf-8') as f:
            json.dump(model_card, f, indent=2)
            
        print(f"L2Builder: Wrote temporary artifact files to {cat_dir}")
        
        # Run sandbox verifier on the newly built files
        cmd = [self.venv_python, str(self.verifier_path), str(card_file)]
        
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"L2Builder: Sandbox verification passed for {artifact_id}!")
            try:
                with open(card_file, 'r', encoding='utf-8') as f:
                    model_card = json.load(f)
            except Exception:
                pass
            
            # Commit to L1 pointers list if not already present
            if commit:
                if artifact_id not in l1_index[category]["l2_pointers"]:
                    l1_index[category]["l2_pointers"].append(artifact_id)
                    with open(self.l1_path, 'w', encoding='utf-8') as f:
                        json.dump(l1_index, f, indent=2)
            return True, category, artifact_id, model_card
        except subprocess.CalledProcessError as e:
            print(f"L2Builder ERROR: Sandbox verification failed for {artifact_id}.")
            if e.stdout:
                print(e.stdout)
            print(e.stderr)
            # Remove bad files to avoid corrupting pool
            if code_file.exists():
                code_file.unlink()
            if card_file.exists():
                card_file.unlink()
            return False, None, None, None

    def commit_artifact(self, category: str, artifact_id: str, local_code_file: Path, local_card_file: Path) -> bool:
        """
        Moves/copies verified local artifact files to the global l2_store and updates the L1 index.
        """
        # Load current L1 index
        with open(self.l1_path, 'r', encoding='utf-8') as f:
            l1_index = json.load(f)
            
        # Ensure category exists in L1
        if category not in l1_index:
            l1_index[category] = {
                "description": f"Category for {category} techniques.",
                "l2_pointers": []
            }
            
        # Target paths in l2_store
        cat_dir = self.l2_store / category
        cat_dir.mkdir(parents=True, exist_ok=True)
        
        dest_card_file = cat_dir / f"{artifact_id}.json"
        dest_code_file = cat_dir / f"{artifact_id}.py"
        
        # Copy files
        import shutil
        try:
            shutil.copy(local_code_file, dest_code_file)
            shutil.copy(local_card_file, dest_card_file)
            
            # Read and update model card to set verified = True
            with open(dest_card_file, 'r', encoding='utf-8') as f:
                card_data = json.load(f)
            card_data["verified"] = True
            card_data["category"] = category
            card_data["code_path"] = f"{artifact_id}.py"
            with open(dest_card_file, 'w', encoding='utf-8') as f:
                json.dump(card_data, f, indent=2)
                
            # Add to L1 pointers if not already present
            if artifact_id not in l1_index[category]["l2_pointers"]:
                l1_index[category]["l2_pointers"].append(artifact_id)
                with open(self.l1_path, 'w', encoding='utf-8') as f:
                    json.dump(l1_index, f, indent=2)
            print(f"L2Builder: Successfully committed artifact {artifact_id} to global category '{category}'.")
            return True
        except Exception as e:
            print(f"L2Builder ERROR: Failed to commit artifact {artifact_id} from local paths: {e}")
            return False
