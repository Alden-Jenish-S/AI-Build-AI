import os
import json
import sys
import subprocess
from pathlib import Path
from typing import Dict, Any
from agents.llm_utils import call_llm
from agents.setup_agent import SetupAgent
from runtime_utils import accelerator_subprocess_env, validate_storage_identifier

class L2Builder:
    def __init__(
        self,
        project_root: Path,
        model_name: str = None,
        venv_path: str = None,
        preferred_accelerator: str = "cpu",
    ):
        self.project_root = project_root
        self.model_name = model_name
        self.l1_path = project_root / "memory_pool" / "l1_index.json"
        self.l2_store = project_root / "memory_pool" / "l2_store"
        self.verifier_path = project_root / "memory_pool" / "builder" / "sandbox_verifier.py"
        self.preferred_accelerator = str(preferred_accelerator).lower()
        if self.preferred_accelerator not in {"cpu", "cuda", "mps"}:
            raise ValueError("preferred_accelerator must be cpu, cuda, or mps")

        import sys
        candidate_path = Path(venv_path or "./.venv/bin/python")
        if not candidate_path.is_absolute():
            candidate_path = project_root / candidate_path
        resolved_path = str(candidate_path.resolve())
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
            "       \"entrypoint\": \"<entrypoint_function_name>(X_train, X_test, y_train=None, ...)\",\n"
            "       \"output_contract\": {\"kind\": \"predictions|transformed_features\", \"aligned_to\": \"X_test\", \"value_type\": \"probability|label|continuous|features\"}\n"
            "     },\n"
            "     \"capabilities\": {\"supported_operators\": [\"refine\", \"tune\", \"diversify\"], \"tunable_parameters\": [\"parameter_name\"], \"gpu_accelerated\": <true|false>, \"supported_accelerators\": [\"cpu\", \"cuda\", \"mps\"], \"input_types\": [\"numeric\", \"categorical\", \"missing\"], \"target_types\": [\"binary_classification|multiclass_classification|regression\"]},\n"
            "     \"scope\": \"<component|model_family|full_pipeline>\",\n"
            "     \"resource_profile\": {\"accelerator\": \"<cpu|gpu|cuda|mps|any>\", \"min_ram_gb\": <number>, \"estimated_runtime_seconds\": <number>},\n"
            "     \"dependencies\": [\"numpy\", \"pandas\", ...],\n"
            "     \"verified\": false,\n"
            "     \"verification_log\": \"\"\n"
            "   }\n"
            "4. 'code': the self-contained importable python code containing the entrypoint function.\n\n"
            "CRITICAL CODE CONSTRAINTS:\n"
            "- The entrypoint function and all helper functions MUST be fully compatible with standard tabular inputs (Pandas DataFrames for X_train/X_test, and NumPy arrays or Pandas Series for y_train).\n"
            "- Do NOT assume inputs are PyTorch/TensorFlow tensors. If the technique requires neural networks, convert Pandas DataFrame/NumPy inputs to tensors inside the entrypoint function (e.g., using torch.tensor(X_train.values) or similar) and convert the outputs back to NumPy arrays or Pandas DataFrames/Series.\n"
            "- Ensure that any optional parameter has a default value (e.g. y_train=None).\n"
            "- The function must execute successfully and return predictions or transformed features without throwing type/attribute errors (such as calling tensor methods directly on DataFrame inputs).\n"
            "- Neural artifacts must use training-fit mixed-type preprocessing, float32 mini-batches, early stopping, batched CPU-detached output, and the AIBUILDAI_MAX_EPOCHS/AIBUILDAI_EARLY_STOPPING_PATIENCE ceilings. Handle singleton final batches and all-missing training columns safely.\n"
            f"- The target runtime prefers {self.preferred_accelerator}, exposed through the "
            "AIBUILDAI_ACCELERATOR environment variable. For compatible models, read that variable, "
            "enable the framework-native accelerator, and retry safely on CPU if the installed package "
            "does not provide that backend. Accurately declare GPU support in capabilities."
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
            
        try:
            category = validate_storage_identifier(category, "category")
            artifact_id = validate_storage_identifier(
                model_card.get("artifact_id"), "artifact_id"
            )
        except ValueError as exc:
            print(f"L2Builder ERROR: Unsafe generated identifier: {exc}")
            return False, None, None, None
        if not isinstance(code, str) or not code.strip():
            print("L2Builder ERROR: Generated code must be a non-empty string.")
            return False, None, None, None
        if model_card.get("scope") not in {
            "component", "model_family", "full_pipeline"
        }:
            print("L2Builder ERROR: Model card must declare a valid artifact scope.")
            return False, None, None, None
        resource_profile = model_card.get("resource_profile")
        if not isinstance(resource_profile, dict) or resource_profile.get(
            "accelerator"
        ) not in {"cpu", "gpu", "cuda", "mps", "any"}:
            print("L2Builder ERROR: Model card must declare a valid resource profile.")
            return False, None, None, None
            
        # Handle new category creation
        if commit and category not in l1_index:
            print(f"L2Builder: Creating new category '{category}'")
            l1_index[category] = {
                "description": data.get("category_description", "Custom category."),
                "l2_pointers": []
            }
                
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

        if card_file.exists() or code_file.exists():
            print(
                f"L2Builder ERROR: Refusing to overwrite existing artifact files for {artifact_id}."
            )
            return False, None, None, None
        
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

        # Verification imports the generated module, so dependencies must exist
        # first. Only packages and versions already present in the repository's
        # human-controlled requirements file are eligible at this unverified stage.
        try:
            SetupAgent(self.venv_python).install_allowlisted_dependencies(
                [model_card], self.project_root / "requirements.txt"
            )
        except Exception as exc:
            print(
                f"L2Builder ERROR: Could not prepare allowlisted dependencies for "
                f"{artifact_id}: {exc}"
            )
            model_card["verified"] = False
            model_card["verification_log"] = (
                f"Dependency preparation failed before verification: {exc}"
            )
            with open(card_file, 'w', encoding='utf-8') as f:
                json.dump(model_card, f, indent=2)
            if commit:
                code_file.unlink(missing_ok=True)
                card_file.unlink(missing_ok=True)
                return False, None, None, None
            return False, category, artifact_id, model_card

        # Run sandbox verification with one focused repair pass. Most generated
        # failures in practice are small interface/type defects that are cheaper to
        # repair from the traceback than to discard and regenerate from scratch.
        cmd = [self.venv_python, str(self.verifier_path), str(card_file)]
        verification_output = ""
        for verification_attempt in range(2):
            try:
                res = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=35,
                    env=accelerator_subprocess_env(self.preferred_accelerator),
                )
                print(f"L2Builder: Sandbox verification passed for {artifact_id}!")
                try:
                    with open(card_file, 'r', encoding='utf-8') as f:
                        model_card = json.load(f)
                except Exception:
                    pass
                if commit and artifact_id not in l1_index[category]["l2_pointers"]:
                    l1_index[category]["l2_pointers"].append(artifact_id)
                    with open(self.l1_path, 'w', encoding='utf-8') as f:
                        json.dump(l1_index, f, indent=2)
                return True, category, artifact_id, model_card
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
                verification_output = (
                    str(getattr(exc, "stdout", "") or "")
                    + "\n"
                    + str(getattr(exc, "stderr", "") or "")
                )[-8000:]
                print(f"L2Builder ERROR: Sandbox verification failed for {artifact_id}.")
                if verification_output.strip():
                    print(verification_output)
                if verification_attempt == 0:
                    try:
                        repair_response = call_llm(
                            "You repair a generated reusable Python artifact. Return only the complete "
                            "corrected Python module in one code block. Preserve the declared entrypoint "
                            "and methodology; fix the concrete traceback and ensure returned predictions "
                            "align with X_test. Preserve use of AIBUILDAI_ACCELERATOR for supported models "
                            "and retain a safe CPU fallback.",
                            f"Entrypoint: {model_card.get('interface', {}).get('entrypoint')}\n"
                            f"Verification failure:\n{verification_output}\n\n"
                            f"Current module:\n```python\n{code_file.read_text(encoding='utf-8')}\n```",
                            model=self.model_name,
                            temperature=0.0,
                        )
                        if "```python" in repair_response:
                            repaired = repair_response.split("```python", 1)[1].split("```", 1)[0]
                        elif "```" in repair_response:
                            repaired = repair_response.split("```", 1)[1].split("```", 1)[0]
                        else:
                            repaired = repair_response
                        compile(repaired, str(code_file), "exec")
                        code_file.write_text(repaired.strip() + "\n", encoding="utf-8")
                        continue
                    except Exception as repair_error:
                        print(f"L2Builder ERROR: Artifact repair failed: {repair_error}")
                        break

        if commit:
            code_file.unlink(missing_ok=True)
            card_file.unlink(missing_ok=True)
            return False, None, None, None
        try:
            with open(card_file, 'r', encoding='utf-8') as f:
                model_card = json.load(f)
        except Exception:
            pass
        return False, category, artifact_id, model_card

    def commit_artifact(self, category: str, artifact_id: str, local_code_file: Path, local_card_file: Path) -> bool:
        """
        Moves/copies verified local artifact files to the global l2_store and updates the L1 index.
        """
        try:
            category = validate_storage_identifier(category, "category")
            artifact_id = validate_storage_identifier(artifact_id, "artifact_id")
            local_code_file = Path(local_code_file)
            local_card_file = Path(local_card_file)
            if local_code_file.is_symlink() or local_card_file.is_symlink():
                raise ValueError("Artifact source files may not be symlinks")
            if not local_code_file.is_file() or not local_card_file.is_file():
                raise ValueError("Artifact source files do not exist")
            if local_code_file.name != f"{artifact_id}.py":
                raise ValueError("Local code filename does not match artifact_id")
            if local_card_file.name != f"{artifact_id}.json":
                raise ValueError("Local card filename does not match artifact_id")
            with open(local_card_file, 'r', encoding='utf-8') as f:
                local_card_data = json.load(f)
            if local_card_data.get("verified") is not True:
                print(f"L2Builder ERROR: Refusing to commit unverified artifact {artifact_id}.")
                return False
            if local_card_data.get("artifact_id") != artifact_id:
                raise ValueError("Model-card artifact_id does not match requested artifact")
            if local_card_data.get("category") != category:
                raise ValueError("Model-card category does not match requested category")
            if local_card_data.get("code_path") != f"{artifact_id}.py":
                raise ValueError("Model-card code_path does not match requested artifact")
        except Exception as e:
            print(f"L2Builder ERROR: Could not validate local model card for {artifact_id}: {e}")
            return False

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

        if dest_card_file.exists() or dest_code_file.exists():
            print(f"L2Builder ERROR: Refusing to overwrite existing artifact {artifact_id}.")
            return False
        
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

    def record_task_validation(
        self,
        category: str,
        artifact_id: str,
        validation: Dict[str, Any],
    ) -> bool:
        """Append or replace a task-level validation on a committed artifact."""
        try:
            category = validate_storage_identifier(category, "category")
            artifact_id = validate_storage_identifier(artifact_id, "artifact_id")
            card_file = self.l2_store / category / f"{artifact_id}.json"
            with open(card_file, 'r', encoding='utf-8') as f:
                card = json.load(f)
            if card.get("verified") is not True:
                raise ValueError("cannot validate an unverified global artifact")

            validations = card.get("task_validations", [])
            if not isinstance(validations, list):
                validations = []
            key = (validation.get("task_name"), validation.get("node_id"))
            validations = [
                item
                for item in validations
                if (item.get("task_name"), item.get("node_id")) != key
            ]
            validations.append(dict(validation))
            card["task_validations"] = validations
            with open(card_file, 'w', encoding='utf-8') as f:
                json.dump(card, f, indent=2)
            return True
        except Exception as exc:
            print(
                f"L2Builder ERROR: Failed to record task validation for "
                f"{category}/{artifact_id}: {exc}"
            )
            return False
