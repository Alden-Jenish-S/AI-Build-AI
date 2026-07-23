import ast
import os
import json
import sys
import subprocess
from pathlib import Path
from typing import Dict, Any
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from agents.llm_utils import call_llm
from agents.setup_agent import SetupAgent
from runtime_utils import (
    absolute_path_without_symlink_resolution,
    accelerator_subprocess_env,
    validate_storage_identifier,
)

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
        if venv_path is None:
            self.venv_python = sys.executable
            return
        candidate_path = Path(venv_path)
        if not candidate_path.is_absolute():
            candidate_path = project_root / candidate_path
        resolved_path = str(
            absolute_path_without_symlink_resolution(candidate_path)
        )
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

    @staticmethod
    def _artifact_code_issues(code: str) -> list[str]:
        """Reject generated artifact code with external or import-time side effects."""
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            return [f"syntax error: {exc}"]

        issues = []
        forbidden_import_roots = {
            "httpx",
            "requests",
            "socket",
            "subprocess",
            "urllib",
        }
        forbidden_calls = {
            "eval",
            "exec",
            "open",
            "os.popen",
            "os.system",
            "pathlib.Path.touch",
            "pathlib.Path.write_bytes",
            "pathlib.Path.write_text",
            "shutil.rmtree",
        }

        def qualified_name(node: ast.AST) -> str:
            parts = []
            current = node
            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                parts.append(current.id)
            return ".".join(reversed(parts))

        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = (
                    [alias.name for alias in node.names]
                    if isinstance(node, ast.Import)
                    else [node.module or ""]
                )
                for name in names:
                    if name.split(".", 1)[0] in forbidden_import_roots:
                        issues.append(
                            f"forbidden external-side-effect import {name!r}"
                        )
            elif isinstance(node, ast.Call):
                name = qualified_name(node.func)
                if (
                    name in forbidden_calls
                    or name.endswith(".write_text")
                    or name.endswith(".write_bytes")
                    or name.endswith(".touch")
                ):
                    issues.append(f"forbidden side-effect call {name!r}")
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = (
                    node.targets if isinstance(node, ast.Assign) else [node.target]
                )
                if any(
                    qualified_name(target) == "os.devnull" for target in targets
                ):
                    issues.append("must not monkey-patch os.devnull")
        return sorted(set(issues))

    def build_from_source(self, source_name: str, source_content: str, commit: bool = True, target_dir: Path = None) -> tuple:
        """
        Takes raw source content, asks LLM to classify it, extract model card + code,
        saves them, and runs sandbox_verifier.
        """
        # Load current L1 index
        with open(self.l1_path, 'r', encoding='utf-8') as f:
            l1_index = json.load(f)
            
        l1_list_str = "\n".join([f"- {cat}: {details['description']}" for cat, details in l1_index.items()])
        requirements_file = self.project_root / "requirements.txt"
        allowed_dependencies = []
        if requirements_file.is_file():
            for raw_line in requirements_file.read_text(
                encoding="utf-8"
            ).splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    allowed_dependencies.append(Requirement(line).name)
                except ValueError:
                    continue
        dependency_allowlist = ", ".join(
            sorted(set(allowed_dependencies))
        ) or "Python standard library only"
        
        system_prompt = (
            "You are the L2 Artifact Builder. Your goal is to convert a raw source snippet "
            "into a single, clean, self-contained reusable ML utility function (.py file) and its metadata "
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
            "       \"entrypoint\": \"<entrypoint_function_name>(<modality-appropriate parameters>, ...)\",\n"
            "       \"input_contract\": {\"modality\": \"tabular|text|image|audio|timeseries|video|graph|multimodal|array\", \"container\": \"dataframe|numpy|list|tensor|paths|file|directory\", \"parameter_roles\": {\"<entrypoint_parameter>\": \"train|test|train.text|test.text|train.image|test.image|target|predictions|numeric_columns|categorical_columns|feature_columns|target_column\"}, \"sample_shape\": [<optional per-sample dimensions>], \"file_extension\": \"<optional .csv|.jsonl|.txt|.npy|.wav|.png>\"},\n"
            "       \"output_contract\": {\"kind\": \"predictions|transformed_features|embeddings|labels|logits|masks|splits|model\", \"aligned_to\": \"<entrypoint parameter or train/test role>\", \"value_type\": \"probability|label|continuous|features\"}\n"
            "     },\n"
            "     \"capabilities\": {\"supported_operators\": [\"refine\", \"tune\", \"diversify\"], \"tunable_parameters\": [\"parameter_name\"], \"gpu_accelerated\": <true|false>, \"supported_accelerators\": [\"cpu\", \"cuda\", \"mps\"], \"input_types\": [\"modality-specific input traits\"], \"target_types\": [\"binary_classification|multiclass_classification|multilabel_classification|regression\"]},\n"
            "     \"scope\": \"<component|model_family|full_pipeline>\",\n"
            "     \"resource_profile\": {\"accelerator\": \"<cpu|gpu|cuda|mps|any>\", \"min_ram_gb\": <number>, \"estimated_runtime_seconds\": <number>},\n"
            "     \"dependencies\": [\"numpy\", \"pandas\", ...],\n"
            "     \"verified\": false,\n"
            "     \"verification_log\": \"\"\n"
            "   }\n"
            "4. 'code': the self-contained importable python code containing the entrypoint function.\n\n"
            "CRITICAL CODE CONSTRAINTS:\n"
            "- Preserve the source technique's real data modality. The interface.input_contract MUST accurately declare the synthetic container, sample shape/file type when relevant, and the role of every required data parameter. Do not force text, image, audio, temporal, graph, or multimodal methods into Pandas DataFrames.\n"
            "- Use framework tensors only when input_contract.container declares tensor; otherwise convert the declared public input container inside the entrypoint and return ordinary arrays/tables/sequences matching output_contract.\n"
            "- Ensure that any optional parameter has a default value (e.g. y_train=None).\n"
            "- The function must execute successfully against the declared modality and return a value satisfying output_contract. Prediction-like outputs must use samples on axis zero and align with the declared train/test parameter.\n"
            "- Artifact modules are pure importable utilities: they may read only file/path parameters explicitly declared by input_contract, and must never write files, discover unrelated local files, access the network, launch subprocesses, mutate os.devnull, or perform import-time side effects.\n"
            "- Third-party imports and Model Card dependencies MUST be limited to "
            f"these project-allowlisted distributions: {dependency_allowlist}. "
            "Do not import or declare any other package. If the source mentions an "
            "unavailable convenience library, implement the mechanism with an "
            "allowlisted lower-level framework instead of silently changing the "
            "methodology.\n"
            "- Neural artifacts must use modality-appropriate training-fit preprocessing, float32 mini-batches where applicable, early stopping, batched CPU-detached output, and the AIBUILDAI_MAX_EPOCHS/AIBUILDAI_EARLY_STOPPING_PATIENCE ceilings. Handle singleton final batches and modality-specific missing/corrupt samples safely.\n"
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
        
        Analyze this raw source. Extract ONE modular, reusable ML technique/model/preprocessing method while preserving its actual data modality.
        Format the response strictly as a JSON object.
        """
        def parse_response(response: str) -> dict:
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0]
            else:
                json_str = response
            parsed = json.loads(json_str.strip())
            if not isinstance(parsed, dict):
                raise ValueError("builder response must be a JSON object")
            return parsed

        response_str = call_llm(system_prompt, user_prompt, model=self.model_name)
        try:
            data = parse_response(response_str)
        except Exception as e:
            print(f"L2Builder ERROR: Failed to parse LLM response as JSON: {e}")
            print(f"Raw response: {response_str}")
            return False, None, None, None

        allowed_names = {
            canonicalize_name(name) for name in allowed_dependencies
        }
        # Dependency mismatch used to terminate before either verification or
        # repair. Give the builder two focused opportunities to regenerate the
        # complete artifact against the explicit project allowlist.
        for dependency_retry in range(2):
            candidate_card = data.get("model_card")
            declared = (
                candidate_card.get("dependencies", [])
                if isinstance(candidate_card, dict)
                else []
            )
            if not isinstance(declared, list):
                unavailable = ["<dependencies must be a list>"]
            else:
                unavailable = []
                for dependency in declared:
                    try:
                        dependency_name = canonicalize_name(
                            Requirement(str(dependency)).name
                        )
                    except ValueError:
                        unavailable.append(str(dependency))
                        continue
                    if dependency_name not in allowed_names:
                        unavailable.append(str(dependency))
            if not unavailable:
                break
            repair_prompt = (
                user_prompt
                + "\n\nThe previous artifact was rejected before verification "
                + "because it declared unavailable dependencies: "
                + repr(unavailable)
                + ". Regenerate the COMPLETE JSON artifact using only the "
                + "allowlisted distributions. Preserve the planned methodology "
                + "and use an allowlisted lower-level implementation where needed."
                + "\n\nPrevious invalid response:\n"
                + response_str[-16000:]
            )
            try:
                response_str = call_llm(
                    system_prompt,
                    repair_prompt,
                    model=self.model_name,
                    temperature=0.0,
                )
                data = parse_response(response_str)
            except Exception as exc:
                print(
                    "L2Builder WARNING: Dependency-constrained regeneration "
                    f"failed: {exc}"
                )
                break
 
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
        interface = model_card.get("interface")
        if (
            not isinstance(interface, dict)
            or not isinstance(interface.get("entrypoint"), str)
            or not interface["entrypoint"].strip()
        ):
            print(
                "L2Builder ERROR: Model card must declare interface.entrypoint."
            )
            return False, None, None, None
        input_contract = interface.get("input_contract")
        if input_contract is not None:
            supported_modalities = {
                "array",
                "audio",
                "graph",
                "image",
                "multimodal",
                "tabular",
                "text",
                "timeseries",
                "video",
            }
            supported_containers = {
                "array",
                "dataframe",
                "dict",
                "directory",
                "file",
                "file_paths",
                "files",
                "list",
                "ndarray",
                "numpy",
                "paths",
                "tensor",
                "torch",
            }
            if (
                not isinstance(input_contract, dict)
                or input_contract.get("modality")
                not in supported_modalities
                or input_contract.get("container")
                not in supported_containers
                or not isinstance(
                    input_contract.get("parameter_roles"), dict
                )
            ):
                print(
                    "L2Builder ERROR: interface.input_contract must declare "
                    "a supported modality, container, and parameter_roles object."
                )
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

        # Run sandbox verification with two focused repair passes. Generated
        # failures are commonly small interface, missing-value, or sandbox defects;
        # preserve the methodology and repair those before falling back.
        cmd = [self.venv_python, str(self.verifier_path), str(card_file)]
        verification_output = ""
        verification_attempts = []
        for verification_attempt in range(3):
            code_issues = self._artifact_code_issues(
                code_file.read_text(encoding="utf-8")
            )
            if code_issues:
                verification_output = (
                    "Static artifact safety rejection:\n- "
                    + "\n- ".join(code_issues)
                )
                verification_attempts.append(
                    {
                        "attempt": verification_attempt + 1,
                        "status": "static_rejection",
                        "diagnostics": verification_output[-4000:],
                    }
                )
                print(
                    f"L2Builder ERROR: Generated artifact {artifact_id} "
                    "failed static safety validation."
                )
                print(verification_output)
            else:
                try:
                    subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        check=True,
                        timeout=90,
                        env=accelerator_subprocess_env(
                            self.preferred_accelerator
                        ),
                    )
                except (
                    subprocess.CalledProcessError,
                    subprocess.TimeoutExpired,
                ) as exc:
                    verification_output = (
                        str(getattr(exc, "stdout", "") or "")
                        + "\n"
                        + str(getattr(exc, "stderr", "") or "")
                    )[-8000:]
                    verification_attempts.append(
                        {
                            "attempt": verification_attempt + 1,
                            "status": "verification_failed",
                            "diagnostics": verification_output[-4000:],
                        }
                    )
                    print(
                        f"L2Builder ERROR: Sandbox verification failed for "
                        f"{artifact_id}."
                    )
                    if verification_output.strip():
                        print(verification_output)
                else:
                    verification_attempts.append(
                        {
                            "attempt": verification_attempt + 1,
                            "status": "verified",
                        }
                    )
                    print(
                        f"L2Builder: Sandbox verification passed for "
                        f"{artifact_id}!"
                    )
                    try:
                        with open(card_file, 'r', encoding='utf-8') as f:
                            model_card = json.load(f)
                    except Exception:
                        pass
                    model_card["builder_verification_attempts"] = (
                        verification_attempts
                    )
                    with open(card_file, "w", encoding="utf-8") as f:
                        json.dump(model_card, f, indent=2)
                    if (
                        commit
                        and artifact_id
                        not in l1_index[category]["l2_pointers"]
                    ):
                        l1_index[category]["l2_pointers"].append(artifact_id)
                        with open(self.l1_path, 'w', encoding='utf-8') as f:
                            json.dump(l1_index, f, indent=2)
                    return True, category, artifact_id, model_card

            if verification_attempt < 2:
                try:
                    repair_response = call_llm(
                        "You repair a generated reusable Python artifact. Return only the complete "
                        "corrected Python module in one code block. Preserve the exact declared "
                        "entrypoint and methodology; fix the concrete failure and ensure returned "
                        "outputs satisfy interface.output_contract and align with its declared "
                        "input parameter. Preserve interface.input_contract exactly; do not convert "
                        "a non-tabular modality into Pandas merely to satisfy verification. The "
                        "module must be a pure utility: it may read only declared input-path "
                        "parameters and must never write files, discover unrelated local files, "
                        "access the network, launch subprocesses, patch os.devnull, or perform "
                        "import-time side effects. Handle modality-appropriate missing/corrupt "
                        "samples. Preserve AIBUILDAI_ACCELERATOR support and a safe CPU fallback.",
                        f"Repair pass: {verification_attempt + 1} of 2\n"
                        f"Entrypoint: {model_card.get('interface', {}).get('entrypoint')}\n"
                        f"Input contract: {json.dumps(model_card.get('interface', {}).get('input_contract', {}))}\n"
                        f"Output contract: {json.dumps(model_card.get('interface', {}).get('output_contract', {}))}\n"
                        f"Verification failure:\n{verification_output}\n\n"
                        f"Current module:\n```python\n"
                        f"{code_file.read_text(encoding='utf-8')}\n```",
                        model=self.model_name,
                        temperature=0.0,
                    )
                    if "```python" in repair_response:
                        repaired = repair_response.split(
                            "```python", 1
                        )[1].split("```", 1)[0]
                    elif "```" in repair_response:
                        repaired = repair_response.split(
                            "```", 1
                        )[1].split("```", 1)[0]
                    else:
                        repaired = repair_response
                    compile(repaired, str(code_file), "exec")
                    code_file.write_text(
                        repaired.strip() + "\n", encoding="utf-8"
                    )
                except Exception as repair_error:
                    verification_output = (
                        f"Artifact repair generation failed: {repair_error}"
                    )
                    verification_attempts.append(
                        {
                            "attempt": verification_attempt + 1,
                            "status": "repair_generation_failed",
                            "diagnostics": verification_output[-4000:],
                        }
                    )
                    print(
                        f"L2Builder ERROR: Artifact repair failed: "
                        f"{repair_error}"
                    )

        if commit:
            code_file.unlink(missing_ok=True)
            card_file.unlink(missing_ok=True)
            return False, None, None, None
        try:
            with open(card_file, 'r', encoding='utf-8') as f:
                model_card = json.load(f)
        except Exception:
            pass
        model_card["builder_verification_attempts"] = verification_attempts
        with open(card_file, "w", encoding="utf-8") as f:
            json.dump(model_card, f, indent=2)
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
