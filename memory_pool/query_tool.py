import json
from pathlib import Path
from runtime_utils import resolve_within, validate_storage_identifier

# Base directory setup relative to this file
BASE_DIR = Path(__file__).resolve().parent

COMPONENT_CATEGORIES = {
    "feature_engineering_tabular",
    "categorical_encoding",
    "cv_strategies",
    "imbalance_handling",
    "hyperparam_search",
    "missing_value_imputation",
    "outlier_handling",
}
FULL_PIPELINE_CATEGORIES = {"blending_stacking"}


def infer_artifact_scope(card: dict, category: str = "") -> str:
    """Return explicit scope, or a conservative scope for legacy model cards."""
    scope = card.get("scope")
    if scope in {"component", "model_family", "full_pipeline"}:
        return scope
    category = category or card.get("category", "")
    if category in COMPONENT_CATEGORIES:
        return "component"
    if category in FULL_PIPELINE_CATEGORIES:
        return "full_pipeline"
    return "model_family"


def normalize_resource_profile(card: dict) -> dict:
    """Normalize the optional feasibility declaration used before experiments."""
    profile = card.get("resource_profile")
    if not isinstance(profile, dict):
        profile = {}
    accelerator = str(profile.get("accelerator", "cpu")).lower()
    if accelerator not in {"cpu", "gpu", "cuda", "mps", "any"}:
        accelerator = "cpu"
    def non_negative_number(value: object) -> float:
        try:
            return max(0.0, float(value or 0.0))
        except (TypeError, ValueError):
            return 0.0

    return {
        "accelerator": accelerator,
        "min_ram_gb": non_negative_number(profile.get("min_ram_gb")),
        "estimated_runtime_seconds": non_negative_number(
            profile.get("estimated_runtime_seconds")
        ),
    }

def query(*args) -> dict:
    """
    Query the memory pool index or specific artifacts.
    Supports two signatures:
      1. query(category: str) -> dict
         Returns L1 category description + list of artifact cards (excluding code/examples).
      2. query(category: str, artifact_id: str) -> dict
         Returns the full L2 Model Card including the raw source code of the artifact.
    """
    if len(args) == 1:
        return query_l1(args[0])
    elif len(args) == 2:
        return query_l2(args[0], args[1])
    else:
        raise TypeError(f"query() takes 1 or 2 positional arguments but {len(args)} were given")

def query_l1(category: str) -> dict:
    """Returns the L1 category description and summary of L2 pointer cards.
    Does NOT leak the code.
    """
    category = validate_storage_identifier(category, "category")
    l1_path = BASE_DIR / "l1_index.json"
    if not l1_path.exists():
        raise FileNotFoundError(f"L1 index file not found at {l1_path}")
        
    with open(l1_path, 'r', encoding='utf-8') as f:
        l1_index = json.load(f)
        
    if category not in l1_index:
        return {"category": category, "description": "", "artifacts": []}
        
    cat_info = l1_index[category]
    l2_summaries = []
    
    for artifact_id in cat_info.get("l2_pointers", []):
        try:
            artifact_id = validate_storage_identifier(artifact_id, "artifact_id")
            card_path = resolve_within(
                BASE_DIR / "l2_store", category, f"{artifact_id}.json"
            )
        except ValueError as exc:
            l2_summaries.append({"artifact_id": str(artifact_id), "error": str(exc)})
            continue
        if card_path.exists():
            try:
                with open(card_path, 'r', encoding='utf-8') as f:
                    card = json.load(f)
                if card.get("verified") is not True:
                    continue
                if card.get("artifact_id") != artifact_id or card.get("category") != category:
                    raise ValueError("Model-card identity does not match its L1 pointer")
                validations = card.get("task_validations", [])
                if not isinstance(validations, list):
                    validations = []
                completed = [
                    item
                    for item in validations
                    if item.get("status") == "completed" and item.get("score") is not None
                ]
                improvements = [
                    item for item in completed if item.get("improved_over_baseline") is True
                ]
                rewards = [
                    float(item["reward"])
                    for item in completed
                    if isinstance(item.get("reward"), (int, float))
                ]
                # Keep only search/selection metadata, drop the source code or heavy details
                summary = {
                    "artifact_id": card["artifact_id"],
                    "category": card["category"],
                    "description": card["description"],
                    "interface": card["interface"],
                    "capabilities": card.get("capabilities", {}),
                    "verified": card.get("verified", False),
                    "verification_level": card.get("verification_level", "legacy"),
                    "scope": infer_artifact_scope(card, category),
                    "resource_profile": normalize_resource_profile(card),
                    "known_pitfalls": card.get("known_pitfalls", []),
                    "validation_summary": {
                        "runs": len(validations),
                        "completed": len(completed),
                        "failures": len(validations) - len(completed),
                        "improvement_rate": (
                            len(improvements) / len(completed) if completed else None
                        ),
                        "mean_reward": sum(rewards) / len(rewards) if rewards else None,
                    },
                    "recent_validations": validations[-3:],
                }
                l2_summaries.append(summary)
            except Exception as e:
                # Log error or append error description
                l2_summaries.append({"artifact_id": artifact_id, "error": f"Failed to load: {e}"})
                
    return {
        "category": category,
        "description": cat_info.get("description", ""),
        "artifacts": l2_summaries
    }

def query_l2(category: str, artifact_id: str) -> dict:
    """Returns the complete Model Card details including the actual code content."""
    category = validate_storage_identifier(category, "category")
    artifact_id = validate_storage_identifier(artifact_id, "artifact_id")
    card_path = resolve_within(BASE_DIR / "l2_store", category, f"{artifact_id}.json")
    if not card_path.exists():
        raise FileNotFoundError(f"L2 model card not found at {card_path}")
        
    with open(card_path, 'r', encoding='utf-8') as f:
        card = json.load(f)
    if card.get("verified") is not True:
        raise ValueError(f"Artifact is not verified: {category}/{artifact_id}")
    if card.get("artifact_id") != artifact_id or card.get("category") != category:
        raise ValueError("Model-card identity does not match the requested artifact")
        
    expected_code_name = f"{artifact_id}.py"
    if card.get("code_path") != expected_code_name:
        raise ValueError("Model-card code_path does not match artifact_id")
    code_path = resolve_within(card_path.parent, expected_code_name)
    if not code_path.exists():
        raise FileNotFoundError(f"Code file not found at {code_path}")
        
    with open(code_path, 'r', encoding='utf-8') as f:
        code_content = f.read()
        
    # Return copy with code embedded
    card_with_code = card.copy()
    card_with_code["scope"] = infer_artifact_scope(card, category)
    card_with_code["resource_profile"] = normalize_resource_profile(card)
    card_with_code["code_content"] = code_content
    return card_with_code
