import json
from pathlib import Path

# Base directory setup relative to this file
BASE_DIR = Path(__file__).resolve().parent

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
        card_path = BASE_DIR / "l2_store" / category / f"{artifact_id}.json"
        if card_path.exists():
            try:
                with open(card_path, 'r', encoding='utf-8') as f:
                    card = json.load(f)
                # Keep only search/selection metadata, drop the source code or heavy details
                summary = {
                    "artifact_id": card["artifact_id"],
                    "category": card["category"],
                    "description": card["description"],
                    "interface": card["interface"],
                    "verified": card.get("verified", False),
                    "known_pitfalls": card.get("known_pitfalls", [])
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
    card_path = BASE_DIR / "l2_store" / category / f"{artifact_id}.json"
    if not card_path.exists():
        raise FileNotFoundError(f"L2 model card not found at {card_path}")
        
    with open(card_path, 'r', encoding='utf-8') as f:
        card = json.load(f)
        
    code_path = card_path.parent / card["code_path"]
    if not code_path.exists():
        raise FileNotFoundError(f"Code file not found at {code_path}")
        
    with open(code_path, 'r', encoding='utf-8') as f:
        code_content = f.read()
        
    # Return copy with code embedded
    card_with_code = card.copy()
    card_with_code["code_content"] = code_content
    return card_with_code
