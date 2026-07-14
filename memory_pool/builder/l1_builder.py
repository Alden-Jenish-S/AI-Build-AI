import json
from pathlib import Path
from agents.llm_utils import call_llm

class L1Builder:
    def __init__(self, project_root: Path, model_name: str = None):
        self.project_root = project_root
        self.model_name = model_name
        self.l1_path = project_root / "memory_pool" / "l1_index.json"
        self.l2_store = project_root / "memory_pool" / "l2_store"

    def rebuild_l1_descriptions(self) -> bool:
        """
        Reads all categories and updates their high-level descriptions by summarizing 
        their verified L2 artifacts.
        """
        if not self.l1_path.exists():
            print("L1Builder ERROR: l1_index.json not found.")
            return False
            
        with open(self.l1_path, 'r', encoding='utf-8') as f:
            l1_index = json.load(f)
            
        print("L1Builder: Updating L1 category descriptions...")
        
        for category, details in l1_index.items():
            pointers = details.get("l2_pointers", [])
            if not pointers:
                continue
                
            # Collect verified L2 descriptions
            cards_summary = []
            for art_id in pointers:
                card_file = self.l2_store / category / f"{art_id}.json"
                if card_file.exists():
                    try:
                        with open(card_file, 'r', encoding='utf-8') as f:
                            card = json.load(f)
                        if card.get("verified", False):
                            cards_summary.append(f"- {art_id}: {card.get('description', '')}")
                    except Exception as e:
                        print(f"L1Builder: Failed to load card {art_id}: {e}")
                        
            if not cards_summary:
                print(f"L1Builder: Skipping '{category}' (no verified L2 artifacts).")
                continue
                
            artifacts_list_str = "\n".join(cards_summary)
            
            system_prompt = (
                "You are the L1 Category Index Builder. Your task is to write a short, precise "
                "1-2 sentence high-level description for a category in the memory pool, summarizing "
                "the set of verified code artifacts it contains. Do not list individual artifact names, "
                "just describe the unified purpose of the category."
            )
            
            user_prompt = f"""
Category Name: {category}
Verified Code Artifacts in this category:
{artifacts_list_str}

Please generate the high-level description for this category.
"""
            summary_desc = call_llm(system_prompt, user_prompt, model=self.model_name).strip()
            details["description"] = summary_desc
            print(f"L1Builder: Updated category '{category}' description: {summary_desc}")
            
        # Save updated index
        with open(self.l1_path, 'w', encoding='utf-8') as f:
            json.dump(l1_index, f, indent=2)
            
        print("L1Builder: Successfully rebuilt L1 category index!")
        return True
