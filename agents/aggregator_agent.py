import pandas as pd
from pathlib import Path
from typing import List, Dict

class AggregatorAgent:
    def __init__(self):
        pass

    def aggregate_submissions(self, run_root: Path, leaf_node_ids: List[str], dest_file: Path) -> bool:
        """
        Loads the submission.csv predictions from each leaf node's run folder, 
        and averages them to generate a final ensembled submission.csv.
        If only one leaf is provided, it copies it directly.
        """
        submissions = []
        for nid in leaf_node_ids:
            sub_file = run_root / nid / "submission" / "submission.csv"
            if sub_file.exists():
                try:
                    df = pd.read_csv(sub_file)
                    submissions.append(df)
                    print(f"AggregatorAgent: Loaded submission for {nid}")
                except Exception as e:
                    print(f"AggregatorAgent: Failed to load submission from {sub_file}: {e}")
                    
        if not submissions:
            print("AggregatorAgent WARNING: No submissions found to aggregate.")
            return False
            
        dest_file.parent.mkdir(parents=True, exist_ok=True)
        
        if len(submissions) == 1:
            submissions[0].to_csv(dest_file, index=False)
            print(f"AggregatorAgent: Copied single best submission to {dest_file}")
            return True
            
        # Standard average ensembling
        base_df = submissions[0].copy()
        pred_col = base_df.columns[1] # e.g. target or label
        
        preds_sum = base_df[pred_col].values.astype(float)
        for df in submissions[1:]:
            preds_sum += df[pred_col].values.astype(float)
            
        base_df[pred_col] = preds_sum / len(submissions)
        base_df.to_csv(dest_file, index=False)
        print(f"AggregatorAgent: Saved averaged ensemble of {len(submissions)} submissions to {dest_file}")
        return True
