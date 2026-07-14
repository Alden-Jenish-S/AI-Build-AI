import pandas as pd
import numpy as np
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
            if not sub_file.is_file():
                print(f"AggregatorAgent: Missing submission for {nid}: {sub_file}")
                return False
            try:
                df = pd.read_csv(sub_file)
                submissions.append(df)
                print(f"AggregatorAgent: Loaded submission for {nid}")
            except Exception as e:
                print(f"AggregatorAgent: Failed to load submission from {sub_file}: {e}")
                return False
                    
        if not submissions:
            print("AggregatorAgent WARNING: No submissions found to aggregate.")
            return False
            
        dest_file.parent.mkdir(parents=True, exist_ok=True)
        
        if len(submissions) == 1:
            submissions[0].to_csv(dest_file, index=False)
            print(f"AggregatorAgent: Copied single best submission to {dest_file}")
            return True
            
        # Average every prediction column after validating and aligning by ID.
        base_df = submissions[0].copy()
        if len(base_df.columns) < 2:
            print("AggregatorAgent: Submission must contain an ID and prediction column.")
            return False
        id_col = base_df.columns[0]
        prediction_cols = list(base_df.columns[1:])
        if base_df[id_col].duplicated().any():
            print(f"AggregatorAgent: Duplicate IDs found in base submission column {id_col!r}.")
            return False

        aligned_predictions = []
        base_ids = base_df[id_col]
        for df in submissions:
            if list(df.columns) != list(base_df.columns):
                print("AggregatorAgent: Submission schemas do not match.")
                return False
            if df[id_col].duplicated().any() or set(df[id_col]) != set(base_ids):
                print("AggregatorAgent: Submission IDs are missing, duplicated, or inconsistent.")
                return False
            aligned = df.set_index(id_col).reindex(base_ids)[prediction_cols]
            try:
                values = aligned.to_numpy(dtype=float)
            except (TypeError, ValueError):
                print("AggregatorAgent: Prediction columns must be numeric.")
                return False
            if not np.isfinite(values).all():
                print("AggregatorAgent: Predictions contain NaN or infinite values.")
                return False
            aligned_predictions.append(values)

        base_df[prediction_cols] = np.mean(aligned_predictions, axis=0)
        base_df.to_csv(dest_file, index=False)
        print(f"AggregatorAgent: Saved averaged ensemble of {len(submissions)} submissions to {dest_file}")
        return True
