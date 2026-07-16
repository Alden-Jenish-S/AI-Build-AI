import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional

class AggregatorAgent:
    def __init__(self):
        pass

    def aggregate_submissions(
        self,
        run_root: Path,
        leaf_node_ids: List[str],
        dest_file: Path,
        weights: Optional[List[float]] = None,
        strategy: str = "average",
    ) -> bool:
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
            if strategy == "rank_average":
                values = np.column_stack(
                    [pd.Series(values[:, index]).rank(pct=True).to_numpy()
                     for index in range(values.shape[1])]
                )
            elif strategy != "average":
                print(f"AggregatorAgent: Unknown strategy {strategy!r}.")
                return False
            aligned_predictions.append(values)

        if weights is None:
            normalized_weights = np.full(len(aligned_predictions), 1.0 / len(aligned_predictions))
        else:
            normalized_weights = np.asarray(weights, dtype=float)
            if (
                normalized_weights.shape != (len(aligned_predictions),)
                or not np.isfinite(normalized_weights).all()
                or (normalized_weights < 0).any()
                or normalized_weights.sum() <= 0
            ):
                print("AggregatorAgent: Invalid ensemble weights.")
                return False
            normalized_weights = normalized_weights / normalized_weights.sum()
        base_df[prediction_cols] = np.tensordot(
            normalized_weights, np.stack(aligned_predictions), axes=(0, 0)
        )
        base_df.to_csv(dest_file, index=False)
        print(
            f"AggregatorAgent: Saved {strategy} ensemble of {len(submissions)} "
            f"submissions to {dest_file} with weights={normalized_weights.tolist()}"
        )
        return True

    @staticmethod
    def _validation_metric(y_true: np.ndarray, prediction: np.ndarray, metric: str) -> float:
        metric = metric.lower()
        if "auc" in metric:
            y_true = np.asarray(y_true, dtype=float)
            ranks = pd.Series(prediction).rank(method="average").to_numpy()
            positives = y_true == 1
            n_pos = int(positives.sum())
            n_neg = len(y_true) - n_pos
            if not n_pos or not n_neg:
                raise ValueError("AUC requires both target classes")
            return float(
                (ranks[positives].sum() - n_pos * (n_pos + 1) / 2)
                / (n_pos * n_neg)
            )
        if "mae" in metric:
            return -float(np.mean(np.abs(y_true - prediction)))
        if "log_loss" in metric or "logloss" in metric:
            clipped = np.clip(prediction, 1e-12, 1 - 1e-12)
            return float(np.mean(y_true * np.log(clipped) + (1 - y_true) * np.log(1 - clipped)))
        if "accuracy" in metric:
            return float(np.mean((prediction >= 0.5) == y_true))
        if metric in {"f1", "f1_score"}:
            labels = prediction >= 0.5
            true_positive = np.sum(labels & (y_true == 1))
            denominator = 2 * true_positive + np.sum(labels & (y_true == 0)) + np.sum((~labels) & (y_true == 1))
            return float(2 * true_positive / denominator) if denominator else 0.0
        if metric in {"r2", "r2_score"}:
            denominator = np.sum((y_true - np.mean(y_true)) ** 2)
            return float(1 - np.sum((y_true - prediction) ** 2) / denominator) if denominator else 0.0
        return -float(np.sqrt(np.mean((y_true - prediction) ** 2)))

    def _oof_weights(
        self, run_root: Path, node_ids: List[str], metric_name: str
    ) -> Optional[List[float]]:
        frames = []
        for node_id in node_ids:
            path = run_root / node_id / "oof_predictions.csv"
            if not path.is_file():
                return None
            frame = pd.read_csv(path)
            required = {"row_id", "target", "prediction"}
            if not required.issubset(frame.columns) or frame["row_id"].duplicated().any():
                return None
            frames.append(frame[list(required)].set_index("row_id").sort_index())
        if not frames or any(not frame.index.equals(frames[0].index) for frame in frames[1:]):
            return None
        targets = frames[0]["target"].to_numpy(dtype=float)
        if any(not np.array_equal(frame["target"].to_numpy(dtype=float), targets) for frame in frames[1:]):
            return None
        predictions = np.stack(
            [frame["prediction"].to_numpy(dtype=float) for frame in frames]
        )
        weights = np.full(len(frames), 1.0 / len(frames))
        best = self._validation_metric(targets, weights @ predictions, metric_name)
        # Deterministic coordinate hill climbing is cheap and robust for small ensembles.
        for step in (0.20, 0.10, 0.05):
            improved = True
            while improved:
                improved = False
                for index in range(len(weights)):
                    candidate = weights.copy()
                    candidate[index] += step
                    candidate /= candidate.sum()
                    value = self._validation_metric(
                        targets, candidate @ predictions, metric_name
                    )
                    if value > best + 1e-12:
                        weights, best, improved = candidate, value, True
        return weights.tolist()

    def aggregate_ranked_candidates(
        self,
        run_root: Path,
        candidates: List[Dict],
        dest_file: Path,
        maximize: bool = True,
        top_k: int = 3,
        strategy: str = "rank_average",
        metric_name: str = "score",
        correlation_limit: float = 0.995,
    ) -> List[str]:
        """Select strong, prediction-diverse candidates and aggregate them."""
        ordered = sorted(
            candidates,
            key=lambda item: float(item["score"]),
            reverse=maximize,
        )
        selected: List[str] = []
        selected_vectors: List[np.ndarray] = []
        reference_ids = None
        reference_columns = None
        for item in ordered:
            node_id = item["node_id"]
            submission_path = run_root / node_id / "submission" / "submission.csv"
            if not submission_path.is_file():
                continue
            frame = pd.read_csv(submission_path)
            if len(frame.columns) < 2:
                continue
            id_col = frame.columns[0]
            prediction_columns = list(frame.columns[1:])
            if frame[id_col].duplicated().any():
                continue
            if reference_ids is None:
                reference_ids = frame[id_col].copy()
                reference_columns = list(frame.columns)
                aligned = frame[prediction_columns]
            else:
                if list(frame.columns) != reference_columns or set(frame[id_col]) != set(reference_ids):
                    continue
                aligned = frame.set_index(id_col).reindex(reference_ids)[prediction_columns]
            vector = aligned.to_numpy(dtype=float).reshape(-1)
            if not np.isfinite(vector).all():
                continue
            too_correlated = False
            for previous in selected_vectors:
                if np.std(vector) == 0 or np.std(previous) == 0:
                    correlation = 1.0
                else:
                    correlation = abs(float(np.corrcoef(vector, previous)[0, 1]))
                if correlation >= correlation_limit:
                    too_correlated = True
                    break
            if too_correlated and selected:
                continue
            selected.append(node_id)
            selected_vectors.append(vector)
            if len(selected) >= max(1, top_k):
                break
        if not selected:
            return []
        weights = self._oof_weights(run_root, selected, metric_name)
        if not self.aggregate_submissions(
            run_root,
            selected,
            dest_file,
            weights=weights,
            strategy=strategy,
        ):
            return []
        return selected
