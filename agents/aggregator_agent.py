import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional

class AggregatorAgent:
    def __init__(self):
        self.last_ensemble_manifest: Dict = {}

    @staticmethod
    def _rank_compatible_metric(metric_name: str) -> bool:
        """Rank averaging is useful for ranking metrics, not calibrated values."""
        return "auc" in str(metric_name).lower()

    @classmethod
    def _resolve_strategy(cls, strategy: str, metric_name: str) -> str:
        if strategy == "auto":
            return (
                "rank_average"
                if cls._rank_compatible_metric(metric_name)
                else "average"
            )
        if strategy == "rank_average" and not cls._rank_compatible_metric(
            metric_name
        ):
            print(
                "AggregatorAgent WARNING: rank_average is incompatible with "
                f"metric {metric_name!r}; preserving prediction scale with average."
            )
            return "average"
        return strategy

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
    def _metric_value(
        y_true: np.ndarray, prediction: np.ndarray, metric: str
    ) -> float:
        """Return a metric in its natural reporting direction."""
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
            return float(np.mean(np.abs(y_true - prediction)))
        if "rmse" in metric:
            return float(np.sqrt(np.mean((y_true - prediction) ** 2)))
        if "log_loss" in metric or "logloss" in metric:
            clipped = np.clip(prediction, 1e-12, 1 - 1e-12)
            return -float(
                np.mean(
                    y_true * np.log(clipped)
                    + (1 - y_true) * np.log(1 - clipped)
                )
            )
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
        raise ValueError(f"unsupported OOF ensemble metric: {metric!r}")

    @classmethod
    def _validation_metric(
        cls, y_true: np.ndarray, prediction: np.ndarray, metric: str
    ) -> float:
        """Return a higher-is-better objective for ensemble optimization."""
        value = cls._metric_value(y_true, prediction, metric)
        normalized = metric.lower()
        if (
            "mae" in normalized
            or "rmse" in normalized
            or "log_loss" in normalized
            or "logloss" in normalized
        ):
            return -value
        return value

    @staticmethod
    def _rank_predictions(predictions: np.ndarray) -> np.ndarray:
        return np.stack(
            [
                pd.Series(model_prediction).rank(pct=True).to_numpy()
                for model_prediction in predictions
            ]
        )

    def _oof_plan(
        self,
        run_root: Path,
        node_ids: List[str],
        metric_name: str,
        strategy: str,
    ) -> Optional[Dict]:
        """Optimize a blend on aligned OOF rows and compare it to every single."""
        frames = []
        for node_id in node_ids:
            path = run_root / node_id / "oof_predictions.csv"
            if not path.is_file():
                return None
            frame = pd.read_csv(path)
            required = {"row_id", "target", "prediction"}
            if (
                not required.issubset(frame.columns)
                or frame["row_id"].duplicated().any()
            ):
                return None
            frames.append(
                frame[["row_id", "target", "prediction"]]
                .set_index("row_id")
                .sort_index()
            )
        if not frames or any(
            not frame.index.equals(frames[0].index) for frame in frames[1:]
        ):
            return None
        targets = frames[0]["target"].to_numpy(dtype=float)
        if any(
            not np.array_equal(
                frame["target"].to_numpy(dtype=float), targets
            )
            for frame in frames[1:]
        ):
            return None
        predictions = np.stack(
            [frame["prediction"].to_numpy(dtype=float) for frame in frames]
        )
        if (
            not np.isfinite(targets).all()
            or not np.isfinite(predictions).all()
        ):
            return None
        if strategy == "rank_average":
            predictions = self._rank_predictions(predictions)

        try:
            single_objectives = [
                self._validation_metric(targets, prediction, metric_name)
                for prediction in predictions
            ]
        except ValueError as exc:
            print(f"AggregatorAgent WARNING: {exc}")
            return None
        single_scores = [
            self._metric_value(targets, prediction, metric_name)
            for prediction in predictions
        ]
        best_single_index = int(np.argmax(single_objectives))
        best_single_objective = single_objectives[best_single_index]

        uniform = np.full(len(frames), 1.0 / len(frames))
        uniform_prediction = uniform @ predictions
        uniform_objective = self._validation_metric(
            targets, uniform_prediction, metric_name
        )
        best_weights = uniform
        best_objective = uniform_objective

        # Always include every single model as a candidate. This is the primary
        # guardrail: an OOF-selected ensemble can never score below the strongest
        # constituent merely because an optimizer or blend is unhelpful.
        for index, objective in enumerate(single_objectives):
            if objective > best_objective + 1e-12:
                best_weights = np.eye(len(frames))[index]
                best_objective = objective

        # SLSQP efficiently solves smooth constrained blends such as RMSE and
        # log-loss. The deterministic transfer search below remains the fallback
        # and also handles threshold/ranking metrics.
        if len(frames) > 1:
            try:
                from scipy.optimize import minimize

                optimized = minimize(
                    lambda candidate: -self._validation_metric(
                        targets, candidate @ predictions, metric_name
                    ),
                    x0=uniform,
                    method="SLSQP",
                    bounds=[(0.0, 1.0)] * len(frames),
                    constraints={
                        "type": "eq",
                        "fun": lambda candidate: float(candidate.sum() - 1.0),
                    },
                    options={"maxiter": 200, "ftol": 1e-12},
                )
                candidate = np.asarray(optimized.x, dtype=float)
                if (
                    optimized.success
                    and np.isfinite(candidate).all()
                    and candidate.sum() > 0
                ):
                    candidate = np.clip(candidate, 0.0, None)
                    candidate /= candidate.sum()
                    objective = self._validation_metric(
                        targets, candidate @ predictions, metric_name
                    )
                    if objective > best_objective + 1e-12:
                        best_weights, best_objective = candidate, objective
            except Exception as exc:
                print(
                    "AggregatorAgent WARNING: Continuous OOF weight "
                    f"optimization failed; using deterministic search: {exc}"
                )

        # Transfer mass in both directions. The old implementation could only
        # increase a coordinate and renormalize all others, which missed useful
        # mixtures and made its result dependent on coordinate order.
        for step in (0.20, 0.10, 0.05, 0.02, 0.01):
            improved = True
            while improved:
                improved = False
                for source in range(len(best_weights)):
                    for target in range(len(best_weights)):
                        if source == target or best_weights[source] <= 0:
                            continue
                        amount = min(step, best_weights[source])
                        candidate = best_weights.copy()
                        candidate[source] -= amount
                        candidate[target] += amount
                        objective = self._validation_metric(
                            targets, candidate @ predictions, metric_name
                        )
                        if objective > best_objective + 1e-12:
                            best_weights = candidate
                            best_objective = objective
                            improved = True

        final_prediction = best_weights @ predictions
        final_score = self._metric_value(
            targets, final_prediction, metric_name
        )
        uniform_score = self._metric_value(
            targets, uniform_prediction, metric_name
        )
        guardrail_applied = (
            uniform_objective < best_single_objective - 1e-12
            and np.count_nonzero(best_weights > 1e-8) == 1
        )
        return {
            "weights": best_weights.tolist(),
            "oof_scores": {
                node_id: score
                for node_id, score in zip(node_ids, single_scores)
            },
            "uniform_oof_score": uniform_score,
            "ensemble_oof_score": final_score,
            "best_single_node_id": node_ids[best_single_index],
            "best_single_oof_score": single_scores[best_single_index],
            "guardrail_applied": guardrail_applied,
            "guardrail_reason": (
                "The candidate blend did not beat the best single model on "
                "aligned OOF rows."
                if guardrail_applied
                else None
            ),
        }

    def _oof_weights(
        self,
        run_root: Path,
        node_ids: List[str],
        metric_name: str,
        strategy: str = "average",
    ) -> Optional[List[float]]:
        plan = self._oof_plan(
            run_root, node_ids, metric_name, strategy=strategy
        )
        return plan["weights"] if plan else None

    def aggregate_ranked_candidates(
        self,
        run_root: Path,
        candidates: List[Dict],
        dest_file: Path,
        maximize: bool = True,
        top_k: int = 3,
        strategy: str = "auto",
        metric_name: str = "score",
        correlation_limit: float = 0.995,
    ) -> List[str]:
        """Select strong, prediction-diverse candidates and aggregate them."""
        self.last_ensemble_manifest = {}
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
        applied_strategy = self._resolve_strategy(strategy, metric_name)
        oof_plan = self._oof_plan(
            run_root,
            selected,
            metric_name,
            strategy=applied_strategy,
        )
        if oof_plan is None:
            # Blind equal-weight blending has no evidence that it improves the
            # selected model. Fall back to the strongest reported candidate.
            selected = selected[:1]
            weights = None
            applied_strategy = "average"
            oof_plan = {
                "weights": [1.0],
                "oof_scores": {},
                "uniform_oof_score": None,
                "ensemble_oof_score": None,
                "best_single_node_id": selected[0],
                "best_single_oof_score": None,
                "guardrail_applied": True,
                "guardrail_reason": (
                    "Aligned OOF predictions were unavailable or the metric was "
                    "unsupported; used the strongest reported candidate."
                ),
            }
        else:
            weights = oof_plan["weights"]
        if not self.aggregate_submissions(
            run_root,
            selected,
            dest_file,
            weights=weights,
            strategy=applied_strategy,
        ):
            return []
        self.last_ensemble_manifest = {
            "requested_strategy": strategy,
            "strategy": applied_strategy,
            "metric_name": metric_name,
            "node_ids": selected,
            **oof_plan,
        }
        return selected
