import json
import hashlib
import shutil
import time

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from core.runtime_contracts import EnsembleBundle, PredictionBundle, SplitPlan
from evaluation.metrics import metric_value, normalized_metric_value
from evaluation.prediction_io import (
    legacy_prediction_payload,
    load_assignment_table,
    load_prediction_table,
    write_assignment_table,
    write_prediction_table,
    write_prediction_bundle,
)

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
                df = pd.read_csv(sub_file, keep_default_na=False)
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
        y_true: np.ndarray,
        prediction: np.ndarray,
        metric: str,
        class_names: tuple[str, ...] = (),
    ) -> float:
        """Return a metric in its natural reporting direction."""
        return metric_value(
            metric,
            y_true,
            prediction,
            class_names=class_names,
        )

    @classmethod
    def _validation_metric(
        cls,
        y_true: np.ndarray,
        prediction: np.ndarray,
        metric: str,
        class_names: tuple[str, ...] = (),
    ) -> float:
        """Return a higher-is-better objective for ensemble optimization."""
        return normalized_metric_value(
            metric,
            y_true,
            prediction,
            class_names=class_names,
        )

    @staticmethod
    def _rank_predictions(predictions: np.ndarray) -> np.ndarray:
        values = np.asarray(predictions, dtype=float)
        if values.ndim == 2:
            return np.stack(
                [
                    pd.Series(model_prediction).rank(pct=True).to_numpy()
                    for model_prediction in values
                ]
            )
        if values.ndim == 3:
            ranked = np.empty_like(values)
            for model_index in range(values.shape[0]):
                for class_index in range(values.shape[2]):
                    ranked[model_index, :, class_index] = pd.Series(
                        values[model_index, :, class_index]
                    ).rank(pct=True).to_numpy()
            return ranked
        raise ValueError("rank averaging supports scalar or class-probability OOF")

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
            try:
                frame = load_prediction_table(
                    run_root / node_id / "oof_predictions"
                )
            except FileNotFoundError:
                return None
            required = {"row_id", "target"}
            if (
                not required.issubset(frame.columns)
                or frame["row_id"].duplicated().any()
            ):
                return None
            frames.append(
                frame.set_index("row_id").sort_index()
            )
        if not frames or any(
            not frame.index.equals(frames[0].index) for frame in frames[1:]
        ):
            return None
        targets = frames[0]["target"].to_numpy()
        if any(
            not np.array_equal(
                frame["target"].to_numpy(), targets
            )
            for frame in frames[1:]
        ):
            return None
        try:
            payloads = [
                legacy_prediction_payload(frame.reset_index())
                for frame in frames
            ]
            class_names = payloads[0][1]
            if any(names != class_names for _, names in payloads[1:]):
                return None
            predictions = np.stack(
                [np.asarray(values, dtype=float) for values, _ in payloads]
            )
        except (TypeError, ValueError):
            return None
        if (
            not np.isfinite(predictions).all()
        ):
            return None
        if strategy == "rank_average":
            predictions = self._rank_predictions(predictions)

        def blend(weights: np.ndarray) -> np.ndarray:
            return np.tensordot(weights, predictions, axes=(0, 0))

        try:
            single_objectives = [
                self._validation_metric(
                    targets,
                    prediction,
                    metric_name,
                    class_names,
                )
                for prediction in predictions
            ]
        except ValueError as exc:
            print(f"AggregatorAgent WARNING: {exc}")
            return None
        single_scores = [
            self._metric_value(
                targets, prediction, metric_name, class_names
            )
            for prediction in predictions
        ]
        best_single_index = int(np.argmax(single_objectives))
        best_single_objective = single_objectives[best_single_index]

        uniform = np.full(len(frames), 1.0 / len(frames))
        uniform_prediction = blend(uniform)
        uniform_objective = self._validation_metric(
            targets, uniform_prediction, metric_name, class_names
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
                        targets,
                        blend(candidate),
                        metric_name,
                        class_names,
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
                        targets,
                        blend(candidate),
                        metric_name,
                        class_names,
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
                            targets,
                            blend(candidate),
                            metric_name,
                            class_names,
                        )
                        if objective > best_objective + 1e-12:
                            best_weights = candidate
                            best_objective = objective
                            improved = True

        final_prediction = blend(best_weights)
        final_score = self._metric_value(
            targets, final_prediction, metric_name, class_names
        )
        uniform_score = self._metric_value(
            targets, uniform_prediction, metric_name, class_names
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
            "class_names": list(class_names),
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

    def merge_nodes(
        self,
        run_root: Path,
        node_ids: List[str],
        destination_dir: Path,
        *,
        metric_name: str,
        strategy: str = "auto",
    ) -> Optional[Dict]:
        """Materialize an effective OOF-backed ensemble of measured nodes.

        This is the typed merge primitive used by ManagerAgent. It combines the
        models' prediction artifacts; it never combines or regenerates their code.
        """
        started = time.monotonic()
        if len(node_ids) < 2 or len(set(node_ids)) != len(node_ids):
            return None
        typed_bundles = []
        for node_id in node_ids:
            manifest_path = (
                run_root / node_id / "predictions" / "manifest.json"
            )
            if manifest_path.is_file():
                try:
                    typed_bundles.append(
                        PredictionBundle.from_dict(
                            json.loads(
                                manifest_path.read_text(encoding="utf-8")
                            )
                        )
                    )
                except (OSError, ValueError, TypeError, json.JSONDecodeError):
                    return None
        if typed_bundles and len(typed_bundles) != len(node_ids):
            return None
        if typed_bundles and len(
            {bundle.compatibility_key for bundle in typed_bundles}
        ) != 1:
            print(
                "AggregatorAgent: Typed prediction bundles are incompatible; "
                "refusing to merge them."
            )
            return None
        applied_strategy = self._resolve_strategy(strategy, metric_name)
        plan = self._oof_plan(
            run_root,
            node_ids,
            metric_name,
            strategy=applied_strategy,
        )
        if plan is None:
            return None
        weights = np.asarray(plan["weights"], dtype=float)
        # A one-hot plan is not a merge. The OOF guardrail deliberately returns
        # one when ensembling cannot beat the strongest constituent.
        if np.count_nonzero(weights > 1e-8) < 2:
            return None

        frames = []
        for node_id in node_ids:
            frame = load_prediction_table(
                run_root / node_id / "oof_predictions"
            )
            frames.append(frame.set_index("row_id").sort_index())
        reference = frames[0]
        if any(not frame.index.equals(reference.index) for frame in frames[1:]):
            return None
        targets = reference["target"].to_numpy()
        try:
            payloads = [
                legacy_prediction_payload(frame.reset_index())
                for frame in frames
            ]
            class_names = payloads[0][1]
            if any(names != class_names for _, names in payloads[1:]):
                return None
            prediction_matrix = np.stack(
                [np.asarray(values, dtype=float) for values, _ in payloads]
            )
        except (TypeError, ValueError):
            return None
        if applied_strategy == "rank_average":
            prediction_matrix = self._rank_predictions(prediction_matrix)
        merged_prediction = np.tensordot(
            weights, prediction_matrix, axes=(0, 0)
        )
        ensemble_objective = self._validation_metric(
            targets, merged_prediction, metric_name, class_names
        )
        best_single_objective = max(
            self._validation_metric(
                targets, prediction, metric_name, class_names
            )
            for prediction in prediction_matrix
        )
        if ensemble_objective <= best_single_objective + 1e-12:
            return None

        if "fold_id" in reference.columns:
            fold_ids = reference["fold_id"].to_numpy(dtype=int)
        else:
            assignments = (
                load_assignment_table(
                    run_root / node_ids[0] / "fold_assignments"
                )
                .set_index("row_id")
                .reindex(reference.index)
            )
            if assignments["fold_id"].isna().any():
                return None
            fold_ids = assignments["fold_id"].to_numpy(dtype=int)
        for frame in frames[1:]:
            if "fold_id" in frame.columns and not np.array_equal(
                frame["fold_id"].to_numpy(dtype=int), fold_ids
            ):
                return None

        destination_dir.mkdir(parents=True, exist_ok=True)
        write_prediction_table(
            destination_dir / "oof_predictions.npz",
            sample_ids=reference.index.to_numpy(),
            targets=targets,
            predictions=merged_prediction,
            fold_ids=fold_ids,
            class_names=class_names,
        )
        write_assignment_table(
            destination_dir / "fold_assignments.npz",
            sample_ids=reference.index.to_numpy(),
            fold_ids=fold_ids,
        )

        submission_path = destination_dir / "submission" / "submission.csv"
        if not self.aggregate_submissions(
            run_root,
            node_ids,
            submission_path,
            weights=weights.tolist(),
            strategy=applied_strategy,
        ):
            return None
        source_manifest = run_root / node_ids[0] / "evaluation_manifest.json"
        if source_manifest.is_file():
            shutil.copy2(
                source_manifest, destination_dir / "evaluation_manifest.json"
            )

        fold_scores = [
            self._metric_value(
                targets[fold_ids == fold_id],
                merged_prediction[fold_ids == fold_id],
                metric_name,
                class_names,
            )
            for fold_id in sorted(np.unique(fold_ids))
        ]
        score = float(np.mean(fold_scores))
        merge_manifest = {
            "operator": "merge_ensemble",
            "manager_owned": True,
            "source_node_ids": node_ids,
            "requested_strategy": strategy,
            "strategy": applied_strategy,
            "weights": weights.tolist(),
            "oof_scores": plan["oof_scores"],
            "ensemble_oof_score": plan["ensemble_oof_score"],
            "best_single_node_id": plan["best_single_node_id"],
            "best_single_oof_score": plan["best_single_oof_score"],
            "raw_code_fusion": False,
        }
        (destination_dir / "merge_manifest.json").write_text(
            json.dumps(merge_manifest, indent=2) + "\n", encoding="utf-8"
        )
        if typed_bundles:
            split_plan = SplitPlan(
                assignments={
                    str(sample_id): int(fold)
                    for sample_id, fold in zip(
                        reference.index.to_numpy(), fold_ids
                    )
                },
                strategy="manager_owned_ensemble",
                seed=42,
                split_fingerprint=typed_bundles[0].split_fingerprint,
            )
            prediction_bundle = write_prediction_bundle(
                destination_dir,
                task_fingerprint=typed_bundles[0].task_fingerprint,
                split_plan=split_plan,
                output_type=typed_bundles[0].output_type,
                sample_ids=[
                    str(item) for item in reference.index.to_numpy()
                ],
                predictions=merged_prediction,
                targets=targets,
                fold_ids=fold_ids,
                class_names=(
                    class_names or typed_bundles[0].class_names
                ),
                metadata={
                    "operator": "merge_ensemble",
                    "source_node_ids": node_ids,
                },
            )
            compatibility_key = prediction_bundle.compatibility_key
        else:
            compatibility_key = hashlib.sha256(
                json.dumps(
                    {
                        "row_ids": [
                            str(item)
                            for item in reference.index.to_numpy()
                        ],
                        "metric": metric_name,
                    },
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest()

        component_bundle_paths = []
        for node_id in node_ids:
            model_manifest = run_root / node_id / "model" / "manifest.json"
            component_bundle_paths.append(
                str(model_manifest)
                if model_manifest.is_file()
                else str(run_root / node_id / "algorithm.py")
            )
        ensemble_bundle = EnsembleBundle(
            strategy=applied_strategy,
            component_nodes=tuple(node_ids),
            component_bundles=tuple(component_bundle_paths),
            output_type=(
                typed_bundles[0].output_type
                if typed_bundles
                else "scalar_predictions"
            ),
            compatibility_key=compatibility_key,
            weights=tuple(float(value) for value in weights),
            inference_order=tuple([*node_ids, "weighted_combiner"]),
            metadata={
                "manager_owned": True,
                "raw_code_fusion": False,
                "metric_name": metric_name,
            },
        )
        model_dir = destination_dir / "model"
        model_dir.mkdir(parents=True, exist_ok=True)
        ensemble_manifest_path = model_dir / "manifest.json"
        ensemble_manifest_path.write_text(
            json.dumps(
                ensemble_bundle.to_dict(), indent=2, sort_keys=True
            )
            + "\n",
            encoding="utf-8",
        )
        return {
            "status": "completed",
            "score": score,
            "validation": {
                "cv_mean": score,
                "cv_std": float(np.std(fold_scores)),
                "folds": len(fold_scores),
                "fold_scores": [float(value) for value in fold_scores],
            },
            "oof_path": str(destination_dir / "oof_predictions.npz"),
            "code_path": str(destination_dir / "merge_manifest.json"),
            "diagnostics": (
                "ManagerAgent ensembled measured node predictions using "
                f"{applied_strategy} with OOF-selected weights."
            ),
            "elapsed_seconds": time.monotonic() - started,
            "merge": merge_manifest,
            "prediction_bundle": (
                str(destination_dir / "predictions" / "manifest.json")
                if typed_bundles
                else None
            ),
            "model_bundle": str(ensemble_manifest_path),
            "compatibility_key": compatibility_key,
        }

    def aggregate_ranked_candidates(
        self,
        run_root: Path,
        candidates: List[Dict],
        dest_file: Path,
        maximize: bool = True,
        top_k: int = 3,
        strategy: str = "auto",
        metric_name: str | None = None,
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
            frame = pd.read_csv(submission_path, keep_default_na=False)
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
            try:
                vector = aligned.to_numpy(dtype=float).reshape(-1)
            except (TypeError, ValueError):
                # Structured encodings such as RLE masks cannot be averaged in
                # CSV space. Preserve the strongest ranked candidate and let a
                # typed model/output merger handle any future structured blend.
                if not selected:
                    selected.append(node_id)
                break
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
        applied_strategy = self._resolve_strategy(
            strategy, metric_name or ""
        )
        oof_plan = (
            self._oof_plan(
                run_root,
                selected,
                metric_name,
                strategy=applied_strategy,
            )
            if metric_name
            else None
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
