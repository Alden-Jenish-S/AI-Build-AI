import os
import sys
import json
from pathlib import Path

from evaluation_contract import FIDELITY_PROFILES
from .data_analyzer import discover_dataset_layout
from .llm_utils import call_llm
from .modality_scaffold import indexed_loader_source
from .prompt_context import modality_prompt_context
from .task_analyzer import TaskAnalyzer


def infer_metric_from_description(description: str) -> tuple[str, str]:
    """Infer a supported evaluation metric when a task has no explicit config."""
    description_lower = description.lower()
    if "area under" in description_lower and "roc" in description_lower:
        return "roc_auc", "maximize"
    if "root mean squared" in description_lower or "rmse" in description_lower:
        return "rmse", "minimize"
    if "mean absolute error" in description_lower or "mae" in description_lower:
        return "mae", "minimize"
    if "accuracy" in description_lower:
        return "accuracy", "maximize"
    return "score", "maximize"


class InitialAgent:
    def __init__(self, model_name: str = None):
        self.model_name = model_name

    @staticmethod
    def _extract_python_code(response: str) -> str:
        """Remove an optional Markdown fence from a generated Python file."""
        if "```python" in response:
            return response.split("```python", 1)[1].split("```", 1)[0].strip()
        if "```" in response:
            return response.split("```", 1)[1].split("```", 1)[0].strip()
        return response.strip()

    def repair_initial_algorithm(
        self,
        dataloader_path: Path,
        algorithm_path: Path,
        failure_output: str,
        metric_name: str,
        metric_direction: str,
        fidelity: str = "screen",
        task_type: str = "supervised",
        modality: str = "tabular",
    ) -> None:
        """Repair a generated baseline after a failed local execution."""
        dataloader_code = dataloader_path.read_text(encoding="utf-8")
        algorithm_code = algorithm_path.read_text(encoding="utf-8")
        system_prompt = (
            f"You are debugging a generated {modality}-ML baseline. Return the complete "
            "corrected initial_algorithm.py only, inside one Python code block. Treat "
            "initial_dataloader.py as an immutable interface: inspect its lifecycle and "
            "call it correctly. Keep the model a simple deterministic baseline, avoid "
            "train/test leakage, write a finite result.json score with the requested "
            "metric and direction, and write submission/submission.csv. Add a short "
            "leading comment explaining the failure and fix. Keep the harness-owned "
            "evaluation contract: call `X_eval, y_eval, row_ids, fold_ids, meta = "
            f"evaluation_contract.prepare_evaluation_data(train_data, '{fidelity}')`. If y_eval is None or the task "
            "is unsupervised clustering, call `eval_res = evaluate_clustering_predictions(...)` and extract "
            "the numerical float score via `score = float(eval_res['score'])` to write result.json and score the clustering outputs; otherwise write complete "
            "supervised oof_predictions.csv."
            " Emit and flush a concise progress line before and after each fold or "
            "training stage so the autonomous progress lease can supervise retries."
        )
        user_prompt = f"""
Requested metric: {metric_name}
Requested direction: {metric_direction}
Task type: {task_type}
Modality: {modality}
Required baseline fidelity: {fidelity}

Execution failure:
{failure_output[-6000:]}

Immutable initial_dataloader.py:
```python
{dataloader_code}
```

Failing initial_algorithm.py:
```python
{algorithm_code}
```
"""
        response = call_llm(
            system_prompt,
            user_prompt,
            model=self.model_name,
            temperature=0.0,
        )
        repaired_code = self._extract_python_code(response)
        if not repaired_code:
            raise ValueError("Baseline debugging returned empty Python code")
        compile(repaired_code, str(algorithm_path), "exec")
        algorithm_path.write_text(repaired_code, encoding="utf-8")

    def generate_initial_code(
        self,
        task_dir: Path,
        output_dir: Path,
        temperature: float = 0.2,
        fidelity: str = "screen",
    ):
        """
        Read immutable task inputs and write generated baseline assets to a run.
        
        Args:
            task_dir: Read-only task directory containing description/config/data
            output_dir: Run directory receiving analysis and generated code
            temperature: LLM sampling temperature (use 0.0 for reproducible baseline)
            fidelity: Harness-owned evaluation profile for the baseline
        """
        if fidelity not in FIDELITY_PROFILES:
            raise ValueError(f"unknown baseline fidelity: {fidelity!r}")
        task_dir = Path(task_dir)
        output_dir = Path(output_dir)
        task_root = task_dir.resolve()
        output_root = output_dir.resolve()
        if output_root == task_root or task_root in output_root.parents:
            raise ValueError(
                "Generated baseline output_dir must not be inside the read-only "
                f"task directory: {output_dir}"
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        analyzer = TaskAnalyzer()
        resolved_task = analyzer.resolve(task_dir)
        desc_file = task_dir / "task_description.md"
        if desc_file.exists():
            with open(desc_file, 'r', encoding='utf-8') as f:
                description = f.read()
        else:
            print(
                f"Warning: task_description.md not found at {desc_file}. "
                "Using the resolved task contract."
            )
            description = (
                f"{resolved_task.problem_type} task over "
                f"{resolved_task.modality} data with output "
                f"{resolved_task.output.type}."
            )

        task_analysis = analyzer.analyze(
            task_dir,
            output_dir=output_dir,
            include_index=resolved_task.modality != "tabular",
        )
        layout = (
            discover_dataset_layout(task_dir)
            if resolved_task.modality == "tabular"
            else {
                "roles": {
                    name: input_spec.source
                    for name, input_spec in resolved_task.inputs.items()
                }
            }
        )
        task_type = task_analysis.task_spec.problem_type
        modality = task_analysis.task_spec.modality
        roles = layout["roles"]
        expected_folds = int(FIDELITY_PROFILES[fidelity]["cv_folds"])

        # TaskAnalyzer owns discovery/profile persistence. The human-readable
        # report remains part of the generated model prompt for compatibility.
        print(
            f"InitialAgent: Resolved {task_analysis.task_spec.modality} "
            f"{task_type} task {task_analysis.task_spec.task_id!r}."
        )
        analysis_report = task_analysis.report
        print(analysis_report)
        dataset_snapshot = (
            "=== Dataset Analysis & Schema Report ===\n"
            f"{analysis_report}\n"
            "=== Canonical Task Contract ===\n"
            f"{json.dumps(task_analysis.task_spec.to_dict(), indent=2)}\n"
            "========================================\n"
        )

        metric_name = task_analysis.task_spec.primary_metric
        metric_direction = task_analysis.task_spec.metric_direction
        if metric_name == "score" and not (task_dir / "task_config.json").exists():
            metric_name, metric_direction = infer_metric_from_description(
                description
            )

        print(f"InitialAgent: Generating dataloader for task in {task_dir}...")

        role_mapping = json.dumps(roles, sort_keys=True)
        discovered_roles_info = "\n".join(
            f"- Read role '{role_name}' from './input/{filename}'."
            for role_name, filename in roles.items()
        )
        # Define the structural dictionaries returned by MyDataLoader
        loader_data_contract = (
            "The MyDataLoader class must return two dictionaries from get_data() "
            "(or when the instance is called):\n"
            "1. train_data:\n"
            "   - For supervised tasks:\n"
            "     {'X': pd.DataFrame, 'y': np.ndarray, 'row_ids': np.ndarray, "
            "      'X_val': pd.DataFrame, 'y_val': np.ndarray, 'val_row_ids': np.ndarray, "
            "      'X_full': pd.DataFrame, 'y_full': np.ndarray, 'row_ids_full': np.ndarray, "
            "      'has_val': bool, 'cat_cols': list[str], 'cont_cols': list[str], "
            "      'cat_dims': list[int], 'n_cont': int, 'task_type': str}\n"
            "   - For unsupervised clustering tasks:\n"
            "     {'X': pd.DataFrame, 'X_full': pd.DataFrame, 'y': None, 'y_full': None, "
            "      'row_ids': np.ndarray, 'row_ids_full': np.ndarray, 'has_val': False, "
            "      'task_type': 'unsupervised_clustering', 'X_val': None, 'y_val': None, 'val_row_ids': None}\n"
            "2. test_data:\n"
            "   - {'X_test': pd.DataFrame, 'test_ids': np.ndarray}\n\n"
            "Media data paths are lazy paths relative to './input/'. If non-tabular input roles are present, "
            "exposure follows the harness-generated dataset_index.jsonl where raw sample keys map to local files."
        )
        
        unsupervised_flag = (
            "This is an UNSUPERVISED CLUSTERING task.\n"
            if task_type == "unsupervised_clustering"
            else ""
        )
        
        dataloader_system = (
            "You are an expert ML engineering agent. Write a Python module 'initial_dataloader.py' containing "
            "a Class 'MyDataLoader' that reads raw data only from './input/' and preprocesses it.\n"
            f"Discovered role mapping:\n{discovered_roles_info}\n"
            f"Task Type: {task_type}, Modality: {modality}\n"
            f"{unsupervised_flag}"
            f"{loader_data_contract}\n"
            "The class MUST define a method 'get_data()' which returns train_data and test_data.\n"
            "Both `MyDataLoader()()` and `MyDataLoader().get_data()` MUST work on a fresh instance. "
            "Implement `__call__` and make `get_data()` lazily load/prepare data when necessary.\n"
            "CRITICAL CONSTRAINTS:\n"
            "- Carefully inspect the provided Dataset Analysis & Schema Report. Drop any suggested columns to drop (such as ID or constant columns) in the preprocessing stage.\n"
            "- Check for columns with missing values and ensure they are imputed appropriately.\n"
            "- If rare target classes are flagged as an inconsistency or warning, handle them carefully: DO NOT use a stratified train/test split (e.g. do not pass `stratify=y` to `train_test_split`) OR drop/filter out those rare classes from the training set entirely before splitting, as having fewer than 2 samples of a class will crash stratification.\n"
            "- If you use scikit-learn imputers (e.g. SimpleImputer), make sure to fit and transform columns in a way that avoids ValueError for feature names mismatch. "
            "For example, do NOT fit SimpleImputer on multiple columns and then transform a single-column DataFrame in a loop. Instead, transform all columns together or fit a separate imputer per column.\n"
            "- When calculating 'cat_dims' for train_data, do NOT call fit or transform methods of OrdinalEncoder/OneHotEncoder on single columns in a loop, as that leads to feature names mismatch errors. Instead, compute unique counts directly using pandas (e.g. `[int(train_df[c].nunique()) for c in cat_cols]`) or extract `len(cat) for cat in encoder.categories_`.\n"
            "- Make sure the class inherits from a base object (or stands alone) and is self-contained. Return ONLY valid Python code wrapped in a ```python code block."
        )
        
        dataloader_user = f"""
Task Description:
{description}

{dataset_snapshot}

Please write the complete 'initial_dataloader.py' file.
"""
        if modality == "tabular":
            dataloader_code = call_llm(
                dataloader_system,
                dataloader_user,
                model=self.model_name,
                temperature=temperature,
            )
            clean_loader = self._extract_python_code(dataloader_code)
        else:
            # TaskAnalyzer already resolved and validated the lightweight
            # sample index, so data loading should not be re-invented by an LLM.
            clean_loader = indexed_loader_source()
            
        loader_path = output_dir / "initial_dataloader.py"
        with open(loader_path, 'w', encoding='utf-8') as f:
            f.write(clean_loader.strip())
        print(f"InitialAgent: Saved {loader_path}")

        # 2. Generate Algorithm Skeleton
        print(f"InitialAgent: Generating algorithm skeleton for task in {task_dir}...")

        evaluation_instructions = (
            "3. Load the data using MyDataLoader:\n"
            "   loader = MyDataLoader()\n"
            "   train_data, test_data = loader()\n"
            "   X_test, test_ids = test_data['X_test'], test_data['test_ids']\n"
            "4. Obtain the harness-scheduled evaluation data and folds via `prepare_evaluation_data` from `evaluation_contract`:\n"
            f"   `X_eval, y_eval, row_ids, fold_ids, meta = prepare_evaluation_data(train_data, '{fidelity}')`.\n"
            "5. Build and train a baseline model (e.g. RandomForest, LogisticRegression, or a simple clustering algorithm depending on task type) on the training folds. Handle preprocessing, encoding, and missing values safely.\n"
            "6. Evaluate predictions using the supplied deterministic folds. If y_eval is None or the task is unsupervised clustering, "
            f"call `eval_res = evaluate_clustering_predictions(X_eval, labels, row_ids, fold_ids, fidelity='{fidelity}')` and extract the score "
            "using `score = float(eval_res['score'])`. Otherwise, save out-of-fold predictions to 'oof_predictions.csv' with columns row_id,target,prediction. "
            "CRITICAL: The columns of oof_predictions.csv MUST be named EXACTLY 'row_id', 'target', and 'prediction'. The target column must be named literally 'target' (not after any dataset-specific column name like 'Heart Disease').\n"
            "7. Train a final model on all evaluation rows and save test predictions to './submission/submission.csv'. If './input/sample_submission.csv' exists, "
            "read it and preserve its exact column names, identifier values, row order, and prediction-column order."
        )
        result_metric = "silhouette_score" if task_type == "unsupervised_clustering" else metric_name
        
        algo_system = (
            f"You are an expert {modality} ML engineering agent. Write a Python script 'initial_algorithm.py' that trains "
            "a baseline model (e.g. Scikit-Learn DecisionTree, RandomForest, or LogisticRegression) on the data "
            "provided by the MyDataLoader class from initial_dataloader.py.\n"
            "The script MUST:\n"
            "1. Import all required libraries: 'import numpy as np', 'import pandas as pd', 'import os', etc.\n"
            "2. Import MyDataLoader: 'from initial_dataloader import MyDataLoader'.\n"
            f"{evaluation_instructions}\n"
            "8. Emit and flush a concise progress line before and after every fold "
            "and training stage, then print the score in a format like "
            "'Validation Score: <float>'.\n"
            f"9. At the END of the script, write a JSON file 'result.json' in the current directory with EXACTLY this structure:\n"
            f'   {{"score": <float>, "metric": "{result_metric}", "direction": "{metric_direction}", "fidelity": "{fidelity}", "folds": {expected_folds}}}\n'
            "   Example: import json; json.dump({\"score\": 0.8521, \"metric\": \"roc_auc\", \"direction\": \"maximize\"}, open('result.json', 'w'))\n"
            f"10. Modality correctness contract:\n"
            f"{modality_prompt_context(task_analysis.task_spec, fidelity)}\n"
            "11. If you write custom encoders (e.g. SafeOrdinalEncoder), ensure they encode ALL columns specified in `cat_cols` regardless of their dtype, as columns selected as categorical may be integer-encoded in the raw data.\n"
            "Return ONLY valid Python code wrapped in a ```python code block."
        )
        
        algo_user = f"""
Task Description:
{description}

{dataset_snapshot}

Please write the complete 'initial_algorithm.py' file.
"""
        algo_code = call_llm(algo_system, algo_user, model=self.model_name, temperature=temperature)
        
        clean_algo = self._extract_python_code(algo_code)
            
        algo_path = output_dir / "initial_algorithm.py"
        with open(algo_path, 'w', encoding='utf-8') as f:
            f.write(clean_algo.strip())
        print(f"InitialAgent: Saved {algo_path}")
        
        return True
