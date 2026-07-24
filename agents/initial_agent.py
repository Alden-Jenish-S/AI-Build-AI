import os
import sys
import json
from pathlib import Path

from evaluation_contract import FIDELITY_PROFILES
from .data_analyzer import discover_dataset_layout, run_dataset_analysis
from .llm_utils import call_llm


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
    ) -> None:
        """Repair a generated baseline after a failed local execution."""
        dataloader_code = dataloader_path.read_text(encoding="utf-8")
        algorithm_code = algorithm_path.read_text(encoding="utf-8")
        system_prompt = (
            "You are debugging a generated tabular-ML baseline. Return the complete "
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
        desc_file = task_dir / "task_description.md"
        if not desc_file.exists():
            print(f"Warning: task_description.md not found at {desc_file}. Using generic description.")
            description = f"Predict the target variable for the {task_dir.name} tabular dataset. Please infer the exact details from the train.csv and test.csv files in the input directory."
        else:
            with open(desc_file, 'r', encoding='utf-8') as f:
                description = f.read()

        layout = discover_dataset_layout(task_dir)
        task_type = layout["task_type"]
        roles = layout["roles"]
        expected_folds = int(FIDELITY_PROFILES[fidelity]["cv_folds"])

        # Run dataset analysis and profiling
        report_file = output_dir / "dataset_analysis_report.txt"
        try:
            print(f"InitialAgent: Checking/running dataset analysis for {task_dir.name}...")
            if not report_file.exists():
                analysis_report = run_dataset_analysis(task_dir)
                with open(report_file, 'w', encoding='utf-8') as f:
                    f.write(analysis_report)
            else:
                with open(report_file, 'r', encoding='utf-8') as f:
                    analysis_report = f.read()
            
            print(analysis_report)  # Print report to stdout as tool call output visible to user
            dataset_snapshot = (
                "=== Dataset Analysis & Schema Report ===\n"
                f"{analysis_report}\n"
                "========================================\n"
            )
        except Exception as e:
            print(f"InitialAgent WARNING: Could not generate dataset analysis report: {e}")
            dataset_snapshot = ""

        # Load task config for metric info (if available)
        config_file = task_dir / "task_config.json"
        metric_name = "score"
        metric_direction = "maximize"
        if config_file.exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                task_config = json.load(f)
            metric_name = task_config.get("metric_name", "score")
            metric_direction = task_config.get("metric_direction", "maximize")
        else:
            metric_name, metric_direction = infer_metric_from_description(description)

        print(f"InitialAgent: Generating dataloader for task in {task_dir}...")

        role_mapping = json.dumps(roles, sort_keys=True)
        if task_type == "unsupervised_clustering":
            loader_data_contract = (
                "This is an UNSUPERVISED CLUSTERING task. Read the primary table "
                f"from './input/{roles.get('data', 'data.csv')}'. Do not look for "
                "train.csv/test.csv, do not invent a target, and do not split the "
                "population. Return train_data with X and X_full set to the complete "
                "feature frame, y and y_full set to None, stable row_ids/row_ids_full, "
                "task_type='unsupervised_clustering', has_val=False, and empty "
                "validation fields. Return test_data with X_test equal to the complete "
                "feature population and test_ids aligned to the sample-submission ID "
                "column (or the source ID column)."
            )
        else:
            loader_data_contract = (
                "This is a supervised task. Read the exact train/test role paths from "
                "the mapping below rather than assuming conventional filenames. "
                "Return train_data as {'X': pd.DataFrame, 'y': np.ndarray, "
                "'row_ids': np.ndarray, 'X_val': pd.DataFrame, 'y_val': np.ndarray, "
                "'val_row_ids': np.ndarray, 'X_full': pd.DataFrame, "
                "'y_full': np.ndarray, 'row_ids_full': np.ndarray, 'has_val': bool, "
                "'cat_cols': list_of_str, 'cont_cols': list_of_str, "
                f"'cat_dims': list_of_int, 'n_cont': int, 'task_type': '{task_type}'}}; "
                "return test_data as "
                "{'X_test': pd.DataFrame, 'test_ids': np.ndarray/pd.Series}."
            )
        
        dataloader_system = (
            "You are an expert ML engineering agent. Write a Python module 'initial_dataloader.py' containing "
            "a Class 'MyDataLoader' that reads raw data only from './input/' and preprocesses it.\n"
            f"Discovered role mapping: {role_mapping}\n"
            f"{loader_data_contract}\n"
            "The class MUST define a method 'get_data()' which returns two dicts: train_data and test_data.\n"
            "Both `MyDataLoader()()` and `MyDataLoader().get_data()` MUST work on a fresh instance. "
            "Implement `__call__` and make `get_data()` lazily load/prepare data when necessary; do not require callers to invoke a private method first.\n"
            "CRITICAL CONSTRAINTS:\n"
            "- Carefully inspect the provided Dataset Analysis & Schema Report. Drop any suggested columns to drop (such as ID or constant columns) in the preprocessing stage.\n"
            "- If rare target classes are flagged as an inconsistency or warning, handle them carefully: DO NOT use a stratified train/test split (e.g. do not pass `stratify=y` to `train_test_split`) OR drop/filter out those rare classes from the training set entirely before splitting, as having fewer than 2 samples of a class will crash stratification.\n"
            "- Check for columns with missing values and ensure they are imputed appropriately.\n"
            "- If you use scikit-learn imputers (e.g. SimpleImputer), make sure to fit and transform columns in a way that avoids ValueError for feature names mismatch. "
            "For example, do NOT fit SimpleImputer on multiple columns and then transform a single-column DataFrame in a loop. Instead, transform all columns together or fit a separate imputer per column.\n"
            "- Make sure the class inherits from a base object (or stands alone) and is self-contained. Return ONLY valid Python code wrapped in a ```python code block."
        )
        
        dataloader_user = f"""
Task Description:
{description}

{dataset_snapshot}

Please write the complete 'initial_dataloader.py' file.
"""
        dataloader_code = call_llm(dataloader_system, dataloader_user, model=self.model_name, temperature=temperature)
        
        clean_loader = self._extract_python_code(dataloader_code)
            
        loader_path = output_dir / "initial_dataloader.py"
        with open(loader_path, 'w', encoding='utf-8') as f:
            f.write(clean_loader.strip())
        print(f"InitialAgent: Saved {loader_path}")

        # 2. Generate Algorithm Skeleton
        print(f"InitialAgent: Generating algorithm skeleton for task in {task_dir}...")

        if task_type == "unsupervised_clustering":
            evaluation_instructions = (
                "3. Instantiate `MyDataLoader`, obtain train_data/test_data, then "
                "unpack: `X_eval, y_eval, row_ids, fold_ids, meta = "
                f"prepare_evaluation_data(train_data, '{fidelity}')`. y_eval "
                "must be None. Build a simple deterministic clustering baseline such "
                "as MiniBatchKMeans using the bounded scheduled X_eval rows. Encode "
                "mixed columns and impute missing values without using hidden labels. "
                "Call `eval_res = evaluate_clustering_predictions(X_eval, labels, row_ids, "
                f"fold_ids, fidelity='{fidelity}')`; note that evaluate_clustering_predictions returns a dict containing 'score' "
                "(and writes the independently verifiable OOF and silhouette-proxy files). Extract `score = float(eval_res['score'])`. "
                "Fit the final clustering pipeline and predict one integer cluster for every X_test row. "
                "Preserve the sample-submission columns and row order exactly."
            )
            result_metric = "silhouette_score"
        else:
            evaluation_instructions = (
                "3. Instantiate and load data EXACTLY like this:\n"
                "   loader = MyDataLoader()\n"
                "   train_data, test_data = loader()\n"
                "   X_train, y_train = train_data['X'], train_data['y']\n"
                "   X_val, y_val = train_data['X_val'], train_data['y_val']\n"
                "   X_test, test_ids = test_data['X_test'], test_data['test_ids']\n"
                "   Then import `prepare_evaluation_data` from evaluation_contract "
                "and obtain the harness-scheduled rows and deterministic folds with: "
                "`X_eval, y_eval, row_ids, fold_ids, meta = "
                f"prepare_evaluation_data(train_data, '{fidelity}')`. Train/evaluate "
                "on X_eval/y_eval using fold_ids; do not create another random split.\n"
                "4. Encode categorical columns with fold-fitted preprocessing so "
                "sklearn models can fit and unseen values are handled safely.\n"
                "5. Evaluate with the supplied deterministic folds and train final "
                "test-prediction models on all X_eval rows. Save OOF predictions to "
                "oof_predictions.csv with columns row_id,target,prediction."
            )
            result_metric = metric_name
        
        algo_system = (
            "You are an expert ML engineering agent. Write a Python script 'initial_algorithm.py' that trains "
            "a baseline model (e.g. Scikit-Learn DecisionTree, RandomForest, or LogisticRegression) on the data "
            "provided by the MyDataLoader class from initial_dataloader.py.\n"
            "The script MUST:\n"
            "1. Import all required libraries: 'import numpy as np', 'import pandas as pd', 'import os', etc.\n"
            "2. Import MyDataLoader: 'from initial_dataloader import MyDataLoader'.\n"
            f"{evaluation_instructions}\n"
            "6. Emit and flush a concise progress line before and after every fold "
            "and training stage, then print the score in a format like "
            "'Validation Score: <float>'.\n"
            f"7. At the END of the script, write a JSON file 'result.json' in the current directory with EXACTLY this structure:\n"
            f'   {{"score": <float>, "metric": "{result_metric}", "direction": "{metric_direction}", "fidelity": "{fidelity}", "folds": {expected_folds}}}\n'
            "   Example: import json; json.dump({\"score\": 0.8521, \"metric\": \"roc_auc\", \"direction\": \"maximize\"}, open('result.json', 'w'))\n"
            "8. Save test predictions to './submission/submission.csv'. If './input/sample_submission.csv' exists, "
            "read it and preserve its exact column names, identifier values, row order, and prediction-column order. "
            "Do not assume the columns are named 'id' or 'target'. If no sample exists, derive the schema from the "
            "task description and test_ids.\n"
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
