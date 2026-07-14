import os
import sys
from pathlib import Path
from .llm_utils import call_llm


class InitialAgent:
    def __init__(self, model_name: str = None):
        self.model_name = model_name

    def generate_initial_code(self, task_dir: Path, temperature: float = 0.2):
        """
        Reads task_description.md and dynamically writes the initial_dataloader.py
        and initial_algorithm.py files.
        
        Args:
            task_dir: Path to the task directory containing task_description.md
            temperature: LLM sampling temperature (use 0.0 for reproducible baseline)
        """
        desc_file = task_dir / "task_description.md"
        if not desc_file.exists():
            print(f"Warning: task_description.md not found at {desc_file}. Using generic description.")
            description = f"Predict the target variable for the {task_dir.name} tabular dataset. Please infer the exact details from the train.csv and test.csv files in the input directory."
        else:
            with open(desc_file, 'r', encoding='utf-8') as f:
                description = f.read()

        # Run dataset analysis and profiling
        report_file = task_dir / "dataset_analysis_report.txt"
        try:
            from .data_analyzer import run_dataset_analysis
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
            import json
            with open(config_file, 'r', encoding='utf-8') as f:
                task_config = json.load(f)
            metric_name = task_config.get("metric_name", "score")
            metric_direction = task_config.get("metric_direction", "maximize")

        print(f"InitialAgent: Generating dataloader for task in {task_dir}...")
        
        dataloader_system = (
            "You are an expert ML engineering agent. Write a Python module 'initial_dataloader.py' containing "
            "a Class 'MyDataLoader' that reads raw data from './input/train.csv' and './input/test.csv' and pre-processes them.\n"
            "The class MUST define a method 'get_data()' which returns two dicts: train_data and test_data.\n"
            "Format of train_data: {'X': pd.DataFrame, 'y': np.ndarray, 'X_val': pd.DataFrame, 'y_val': np.ndarray, "
            "'has_val': bool, 'cat_cols': list_of_str, 'cont_cols': list_of_str, 'cat_dims': list_of_int, 'n_cont': int}.\n"
            "Format of test_data: {'X_test': pd.DataFrame, 'test_ids': np.ndarray/pd.Series}.\n"
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
        
        # Clean markdown code block
        clean_loader = dataloader_code
        if "```python" in dataloader_code:
            clean_loader = dataloader_code.split("```python")[1].split("```")[0]
        elif "```" in dataloader_code:
            clean_loader = dataloader_code.split("```")[1].split("```")[0]
            
        loader_path = task_dir / "initial_dataloader.py"
        with open(loader_path, 'w', encoding='utf-8') as f:
            f.write(clean_loader.strip())
        print(f"InitialAgent: Saved {loader_path}")

        # 2. Generate Algorithm Skeleton
        print(f"InitialAgent: Generating algorithm skeleton for task in {task_dir}...")
        
        algo_system = (
            "You are an expert ML engineering agent. Write a Python script 'initial_algorithm.py' that trains "
            "a baseline model (e.g. Scikit-Learn DecisionTree, RandomForest, or LogisticRegression) on the data "
            "provided by the MyDataLoader class from initial_dataloader.py.\n"
            "The script MUST:\n"
            "1. Import all required libraries: 'import numpy as np', 'import pandas as pd', 'import os', etc.\n"
            "2. Import MyDataLoader: 'from initial_dataloader import MyDataLoader'.\n"
            "3. Instantiate and load data EXACTLY like this:\n"
            "   loader = MyDataLoader()\n"
            "   train_data, test_data = loader.get_data()\n"
            "   X_train, y_train = train_data['X'], train_data['y']\n"
            "   X_val, y_val = train_data['X_val'], train_data['y_val']\n"
            "   X_test, test_ids = test_data['X_test'], test_data['test_ids']\n"
            "4. Encode any categorical columns present in train_data['cat_cols'] using sklearn.preprocessing.LabelEncoder "
            "so sklearn models can fit without ValueError. Be sure to convert features to string or handle unseen labels gracefully.\n"
            "5. Train the baseline model, evaluate it on the validation set (if available). Be careful of any rare classes in validation scoring.\n"
            "6. Print the score to stdout in a format like 'Validation Score: <float>'.\n"
            f"7. At the END of the script, write a JSON file 'result.json' in the current directory with EXACTLY this structure:\n"
            f'   {{"score": <float>, "metric": "{metric_name}", "direction": "{metric_direction}"}}\n'
            "   Example: import json; json.dump({{\"score\": 0.8521, \"metric\": \"roc_auc\", \"direction\": \"maximize\"}}, open('result.json', 'w'))\n"
            "8. Save test predictions to './submission/submission.csv' with columns ['id', 'target'] or matching the submission schema.\n"
            "Return ONLY valid Python code wrapped in a ```python code block."
        )
        
        algo_user = f"""
Task Description:
{description}

{dataset_snapshot}

Please write the complete 'initial_algorithm.py' file.
"""
        algo_code = call_llm(algo_system, algo_user, model=self.model_name, temperature=temperature)
        
        clean_algo = algo_code
        if "```python" in algo_code:
            clean_algo = algo_code.split("```python")[1].split("```")[0]
        elif "```" in algo_code:
            clean_algo = algo_code.split("```")[1].split("```")[0]
            
        algo_path = task_dir / "initial_algorithm.py"
        with open(algo_path, 'w', encoding='utf-8') as f:
            f.write(clean_algo.strip())
        print(f"InitialAgent: Saved {algo_path}")
        
        return True
