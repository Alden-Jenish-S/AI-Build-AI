import json

import pandas as pd
import numpy as np
from pathlib import Path

def run_dataset_analysis(task_dir: Path) -> str:
    """
    Profiles the datasets in the task directory (train.csv and test.csv).
    Identifies features, data types, missing values, columns to drop, 
    and target variable distribution (including rare classes).
    
    Returns:
        A formatted string report of the dataset characteristics.
    """
    train_csv = task_dir / "train.csv"
    if not train_csv.exists() and (task_dir / "input" / "train.csv").exists():
        train_csv = task_dir / "input" / "train.csv"
    
    test_csv = task_dir / "test.csv"
    if not test_csv.exists() and (task_dir / "input" / "test.csv").exists():
        test_csv = task_dir / "input" / "test.csv"

    if not train_csv.exists():
        return "Dataset Analysis: train.csv not found."

    print(f"DataAnalyzer: Inspecting {train_csv.name}...")
    analysis_report = []
    analysis_report.append("=== AUTOMATIC DATASET ANALYSIS REPORT ===")
    
    try:
        # Load sample to read headers and infer columns
        df_train_sample = pd.read_csv(train_csv, nrows=5)
        df_test_sample = pd.read_csv(test_csv, nrows=5) if (test_csv and test_csv.exists()) else None
        
        if df_train_sample.empty:
            raise ValueError("train.csv contains no data rows")

        # Prefer an explicit task setting, then infer from train/test and the
        # sample-submission schema. Never silently assume a fixed target name.
        configured_target = None
        config_file = task_dir / "task_config.json"
        if config_file.exists():
            with open(config_file, "r", encoding="utf-8") as f:
                configured_target = json.load(f).get("target_column")

        if df_test_sample is not None:
            target_cols = [c for c in df_train_sample.columns if c not in df_test_sample.columns]
        else:
            target_cols = []

        sample_file = task_dir / "sample_submission.csv"
        if not sample_file.exists() and (task_dir / "input" / "sample_submission.csv").exists():
            sample_file = task_dir / "input" / "sample_submission.csv"
        sample_prediction_cols = []
        if sample_file.exists():
            sample_cols = pd.read_csv(sample_file, nrows=0).columns.tolist()
            sample_prediction_cols = sample_cols[1:]
            analysis_report.append(f"Sample Submission Columns: {sample_cols}")

        if configured_target in df_train_sample.columns:
            target_col = configured_target
        elif len(target_cols) == 1:
            target_col = target_cols[0]
        elif len(set(target_cols) & set(sample_prediction_cols)) == 1:
            target_col = list(set(target_cols) & set(sample_prediction_cols))[0]
        else:
            # Fallback if we cannot infer from test.csv
            # Search for common target names or pick the last column
            common_targets = ["target", "label", "cover_type", "class"]
            found_common = [c for c in df_train_sample.columns if c.lower() in common_targets]
            if found_common:
                target_col = found_common[0]
            else:
                target_col = df_train_sample.columns[-1]
            if len(target_cols) > 1:
                analysis_report.append(
                    f"WARNING: Ambiguous train-only columns {target_cols}; inferred {target_col!r}. "
                    "Set target_column in task_config.json to make this explicit."
                )
            
        analysis_report.append(f"Inferred Target Column: '{target_col}'")
        
        # Determine file size
        file_size_bytes = train_csv.stat().st_size
        large_file = file_size_bytes > 300 * 1024 * 1024  # 300 MB
        
        # Read train data
        if large_file:
            # Read first 100,000 rows for general profiling to avoid OOM or high latency
            df_train = pd.read_csv(train_csv, nrows=100000)
            analysis_report.append(f"Note: train.csv is large ({file_size_bytes / (1024**2):.1f} MB). Statistics are computed on the first 100,000 rows.")
        else:
            df_train = pd.read_csv(train_csv)
            analysis_report.append(f"Loaded train.csv completely: {len(df_train)} rows.")
            
        num_rows = len(df_train)
        all_cols = df_train.columns.tolist()
        
        # 1. Features present
        analysis_report.append("\n1. Features and Column Types:")
        for col in all_cols:
            dtype = df_train[col].dtype
            null_count = df_train[col].isnull().sum()
            null_pct = (null_count / num_rows) * 100
            num_unique = df_train[col].nunique()
            analysis_report.append(f"  - '{col}': type={dtype}, unique_values={num_unique}, nulls={null_count} ({null_pct:.2f}%)")
            
        # 2. Suggested features to drop
        drop_suggestions = []
        for col in all_cols:
            if col == target_col:
                continue
            
            # Constant columns
            if df_train[col].nunique() == 1:
                drop_suggestions.append(f"  - '{col}' (constant column with single value)")
            # Sequential ID columns
            elif col.lower() in ["id", "uuid", "index"] or (
                df_train[col].nunique() == num_rows
                and pd.api.types.is_integer_dtype(df_train[col].dtype)
            ):
                drop_suggestions.append(f"  - '{col}' (looks like an ID or sequential index column)")
            # Extreme missing values
            elif df_train[col].isnull().mean() > 0.90:
                drop_suggestions.append(f"  - '{col}' (extremely high missing rate: {df_train[col].isnull().mean()*100:.1f}%)")
                
        analysis_report.append("\n2. Suggested Features to Drop:")
        if drop_suggestions:
            analysis_report.extend(drop_suggestions)
        else:
            analysis_report.append("  - None detected")
            
        # 3. Missing values summary
        missing_cols = [col for col in all_cols if df_train[col].isnull().any()]
        analysis_report.append("\n3. Missing Value Analysis:")
        if missing_cols:
            for col in missing_cols:
                null_count = df_train[col].isnull().sum()
                null_pct = (null_count / num_rows) * 100
                analysis_report.append(f"  - '{col}' has {null_count} missing values ({null_pct:.2f}%)")
        else:
            analysis_report.append("  - No missing values detected in the training set.")
            
        # 4. Target analysis and inconsistencies
        analysis_report.append(f"\n4. Target Distribution and Inconsistencies for '{target_col}':")
        # For target analysis, always get the full distribution (extremely fast if only loading target column)
        if large_file:
            target_series = pd.read_csv(train_csv, usecols=[target_col])[target_col]
        else:
            target_series = df_train[target_col]
            
        non_null_target = target_series.dropna()
        target_nunique = non_null_target.nunique()
        analysis_report.append(f"  Total instances in target column: {len(target_series)}")
        regression_threshold = max(20, int(np.sqrt(max(len(non_null_target), 1))))
        is_regression = (
            pd.api.types.is_numeric_dtype(non_null_target.dtype)
            and target_nunique > regression_threshold
        )
        if is_regression:
            analysis_report.append("  Inferred task type: regression (continuous numeric target)")
            analysis_report.append(
                "  Target summary: "
                + non_null_target.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).to_string()
            )
        else:
            class_counts = non_null_target.value_counts()
            analysis_report.append("  Inferred task type: classification")
            analysis_report.append("  Class Counts:")
            for val, cnt in class_counts.head(50).items():
                analysis_report.append(
                    f"    - Class {val}: {cnt} instances "
                    f"({cnt / max(len(non_null_target), 1) * 100:.4f}%)"
                )
            if len(class_counts) > 50:
                analysis_report.append(
                    f"    - ... {len(class_counts) - 50} additional classes omitted"
                )

            rare_classes = class_counts[class_counts < 10]
            if not rare_classes.empty:
                analysis_report.append("\n!!! CRITICAL INCONSISTENCY DETECTED !!!")
                analysis_report.append(
                    f"  {len(rare_classes)} target classes have fewer than 10 instances."
                )
                for val, cnt in rare_classes.head(20).items():
                    analysis_report.append(f"    - Class {val}: {cnt} instances")

                very_rare = rare_classes[rare_classes < 2]
                if not very_rare.empty:
                    analysis_report.append(
                        "  WARNING: Classes with only 1 instance will crash standard "
                        "stratified splits (e.g. train_test_split(..., stratify=y))."
                    )
                    analysis_report.append("  RECOMMENDED ACTIONS IN DATALOADER:")
                    analysis_report.append(
                        "    - Do NOT use stratification unless every class has enough samples, OR"
                    )
                    analysis_report.append(
                        "    - Use a custom validation strategy that handles rare classes gracefully."
                    )
                else:
                    analysis_report.append(
                        "  RECOMMENDED ACTION: Ensure minimum class count is satisfied "
                        "before stratified splits, or disable stratification."
                    )
                
    except Exception as e:
        analysis_report.append(f"Error during dataset analysis: {e}")
        
    analysis_report.append("=========================================")
    return "\n".join(analysis_report)
