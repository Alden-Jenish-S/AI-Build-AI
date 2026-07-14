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
        
        # Identify target column (difference between train and test headers)
        if df_test_sample is not None:
            target_cols = [c for c in df_train_sample.columns if c not in df_test_sample.columns]
        else:
            target_cols = []
            
        if not target_cols:
            # Fallback if we cannot infer from test.csv
            # Search for common target names or pick the last column
            common_targets = ["target", "label", "cover_type", "class"]
            found_common = [c for c in df_train_sample.columns if c.lower() in common_targets]
            if found_common:
                target_col = found_common[0]
            else:
                target_col = df_train_sample.columns[-1]
        else:
            target_col = target_cols[0]
            
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
            elif col.lower() in ["id", "uuid", "index"] or (df_train[col].nunique() == num_rows and np.issubdtype(df_train[col].dtype, np.integer)):
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
            
        class_counts = target_series.value_counts().sort_index()
        analysis_report.append(f"  Total instances in target column: {len(target_series)}")
        analysis_report.append("  Class Counts:")
        for val, cnt in class_counts.items():
            analysis_report.append(f"    - Class {val}: {cnt} instances ({cnt/len(target_series)*100:.4f}%)")
            
        # Check for rare classes (count < 10)
        rare_classes = {val: cnt for val, cnt in class_counts.items() if cnt < 10}
        if rare_classes:
            analysis_report.append("\n!!! CRITICAL INCONSISTENCY DETECTED !!!")
            analysis_report.append("  Rare target classes detected with very few instances:")
            for val, cnt in rare_classes.items():
                analysis_report.append(f"    - Class {val}: {cnt} instances")
            
            # Warn specifically about stratification crashes if count < 2
            very_rare = [val for val, cnt in rare_classes.items() if cnt < 2]
            if very_rare:
                analysis_report.append("  WARNING: Classes with only 1 instance will crash standard stratified splits (e.g. train_test_split(..., stratify=y)).")
                analysis_report.append("  RECOMMENDED ACTIONS IN DATALOADER:")
                analysis_report.append("    - Filter out (drop) rows belonging to these rare classes from the training set BEFORE performing train_test_split, OR")
                analysis_report.append("    - Do NOT use stratified split (disable `stratify=y` or set stratification to None), OR")
                analysis_report.append("    - Use a custom validation splitting strategy that handles rare classes gracefully.")
            else:
                analysis_report.append("  RECOMMENDED ACTION: Ensure minimum class count is satisfied before stratified splits, or disable stratification.")
                
    except Exception as e:
        analysis_report.append(f"Error during dataset analysis: {e}")
        
    analysis_report.append("=========================================")
    return "\n".join(analysis_report)
