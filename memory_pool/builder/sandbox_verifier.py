import json
import os
import sys
import subprocess
import tempfile
import time
from pathlib import Path

def generate_mock_data():
    """Generates standard mock tabular datasets for testing."""
    import numpy as np
    import pandas as pd
    
    np.random.seed(42)
    n_train = 100
    n_test = 30
    
    # Features
    X_train = pd.DataFrame({
        'num1': np.random.randn(n_train),
        'num2': np.random.randn(n_train) * 10,
        'num3': np.random.rand(n_train),
        'num4': np.random.randn(n_train),
        'cat1': np.random.choice(['A', 'B', 'C'], size=n_train),
        'cat2': np.random.choice(['X', 'Y'], size=n_train)
    })
    # Add some NaNs to num3/cat2 to test imputation
    X_train.loc[np.random.choice(n_train, 10, replace=False), 'num3'] = np.nan
    X_train.loc[np.random.choice(n_train, 10, replace=False), 'cat2'] = np.nan
    
    X_test = pd.DataFrame({
        'num1': np.random.randn(n_test),
        'num2': np.random.randn(n_test) * 10,
        'num3': np.random.rand(n_test),
        'num4': np.random.randn(n_test),
        'cat1': np.random.choice(['A', 'B', 'C'], size=n_test),
        'cat2': np.random.choice(['X', 'Y'], size=n_test)
    })
    
    y_train = np.random.choice([0, 1], size=n_train)
    
    preds_list = [np.random.rand(n_train), np.random.rand(n_train)]
    
    return X_train, y_train, X_test, preds_list

def verify_artifact(json_path: Path) -> bool:
    """Verifies a single artifact's code in a sandbox subprocess.
    Updates the artifact JSON with verified status and logs.
    """
    print(f"Verifying artifact: {json_path.name}")
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            card = json.load(f)
    except Exception as e:
        print(f"Error reading JSON {json_path}: {e}")
        return False

    artifact_dir = json_path.parent
    code_file = artifact_dir / card["code_path"]
    if not code_file.exists():
        print(f"Error: Python code file not found at {code_file}")
        return False

    # Create temporary script to run the module's entrypoint
    entrypoint_sig = card["interface"]["entrypoint"]
    entrypoint_name = entrypoint_sig.split('(')[0].strip()
    
    # Extract parameter names from the entrypoint signature for generic calling
    params_str = entrypoint_sig.split('(', 1)[1].rsplit(')', 1)[0] if '(' in entrypoint_sig else ""

    # Generate isolated verification script with GENERIC calling logic
    with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False) as tmp:
        tmp_name = tmp.name
        tmp.write(f"""
import sys
import os
import inspect
import numpy as np
import pandas as pd

# Add code file directory to path
sys.path.insert(0, r"{str(artifact_dir)}")
from {code_file.stem} import {entrypoint_name}

# Generate mock tabular data
np.random.seed(42)
n_train = 100
n_test = 30
X_train = pd.DataFrame({{
    'num1': np.random.randn(n_train),
    'num2': np.random.randn(n_train) * 10,
    'num3': np.random.rand(n_train),
    'num4': np.random.randn(n_train),
    'cat1': np.random.choice(['A', 'B', 'C'], size=n_train),
    'cat2': np.random.choice(['X', 'Y'], size=n_train)
}})
X_train.loc[np.random.choice(n_train, 10, replace=False), 'num3'] = np.nan
X_train.loc[np.random.choice(n_train, 10, replace=False), 'cat2'] = np.nan

X_test = pd.DataFrame({{
    'num1': np.random.randn(n_test),
    'num2': np.random.randn(n_test) * 10,
    'num3': np.random.rand(n_test),
    'num4': np.random.randn(n_test),
    'cat1': np.random.choice(['A', 'B', 'C'], size=n_test),
    'cat2': np.random.choice(['X', 'Y'], size=n_test)
}})
y_train = np.random.choice([0, 1], size=n_train)
preds_list = [np.random.rand(n_train), np.random.rand(n_train)]

# Use only numeric columns (safe for any ML function)
X_train_num = X_train[['num1', 'num2', 'num4']].copy()
X_test_num = X_test[['num1', 'num2', 'num4']].copy()

try:
    # Introspect the function signature to build arguments dynamically
    sig = inspect.signature({entrypoint_name})
    param_names = list(sig.parameters.keys())
    
    # Build a mapping of common parameter name patterns to mock data
    arg_map = {{
        'X_train': X_train_num,
        'X_test': X_test_num,
        'X_tr': X_train_num,
        'X_te': X_test_num,
        'X': X_train_num,
        'y_train': y_train,
        'y': y_train,
        'train_data': X_train_num,
        'test_data': X_test_num,
        'preds_list': preds_list,
        'predictions': preds_list,
        'preds': preds_list,
    }}
    
    # Build kwargs by matching parameter names
    kwargs = {{}}
    positional_args = []
    for i, pname in enumerate(param_names):
        p = sig.parameters[pname]
        
        # Skip **kwargs and *args
        if p.kind in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL):
            continue
        
        # Check direct name match
        if pname in arg_map:
            kwargs[pname] = arg_map[pname]
        # Check partial name matches
        elif any(key in pname.lower() for key in ['x_train', 'train_x', 'x_tr']):
            kwargs[pname] = X_train_num
        elif any(key in pname.lower() for key in ['x_test', 'test_x', 'x_te']):
            kwargs[pname] = X_test_num
        elif any(key in pname.lower() for key in ['y_train', 'target', 'label', 'y_tr']):
            kwargs[pname] = y_train
        elif 'pred' in pname.lower():
            kwargs[pname] = preds_list
        elif 'col' in pname.lower() or 'feature' in pname.lower():
            kwargs[pname] = ['num1', 'num2', 'num4']
        elif 'cat' in pname.lower():
            kwargs[pname] = None
        elif 'fold' in pname.lower() or 'split' in pname.lower():
            kwargs[pname] = 3
        elif 'classif' in pname.lower():
            kwargs[pname] = True
        elif 'weight' in pname.lower():
            kwargs[pname] = [0.5, 0.5]
        elif p.default is not inspect.Parameter.empty:
            # Has a default value, skip it (will use default)
            continue
        else:
            # Unknown required param — try passing X_train as first, X_test as second
            if i == 0:
                kwargs[pname] = X_train_num
            elif i == 1:
                kwargs[pname] = X_test_num if pname != 'y' else y_train
            elif i == 2:
                kwargs[pname] = X_test_num
            else:
                kwargs[pname] = None
    
    # Call the entrypoint
    result = {entrypoint_name}(**kwargs)
    
    # Basic validation: it should return something without crashing
    if result is not None:
        print(f"Function returned type: {{type(result).__name__}}")
        if hasattr(result, '__len__'):
            print(f"Result length: {{len(result)}}")
        if isinstance(result, tuple):
            print(f"Tuple of {{len(result)}} elements")
    
    print("SUCCESS")

except Exception as ex:
    import traceback
    print("FAILURE:")
    traceback.print_exc()
    sys.exit(1)
""")

    # Execute temp script in subprocess with timeout
    try:
        res = subprocess.run([sys.executable, tmp_name], capture_output=True, text=True, timeout=25)
        success = (res.returncode == 0) and ("SUCCESS" in res.stdout)
        log_message = res.stdout + "\n" + res.stderr
    except subprocess.TimeoutExpired:
        success = False
        log_message = "Execution timed out (25 seconds limit)"
    except Exception as e:
        success = False
        log_message = f"Process launch failed: {e}"
    finally:
        try:
            os.unlink(tmp_name)
        except Exception:
            pass

    if success:
        print(f"Artifact {card['artifact_id']} passed sandbox verification!")
        print("WARNING: Verified against synthetic mock data only (shape-only-mock-data), not real task data.")
        card["verified"] = True
        card["verification_level"] = "shape-only-mock-data"
        card["verification_log"] = f"Passed sandbox verification at {time.strftime('%Y-%m-%d %H:%M:%S')}.\nOutput:\n{log_message.strip()}"
        
        # Write back to JSON
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(card, f, indent=2)
            return True
        except Exception as e:
            print(f"Failed to update JSON metadata: {e}")
            return False
    else:
        print(f"Artifact {card['artifact_id']} failed sandbox verification!")
        print(f"Log:\n{log_message}")
        card["verified"] = False
        card["verification_log"] = f"Verification failed at {time.strftime('%Y-%m-%d %H:%M:%S')}.\nLog:\n{log_message.strip()}"
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(card, f, indent=2)
        except Exception:
            pass
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Run on specific artifact JSON file
        ok = verify_artifact(Path(sys.argv[1]))
        sys.exit(0 if ok else 1)
    else:
        # Scan and verify all JSONs in l2_store/
        root_dir = Path(__file__).resolve().parent.parent / "l2_store"
        if not root_dir.exists():
            print(f"Store directory does not exist: {root_dir}")
            sys.exit(1)
            
        all_json_files = list(root_dir.glob("**/*.json"))
        print(f"Found {len(all_json_files)} artifacts to verify.")
        success_count = 0
        for jf in all_json_files:
            if verify_artifact(jf):
                success_count += 1
                
        print(f"\nVerification summary: {success_count}/{len(all_json_files)} verified successfully.")
        if success_count < len(all_json_files):
            sys.exit(1)
        else:
            sys.exit(0)
