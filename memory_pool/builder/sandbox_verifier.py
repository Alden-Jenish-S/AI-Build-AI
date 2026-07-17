import json
import os
import sys
import subprocess
import tempfile
import time
import re
import shutil
from pathlib import Path

# This file is launched directly by L2Builder, so Python otherwise exposes only
# memory_pool/builder on sys.path and cannot import project-level helpers.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runtime_utils import (
    redact_local_paths,
    resolve_within,
    sanitized_subprocess_env,
    validate_storage_identifier,
)


def _limit_verification_process() -> None:
    """Apply conservative Unix resource limits to untrusted artifact checks."""
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_CPU, (20, 20))
        resource.setrlimit(resource.RLIMIT_FSIZE, (20 * 1024 * 1024,) * 2)
        resource.setrlimit(resource.RLIMIT_NOFILE, (256, 256))
        if hasattr(resource, "RLIMIT_AS"):
            resource.setrlimit(resource.RLIMIT_AS, (8 * 1024**3,) * 2)
    except (ImportError, OSError, ValueError):
        # Timeout and process-group isolation remain active on unsupported hosts.
        pass

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
    json_path = Path(json_path).resolve()
    print(f"Verifying artifact: {json_path.name}")
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            card = json.load(f)
    except Exception as e:
        print(f"Error reading JSON {json_path}: {e}")
        return False

    artifact_dir = json_path.parent
    try:
        artifact_id = validate_storage_identifier(card.get("artifact_id"), "artifact_id")
        category = validate_storage_identifier(card.get("category"), "category")
        if artifact_dir.name != category and artifact_dir.name != artifact_id:
            # Local, not-yet-committed artifacts live in a node directory, so only
            # enforce the category folder for global pool paths.
            if artifact_dir.parent.name == "l2_store":
                raise ValueError("Model-card category does not match its directory")
        expected_code_name = f"{artifact_id}.py"
        if card.get("code_path") != expected_code_name:
            raise ValueError("Model-card code_path must match '<artifact_id>.py'")
        code_file = resolve_within(artifact_dir, expected_code_name)
    except ValueError as exc:
        print(f"Error: Invalid model card: {exc}")
        return False
    if not code_file.exists():
        print(f"Error: Python code file not found at {code_file}")
        return False

    # Create temporary script to run the module's entrypoint
    interface = card.get("interface")
    entrypoint_sig = interface.get("entrypoint") if isinstance(interface, dict) else None
    if not isinstance(entrypoint_sig, str) or not entrypoint_sig.strip():
        print("Error: Model card must define interface.entrypoint")
        return False
    entrypoint_name = entrypoint_sig.split('(')[0].strip()
    if not re.fullmatch(r"[A-Za-z_]\w*", entrypoint_name):
        print(f"Error: Invalid entrypoint function name: {entrypoint_name!r}")
        return False
    
    # Extract parameter names from the entrypoint signature for generic calling
    params_str = entrypoint_sig.split('(', 1)[1].rsplit(')', 1)[0] if '(' in entrypoint_sig else ""
    dependencies = [str(item).strip().lower() for item in card.get("dependencies", [])]
    requires_neural_runtime = any(
        dependency.startswith(
            ("torch", "pytorch", "tensorflow", "keras", "jax", "flax")
        )
        for dependency in dependencies
    )
    capabilities = card.get("capabilities", {})
    input_types = (
        capabilities.get("input_types", [])
        if isinstance(capabilities, dict)
        else []
    )
    target_types = (
        capabilities.get("target_types", [])
        if isinstance(capabilities, dict)
        else []
    )
    use_string_binary_target = (
        requires_neural_runtime and "binary_classification" in target_types
    )
    use_mixed_contract = requires_neural_runtime and (
        not input_types
        or "categorical" in input_types
        or "missing" in input_types
    )
    output_contract = interface.get("output_contract", {})
    output_kind = (
        output_contract.get("kind")
        if isinstance(output_contract, dict)
        else None
    )
    output_value_type = (
        output_contract.get("value_type")
        if isinstance(output_contract, dict)
        else None
    )

    # Generate the verification program in a dedicated writable directory.
    verification_dir = Path(tempfile.mkdtemp(prefix="aibuildai_verify_"))
    with tempfile.NamedTemporaryFile(
        'w', suffix='.py', delete=False, dir=verification_dir
    ) as tmp:
        tmp_name = tmp.name
        tmp.write(f"""
import sys
import os
import inspect
import numpy as np
import pandas as pd

# Verification must be fast and deterministic even when the host advertises a GPU.
os.environ["AIBUILDAI_VERIFY"] = "1"
os.environ["AIBUILDAI_ACCELERATOR"] = "cpu"
os.environ["AIBUILDAI_MAX_EPOCHS"] = "2"
os.environ["AIBUILDAI_EARLY_STOPPING_PATIENCE"] = "1"

# Add code file directory to path
sys.path.insert(0, r"{str(artifact_dir)}")
from {code_file.stem} import {entrypoint_name}

# Generate mock tabular data
np.random.seed(42)
n_train = 65
n_test = 30
X_train = pd.DataFrame({{
    'num1': np.random.randn(n_train),
    'num2': np.random.randn(n_train) * 10,
    'num3': np.random.rand(n_train),
    'num4': np.random.randn(n_train),
    'all_missing_num': np.full(n_train, np.nan),
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
    'all_missing_num': np.random.randn(n_test),
    'cat1': np.random.choice(['A', 'B', 'C'], size=n_test),
    'cat2': np.random.choice(['X', 'Y'], size=n_test)
}})
X_test.loc[:4, 'num3'] = np.nan
X_test.loc[:4, 'cat2'] = np.nan
X_test.loc[5:8, 'cat1'] = 'UNSEEN'
y_train = np.random.choice([0, 1], size=n_train)
preds_list = [np.random.rand(n_train), np.random.rand(n_train)]

# Use only numeric columns (safe for any ML function)
X_train_num = X_train[['num1', 'num2', 'num4']].copy()
X_test_num = X_test[['num1', 'num2', 'num4']].copy()
use_mixed_input = {use_mixed_contract!r}
X_train_input = X_train.copy() if use_mixed_input else X_train_num
X_test_input = X_test.copy() if use_mixed_input else X_test_num
y_train_input = np.where(y_train == 1, 'positive', 'negative') if {use_string_binary_target!r} else y_train
train_df = X_train_input.copy()
train_df['target'] = y_train_input

try:
    # Introspect the function signature to build arguments dynamically
    sig = inspect.signature({entrypoint_name})
    param_names = list(sig.parameters.keys())
    
    # Build a mapping of common parameter name patterns to mock data
    arg_map = {{
        'X_train': X_train_input,
        'X_test': X_test_input,
        'X_tr': X_train_input,
        'X_te': X_test_input,
        'X': X_train_input,
        'y_train': y_train_input,
        'y': y_train_input,
        'train_data': X_train_input,
        'test_data': X_test_input,
        'train_df': train_df,
        'test_df': X_test_input,
        'target': 'target',
        'target_col': 'target',
        'label_col': 'target',
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
            kwargs[pname] = X_train_input
        elif any(key in pname.lower() for key in ['x_test', 'test_x', 'x_te']):
            kwargs[pname] = X_test_input
        elif any(key in pname.lower() for key in ['y_train', 'target', 'label', 'y_tr']):
            kwargs[pname] = y_train_input
        elif 'pred' in pname.lower():
            kwargs[pname] = preds_list
        elif 'numeric' in pname.lower():
            kwargs[pname] = ['num1', 'num2', 'num3', 'num4', 'all_missing_num']
        elif 'cat' in pname.lower():
            kwargs[pname] = ['cat1', 'cat2'] if use_mixed_input else []
        elif 'col' in pname.lower() or 'feature' in pname.lower():
            kwargs[pname] = list(X_train_input.columns)
        elif 'fold' in pname.lower() or 'split' in pname.lower():
            kwargs[pname] = 3
        elif 'classif' in pname.lower():
            kwargs[pname] = True
        elif 'weight' in pname.lower():
            if p.default is inspect.Parameter.empty:
                kwargs[pname] = [0.5, 0.5]
            else:
                continue
        elif pname.lower() in ('epochs', 'num_epochs', 'n_epochs'):
            kwargs[pname] = 2
        elif 'patience' in pname.lower():
            kwargs[pname] = 1
        elif pname.lower() == 'batch_size':
            kwargs[pname] = 32
        elif pname.lower() == 'device':
            kwargs[pname] = 'cpu'
        elif p.default is not inspect.Parameter.empty:
            # Has a default value, skip it (will use default)
            continue
        else:
            # Unknown required param — try passing X_train as first, X_test as second
            if i == 0:
                kwargs[pname] = X_train_input
            elif i == 1:
                kwargs[pname] = X_test_input if pname != 'y' else y_train_input
            elif i == 2:
                kwargs[pname] = X_test_input
            else:
                kwargs[pname] = None
    
    # Call the entrypoint
    result = {entrypoint_name}(**kwargs)
    
    # Contract validation: at least one returned tabular/numeric object must
    # align with X_test, and all numeric outputs must be finite.
    if result is None:
        raise AssertionError("entrypoint returned None")
    print(f"Function returned type: {{type(result).__name__}}")

    returned_lengths = []
    prediction_observations = []
    def inspect_output(value):
        if isinstance(value, (pd.DataFrame, pd.Series, np.ndarray)):
            array = np.asarray(value)
            if array.ndim == 0:
                return
            returned_lengths.append(len(array))
            if np.issubdtype(array.dtype, np.number) and not np.isfinite(array).all():
                raise AssertionError("entrypoint returned NaN or infinite values")
            if len(array) == n_test and np.issubdtype(array.dtype, np.number):
                numeric = array.astype(float)
                row_variance = float(
                    np.max(np.var(numeric, axis=0))
                    if numeric.ndim > 1
                    else np.var(numeric)
                )
                prediction_observations.append(
                    (array.shape, row_variance, float(np.min(numeric)), float(np.max(numeric)))
                )
        elif isinstance(value, (tuple, list)):
            array = np.asarray(value)
            if (
                value
                and array.ndim > 0
                and np.issubdtype(array.dtype, np.number)
            ):
                returned_lengths.append(len(array))
                if not np.isfinite(array).all():
                    raise AssertionError("entrypoint returned NaN or infinite values")
                if len(array) == n_test:
                    numeric = array.astype(float)
                    row_variance = float(
                        np.max(np.var(numeric, axis=0))
                        if numeric.ndim > 1
                        else np.var(numeric)
                    )
                    prediction_observations.append(
                        (array.shape, row_variance, float(np.min(numeric)), float(np.max(numeric)))
                    )
            else:
                for item in value:
                    inspect_output(item)

    inspect_output(result)
    print(f"Returned tabular lengths: {{returned_lengths}}")
    if n_test not in returned_lengths:
        raise AssertionError(
            f"no returned prediction/transformation aligns with X_test length {{n_test}}; "
            f"observed lengths={{returned_lengths}}"
        )
    if {output_kind!r} == 'predictions':
        valid_predictions = [
            observation for observation in prediction_observations
            for shape in [observation[0]]
            if len(shape) == 1 or (len(shape) == 2 and shape[1] in (1, 2))
        ]
        if not valid_predictions:
            raise AssertionError(
                f"prediction output must have shape (n,), (n, 1), or (n, 2); "
                f"observed={{[item[0] for item in prediction_observations]}}"
            )
        if max(item[1] for item in valid_predictions) <= 1e-12:
            raise AssertionError("prediction output is constant on varied test rows")
        if {output_value_type!r} == 'probability' and not any(
            item[2] >= -1e-12 and item[3] <= 1.0 + 1e-12
            for item in valid_predictions
        ):
            raise AssertionError("probability predictions must stay within [0, 1]")
    
    print("SUCCESS")

except Exception as ex:
    import traceback
    print("FAILURE:")
    traceback.print_exc()
    sys.exit(1)
""")

    # Execute temp script in subprocess with timeout
    isolation_mode = "resource-limited-subprocess"
    try:
        command = [sys.executable, tmp_name]
        sandbox_exec = shutil.which("sandbox-exec")
        if sandbox_exec:
            writable = str(verification_dir).replace('"', '\\"')
            profile = (
                '(version 1)\n'
                '(deny default)\n'
                '(allow process*)\n'
                '(allow file-read*)\n'
                f'(allow file-write* (subpath "{writable}"))\n'
                '(allow sysctl-read)\n'
            )
            command = [sandbox_exec, "-p", profile, *command]
            isolation_mode = "sandbox-exec"
        child_env = sanitized_subprocess_env()
        child_env.update(
            {
                "HOME": str(verification_dir),
                "XDG_CACHE_HOME": str(verification_dir / "cache"),
                "PYTHONDONTWRITEBYTECODE": "1",
                "HTTP_PROXY": "http://127.0.0.1:9",
                "HTTPS_PROXY": "http://127.0.0.1:9",
                "ALL_PROXY": "http://127.0.0.1:9",
                "http_proxy": "http://127.0.0.1:9",
                "https_proxy": "http://127.0.0.1:9",
                "all_proxy": "http://127.0.0.1:9",
                "NO_PROXY": "",
                "no_proxy": "",
            }
        )
        res = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=25,
            cwd=verification_dir,
            env=child_env,
            preexec_fn=_limit_verification_process if os.name == "posix" else None,
        )
        if (
            sandbox_exec
            and res.returncode != 0
            and "sandbox_apply: Operation not permitted" in res.stderr
        ):
            # Some containerized/managed hosts expose sandbox-exec but prohibit
            # nested profiles. Retain resource limits and the isolated writable
            # directory rather than treating every artifact as incompatible.
            res = subprocess.run(
                [sys.executable, tmp_name],
                capture_output=True,
                text=True,
                timeout=25,
                cwd=verification_dir,
                env=child_env,
                preexec_fn=(
                    _limit_verification_process if os.name == "posix" else None
                ),
            )
            isolation_mode = "resource-limited-subprocess"
        success = (res.returncode == 0) and ("SUCCESS" in res.stdout)
        log_message = redact_local_paths(
            res.stdout + "\n" + res.stderr,
            artifact_dir,
            Path(tmp_name).parent,
        )
    except subprocess.TimeoutExpired:
        success = False
        log_message = "Execution timed out (25 seconds limit)"
    except Exception as e:
        success = False
        log_message = f"Process launch failed: {e}"
    finally:
        shutil.rmtree(verification_dir, ignore_errors=True)

    if success:
        print(f"Artifact {card['artifact_id']} passed sandbox verification!")
        print("WARNING: Verified against synthetic contract data only, not real task data.")
        card["verified"] = True
        card["verification_level"] = (
            "mixed-missing-contract-mock-data"
            if use_mixed_contract
            else "contract-mock-data"
        )
        card["verification_isolation"] = isolation_mode
        card["verification_log"] = f"Passed subprocess verification at {time.strftime('%Y-%m-%d %H:%M:%S')}.\nOutput:\n{log_message.strip()}"
        
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
        card["verification_log"] = f"Verification failed at {time.strftime('%Y-%m-%d %H:%M:%S')}.\nLog:\n{redact_local_paths(log_message, artifact_dir)}"
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
