# Project Components: `agent_system`

This document provides a high-level overview of the directories and files within the `agent_system` project, detailing the responsibilities of each component and the verification mechanisms in place.

---

## 📂 `agents/`
Orchestrating agents that collaborate to analyze tasks, design strategies, write code, run experiments, and aggregate results.

* **[manager_agent.py](file:///Users/aldenjenish/Desktop/Openclaw_workspace/AIBuildAI_UCSD/agent_system/agents/manager_agent.py)**: The central orchestrator. Manages the execution lifecycle of the method tree, handles budgeting, coordinates downstream agents, and calls the scheduler.
* **[task_analyzer.py](file:///Users/aldenjenish/Desktop/Openclaw_workspace/AIBuildAI_UCSD/agent_system/agents/task_analyzer.py)**: Inspects the raw datasets and target definitions to build a canonical `TaskSpec` and a dataset analysis profile.
* **[modality_scaffold.py](file:///Users/aldenjenish/Desktop/Openclaw_workspace/AIBuildAI_UCSD/agent_system/agents/modality_scaffold.py)**: Generates a deterministic, task-specific `TaskDataLoader` class customized to load the input modality files and targets correctly.
* **[technique_agent.py](file:///Users/aldenjenish/Desktop/Openclaw_workspace/AIBuildAI_UCSD/agent_system/agents/technique_agent.py)**: Proposes ML techniques, model designs, and search strategies based on task specifications.
* **[implementation_agent.py](file:///Users/aldenjenish/Desktop/Openclaw_workspace/AIBuildAI_UCSD/agent_system/agents/implementation_agent.py)**: Writes executable Python scripts that implement the proposed techniques, including model definitions, training loops, and predictions.
* **[aggregator_agent.py](file:///Users/aldenjenish/Desktop/Openclaw_workspace/AIBuildAI_UCSD/agent_system/agents/aggregator_agent.py)**: Blends/ensembles multiple compatible models by aggregating Out-Of-Fold (OOF) predictions.
* **[validation_guard.py](file:///Users/aldenjenish/Desktop/Openclaw_workspace/AIBuildAI_UCSD/agent_system/agents/validation_guard.py)**: Enforces safety constraints and code checks, ensuring no target leakage occurs.
* **[setup_agent.py](file:///Users/aldenjenish/Desktop/Openclaw_workspace/AIBuildAI_UCSD/agent_system/agents/setup_agent.py)**: Manages environment preparation and validates allowlisted dependencies.
* **[web_search.py](file:///Users/aldenjenish/Desktop/Openclaw_workspace/AIBuildAI_UCSD/agent_system/agents/web_search.py)**: Provides agent capabilities to search the web for external libraries, documentations, and best practices.
* **[llm_utils.py](file:///Users/aldenjenish/Desktop/Openclaw_workspace/AIBuildAI_UCSD/agent_system/agents/llm_utils.py)**: Contains utility functions for LLM API prompts, token counts, and structured interactions.

---

## 📂 `core/`
The foundational data contracts and registries defining how inputs, models, and modalities interact.

* **[contracts.py](file:///Users/aldenjenish/Desktop/Openclaw_workspace/AIBuildAI_UCSD/agent_system/core/contracts.py)**: Schema specifications for `TaskSpec`, targets, predictions, and metrics.
* **[runtime_contracts.py](file:///Users/aldenjenish/Desktop/Openclaw_workspace/AIBuildAI_UCSD/agent_system/core/runtime_contracts.py)**: Defines runtime entities such as datasets, split lists, predictions, and model evaluation bundles.
* **[modality_registry.py](file:///Users/aldenjenish/Desktop/Openclaw_workspace/AIBuildAI_UCSD/agent_system/core/modality_registry.py)**: A registry mapping modalities to their respective data-adapter implementations.

---

## 📂 `modalities/`
Modality-specific data-handling adapters. They inspect, profile, and index raw dataset files into structures that the models can ingest.

* **[tabular.py](file:///Users/aldenjenish/Desktop/Openclaw_workspace/AIBuildAI_UCSD/agent_system/modalities/tabular.py)**: Processes and profiles classic tabular datasets (CSVs, Parquet files).
* **[text.py](file:///Users/aldenjenish/Desktop/Openclaw_workspace/AIBuildAI_UCSD/agent_system/modalities/text.py)**: Manages natural language text corpora and text inputs.
* **[image.py](file:///Users/aldenjenish/Desktop/Openclaw_workspace/AIBuildAI_UCSD/agent_system/modalities/image.py)**: Profiles and loads computer vision image datasets.
* **[audio.py](file:///Users/aldenjenish/Desktop/Openclaw_workspace/AIBuildAI_UCSD/agent_system/modalities/audio.py)**: Loads, sample-rates, and prepares audio files.
* **[video.py](file:///Users/aldenjenish/Desktop/Openclaw_workspace/AIBuildAI_UCSD/agent_system/modalities/video.py)**: Configures video ingestion, frame sampling rates, and clips.
* **[multimodal.py](file:///Users/aldenjenish/Desktop/Openclaw_workspace/AIBuildAI_UCSD/agent_system/modalities/multimodal.py)**: Combines multiple modalities (e.g., text + image metadata) into a unified dataset profile.
* **[base.py](file:///Users/aldenjenish/Desktop/Openclaw_workspace/AIBuildAI_UCSD/agent_system/modalities/base.py)**, **[media_base.py](file:///Users/aldenjenish/Desktop/Openclaw_workspace/AIBuildAI_UCSD/agent_system/modalities/media_base.py)**, & **[common.py](file:///Users/aldenjenish/Desktop/Openclaw_workspace/AIBuildAI_UCSD/agent_system/common.py)**: House common abstract classes, helpers, and file formats used by all modality adapters.

---

## 📂 `evaluation/`
The validation lifecycle, cross-validation splitters, and evaluation metrics registry.

* **[metrics.py](file:///Users/aldenjenish/Desktop/Openclaw_workspace/AIBuildAI_UCSD/agent_system/evaluation/metrics.py)**: Registry of task metrics (Accuracy, F1, RMSE, Box IoU, etc.) and calculations.
* **[splitters.py](file:///Users/aldenjenish/Desktop/Openclaw_workspace/AIBuildAI_UCSD/agent_system/evaluation/splitters.py)**: Provides deterministic split strategies (K-Fold, Stratified, Grouped) preventing data leakage.
* **[fidelity.py](file:///Users/aldenjenish/Desktop/Openclaw_workspace/AIBuildAI_UCSD/agent_system/evaluation/fidelity.py)**: Standardizes computational fidelity levels (`screen`, `medium`, `full`) specifying caps on training epochs, sample ratios, etc.
* **[runner.py](file:///Users/aldenjenish/Desktop/Openclaw_workspace/AIBuildAI_UCSD/agent_system/evaluation/runner.py)**: Executes the generated algorithm scripts in sandbox environments.
* **[prediction_io.py](file:///Users/aldenjenish/Desktop/Openclaw_workspace/AIBuildAI_UCSD/agent_system/evaluation/prediction_io.py)**: Handles serialization and formatting of model predictions (OOF and test).
* **[submission.py](file:///Users/aldenjenish/Desktop/Openclaw_workspace/AIBuildAI_UCSD/agent_system/evaluation/submission.py)**: Validates, aligns, and formats final prediction submissions.
* **[policy.py](file:///Users/aldenjenish/Desktop/Openclaw_workspace/AIBuildAI_UCSD/agent_system/evaluation/policy.py)**: Specifies runtime limits, timeouts, and resource safety policies.

---

## 📂 `search/`
Search, optimization, and statistical validation policies.

* **[policies.py](file:///Users/aldenjenish/Desktop/Openclaw_workspace/AIBuildAI_UCSD/agent_system/search/policies.py)**: Decides whether to prune, promote, or diversify nodes based on current search evidence.
* **[evidence.py](file:///Users/aldenjenish/Desktop/Openclaw_workspace/AIBuildAI_UCSD/agent_system/search/evidence.py)**: Statistically compares candidate node scores against baseline/parents to gauge significance.
* **[tuning.py](file:///Users/aldenjenish/Desktop/Openclaw_workspace/AIBuildAI_UCSD/agent_system/search/tuning.py)**: Orchestrates parameter selection and tuning tasks.
* **[provenance.py](file:///Users/aldenjenish/Desktop/Openclaw_workspace/AIBuildAI_UCSD/agent_system/search/provenance.py)**: Tracks code lineages and dependencies across tree expansions.

---

## 📂 `tree/`
Node scheduling and search tree management.

* **[scheduler.py](file:///Users/aldenjenish/Desktop/Openclaw_workspace/AIBuildAI_UCSD/agent_system/tree/scheduler.py)**: Decides the next best action/node to execute using tree search heuristics (like Upper Confidence Bound - UCB).
* **[node.py](file:///Users/aldenjenish/Desktop/Openclaw_workspace/AIBuildAI_UCSD/agent_system/tree/node.py)**: Data structure representing a single method/implementation node.
* **[global_memory.py](file:///Users/aldenjenish/Desktop/Openclaw_workspace/AIBuildAI_UCSD/agent_system/tree/global_memory.py)**: Keeps track of successful/failed techniques to share insights globally.

---

## 📂 `ensemble/`
Predictive combination and stacking modules.

* **[stacking.py](file:///Users/aldenjenish/Desktop/Openclaw_workspace/AIBuildAI_UCSD/agent_system/ensemble/stacking.py)**: Implements out-of-fold-backed stacking estimators.
* **[structured.py](file:///Users/aldenjenish/Desktop/Openclaw_workspace/AIBuildAI_UCSD/agent_system/ensemble/structured.py)**: Orchestrates voting/averaging for non-scalar predictions.
* **[registry.py](file:///Users/aldenjenish/Desktop/Openclaw_workspace/AIBuildAI_UCSD/agent_system/ensemble/registry.py)**: Holds registered combination algorithms.

---

## 📂 `eval/`
Command-line interfaces and harnesses to launch the agent system on tasks.

* **[run_search.py](file:///Users/aldenjenish/Desktop/Openclaw_workspace/AIBuildAI_UCSD/agent_system/eval/run_search.py)**: Main entry point CLI to run automated search tree pipelines on a given dataset/task.
* **[evolve_harness.py](file:///Users/aldenjenish/Desktop/Openclaw_workspace/AIBuildAI_UCSD/agent_system/eval/evolve_harness.py)**: Evaluates and refines algorithm strategies dynamically.

---

## 📄 Root Utilities & Configs
* **[evaluation_contract.py](file:///Users/aldenjenish/Desktop/Openclaw_workspace/AIBuildAI_UCSD/agent_system/evaluation_contract.py)**: Serves as the interface contract for executing generated model code.
* **[runtime_utils.py](file:///Users/aldenjenish/Desktop/Openclaw_workspace/AIBuildAI_UCSD/agent_system/runtime_utils.py)**: Provides cross-cutting concerns like logging, subprocess execution, sandbox configurations, and metrics monitoring.
* **[run_all_tasks.sh](file:///Users/aldenjenish/Desktop/Openclaw_workspace/AIBuildAI_UCSD/agent_system/run_all_tasks.sh)**: A simple convenience bash script to run consecutive searches over various tasks.

---

## 🔍 Verification and Safety Mechanisms
The project incorporates multiple validation layers to ensure generated pipelines are accurate, deterministic, free of data leakage, and run safely.

### 1. Static Code Analysis (Leakage Guardrails)
* **Component**: [validation_guard.py](file:///Users/aldenjenish/Desktop/Openclaw_workspace/AIBuildAI_UCSD/agent_system/agents/validation_guard.py)
* **What it verifies**: Analyzes the AST (Abstract Syntax Tree) of LLM-generated Python code before execution to detect common implementation and safety bugs.
* **How it verifies**:
  * **Target/Data Leakage**: Scans for fitting estimators (`fit`, `fit_transform`) or calculating metrics (e.g. `mean`, `median`, `value_counts`) using test variables (e.g. `x_test`, `test_df`).
  * **Split Integrity**: Prevents independent train-test splitting (e.g., calling `train_test_split`) when the task evaluation harness expects deterministic fold rows and IDs.
  * **Data Resampling**: Blocks manual resampling of validation/test data or harness-owned indices.
  * **Write Protections**: Disallows generated scripts from writing directly to input directories or modifying harness contract manifests (e.g., `final_training_manifest.json`).
  * **API Misuse**: Ensures helper methods like `metric_value` are invoked with correct argument signatures.

### 2. Submission & Format Validation
* **Component**: [evaluation/submission.py](file:///Users/aldenjenish/Desktop/Openclaw_workspace/AIBuildAI_UCSD/agent_system/evaluation/submission.py)
* **What it verifies**: Validates the schema, prediction types, and formatting of final submission files.
* **How it verifies**:
  * **Structural Integrity**: Confirms the submission file exists, contains the correct prediction shape, and matches the sample submission template's identifiers without duplicates or missing values.
  * **Prediction Boundaries**: Validates that output probabilities are numeric, finite, range strictly between `[0, 1]`, and sum to exactly `1.0` (for classification tasks).

### 3. Dependency & Environment Sanitization
* **Component**: [agents/setup_agent.py](file:///Users/aldenjenish/Desktop/Openclaw_workspace/AIBuildAI_UCSD/agent_system/agents/setup_agent.py) & [runtime_utils.py](file:///Users/aldenjenish/Desktop/Openclaw_workspace/AIBuildAI_UCSD/agent_system/runtime_utils.py)
* **What it verifies**: Ensures safe, deterministic dependency execution and environment security.
* **How it verifies**:
  * **Allowlist Constraint**: Restricts dependencies to an approved set of package versions.
  * **Sandbox Credentials Removal**: Scrubs sensitive environment variables and credentials before spinning up code execution tasks.
  * **Progress Lease**: Monitors execution logs to ensure processes make active progress, automatically killing stalled tasks.
  * **Path-Traversal Blockers**: Prevents tasks from targeting files outside the sandbox boundary.
