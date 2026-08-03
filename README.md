# AIBuildAI: Evidence-Driven Autonomous Machine Learning Engine

AIBuildAI is an autonomous, data-agnostic AutoML and research framework. It inspects the files supplied with each task, verifies a task-local runtime contract, and only then searches, evaluates, fine-tunes, and ensembles machine-learning methods.

The engine uses statistical search policies, candidate lineage tracking, a verified memory pool of reusable templates, and harness-owned evaluation safety to build, optimize, and bundle deployable ensemble solutions.

---

## Key Features

- **Content-First Task Discovery**: Builds a neutral inventory from file signatures, bounded previews, directory collections, and cross-file identifier/stem alignment. No missing configuration silently defaults to a table or a conventional filename.
- **Pre-Planning Verification Gate**: The method tree is created only after required sources and targets exist, indexed records align, and the resolved contract accounts for substantial observed data groups.
- **Evidence-Driven Search Scheduler**: A lineage-aware scheduling tree that manages search and promotion budgets dynamically based on uncertainty-adjusted cross-validation performance.
- **Harness-Owned Evaluation Protocol**: Enforces strict out-of-fold (OOF) cross-validation splits, leakage checks, and multi-fidelity runtime profiles (`screen`, `medium`, and `full`).
- **Dynamic Ensembling & Stacking**: Automatically generates OOF-weighted cross-validated stacking combiners or structured output merges (embeddings, spatial logit maps, bounding boxes) without code fusion.
- **Empirical Memory Pool**: Retrieves reusable methods by branch intent and verified callable evidence; historical modality labels do not prefilter candidates.
- **Sandbox Verification Runtime**: Runs and checks newly discovered models and pipelines within isolated, resource-limited subprocesses (`sandbox-exec` on macOS).

---

## Project Structure

```text
agent_system/
├── core/
│   ├── contracts.py              # TaskSpec, InputSpec, and MetricSpec definitions
│   ├── runtime_contracts.py      # DatasetBundle, ResultRecord, and PredictionBundle
│   └── modality_registry.py      # Registry for modality-specific adapters
│
├── modalities/
│   ├── base.py                   # Protocols defining adapter interfaces
│   ├── generic.py                # Contract-driven indexing for unseen task formats
│   ├── media_base.py             # Shared manifest-driven media indexing logic
│   ├── tabular.py                # Tabular indexing, profiling, and discovery
│   ├── image.py                  # Image lazy loading, sizing, color modes
│   ├── audio.py                  # Audio indexing, duration, and resampling
│   ├── video.py                  # Video clip sampling and metadata profiling
│   ├── text.py                   # Document/caption text handling primitive
│   ├── multimodal.py             # Compositional entity-aligned multimodal adapters
│   └── common.py                 # File resolvers, pandas wrappers, and schema checkers
│
├── evaluation/
│   ├── runner.py                 # Core evaluation and protocol manifest lifecycle
│   ├── splitters.py              # Stratified, group, and entity-aware splitters
│   ├── fidelity.py               # Resource limits and fidelity configurations
│   ├── metrics.py                # Unified classification/regression metric definitions
│   └── prediction_io.py          # PredictionBundle validation and payload IO
│
├── ensemble/
│   ├── registry.py               # Output-type mapping to preferred/fallback strategies
│   ├── stacking.py               # Out-of-fold convex optimized stacker
│   └── structured.py             # Embedding, mask, and bounding box fusion
│
├── agents/
│   ├── task_analyzer.py          # Registry-driven task configuration analyzer
│   ├── task_inventory.py         # Neutral file inspection and contract verification
│   ├── data_analyzer.py          # Backward-compatible tabular analyzer facade
│   ├── manager_agent.py          # Search orchestrator and merge builder
│   ├── technique_agent.py        # Retrieves techniques and designs approaches
│   ├── implementation_agent.py   # Generates and refines executable scripts
│   ├── aggregator_agent.py       # Blends predictions using OOF weights
│   ├── validation_guard.py       # General and modality-specific leakage guards
│   ├── prompt_context.py         # Evidence-derived generation constraints
│   ├── modality_scaffold.py      # Scaffolds for media and multimodal data loading
│   └── llm_utils.py              # Model providers and token tracking
│
├── memory_pool/
│   ├── query_tool.py             # Retrieves compatible templates
│   └── builder/
│       ├── l2_builder.py         # Sandbox verifier wrapper
│       ├── sandbox_verifier.py   # Runs verifications under sandbox-exec
│       └── verification_runtime.py # Runs isolated verification tests
│
├── eval/
│   ├── run_search.py             # Direct method-tree search entrypoint
│   └── evolve_harness.py         # Proposes prompt/harness improvements
│
├── tests/                        # Full test suite covering all modules
├── evaluation_contract.py        # Compatibility facade for legacy models
└── runtime_utils.py              # Subprocess environments, limits, and helper tools
```

---

## Installation & Setup

1. **Clone and Navigate**:
   ```bash
   git clone <repository-url>
   cd agent_system
   ```

2. **Set up Virtual Environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## Execution

To launch an autonomous model search for a specific task under a target experiment budget:

```bash
python eval/run_search.py playground-series-s6e2 --budget 6
```

### Output Artifacts
Each search run generates a workspace directory structure under `runs/<task_name>/`:
- `task_inventory.json` and `task_verification.json`: content-first observations and the verification decision made before planning.
- `resolved_task_spec.json`, `dataset_profile.json`, `dataset_analysis.md`, and, when needed, `dataset_index.jsonl`: Harness-generated task assets used by every root method. The profile owns index metadata, the neutral inventory, verification evidence, and bounded diagnostics.
- `node_<n>/`: Source code, execution logs, OOF predictions, and metrics for each search-tree node.
- `ensemble_manifest.json`: Evaluated ensemble configurations and weights.
- `submission.csv`: Final ensembled predictions ready for deployment.
- `results.md`: Markdown summary of the best method and execution metadata.

---

## Configuration

The engine first inspects every supplied task file. An optional `task_config.json` can state facts that cannot be established from the files themselves; it is not required to use conventional basenames or a closed list of task/data identifiers. Ambiguous evidence is rejected before search rather than filled with defaults.

```json
{
  "schema_version": 2,
  "modality": "task-native-records",
  "problem_type": "next-state-utility",
  "inputs": {
    "observed": {
      "role": "train",
      "source": "input/observed.jsonl",
      "format": "jsonl",
      "id_field": "entity_key"
    },
    "requested": {
      "role": "test",
      "source": "input/requested.jsonl",
      "format": "jsonl",
      "id_field": "entity_key"
    }
  },
  "sample_id_field": "entity_key",
  "target": {
    "source": "input/observed.jsonl",
    "field": "utility"
  },
  "output": {
    "type": "next-state-value"
  },
  "metrics": [
    {"name": "absolute-utility-error", "direction": "minimize"}
  ],
  "primary_metric": "absolute-utility-error",
  "resource_limits": {
    "preferred_accelerator": "auto",
    "max_ram_gb": 32
  }
}
```

Task-kind, objective, output, and metric identifiers are open-ended. Registered
adapters remain optimized decoders for explicitly established storage formats;
unseen identifiers use the contract-driven generic indexer. With no complete
contract, deterministic content and relationship checks run first, then the
task-analysis agent may propose a contract. Every proposed source, target,
coverage claim, and indexed record is checked against disk. Ambiguity stops the
run before method-tree construction. Legacy configurations remain supported.

### Environment Settings
Define your preferred LLM provider and credentials:

```bash
# NVIDIA NIM
export LLM_PROVIDER="nvidia"
export NVIDIA_API_KEY="your-key-here"

# Google Gemini
export LLM_PROVIDER="gemini"
export GEMINI_API_KEY="your-key-here"

# OpenAI or compatible custom endpoint
export LLM_PROVIDER="openai"
export OPENAI_API_KEY="your-key-here"
export LLM_MODEL="your-model-name"
```

---

## Testing

To run the complete suite of unit and contract verification tests:

```bash
pytest
```
