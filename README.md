# AIBuildAI: Evidence-Driven Autonomous Machine Learning Engine

AIBuildAI is an autonomous, modality-neutral AutoML and research framework. It searches, evaluates, fine-tunes, and ensembles machine learning models across tabular, image, audio, video, and multimodal datasets.

The engine uses statistical search policies, candidate lineage tracking, a verified memory pool of reusable templates, and harness-owned evaluation safety to build, optimize, and bundle deployable ensemble solutions.

---

## Key Features

- **Modality-Neutral Architecture**: Native support for **tabular, image, audio, video, text**, and entity-aligned **multimodal** tasks (e.g., joint tabular and image classification/regression).
- **Evidence-Driven Search Scheduler**: A lineage-aware scheduling tree that manages search and promotion budgets dynamically based on uncertainty-adjusted cross-validation performance.
- **Harness-Owned Evaluation Protocol**: Enforces strict out-of-fold (OOF) cross-validation splits, leakage checks, and multi-fidelity runtime profiles (`screen`, `medium`, and `full`).
- **Dynamic Ensembling & Stacking**: Automatically generates OOF-weighted cross-validated stacking combiners or structured output merges (embeddings, spatial logit maps, bounding boxes) without code fusion.
- **Empirical Memory Pool**: Retrieves and ranks reusable templates and hyperparameter configurations from global tuning history based on modality, task profiles, and past performance.
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
│   ├── data_analyzer.py          # Backward-compatible tabular analyzer facade
│   ├── manager_agent.py          # Search orchestrator and merge builder
│   ├── technique_agent.py        # Retrieves techniques and designs approaches
│   ├── implementation_agent.py   # Generates and refines executable scripts
│   ├── aggregator_agent.py       # Blends predictions using OOF weights
│   ├── validation_guard.py       # General and modality-specific leakage guards
│   ├── prompt_context.py         # Modality-specific generation constraints
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
- `resolved_task_spec.json`, `dataset_profile.json`, `dataset_analysis.md`, and, for structured modalities, `dataset_index.jsonl`: Harness-generated task assets used by every root method. The profile owns index metadata and bounded diagnostics; the Markdown report supplies synthesized modeling directives without duplicating the full JSON contract.
- `node_<n>/`: Source code, execution logs, OOF predictions, and metrics for each search-tree node.
- `ensemble_manifest.json`: Evaluated ensemble configurations and weights.
- `submission.csv`: Final ensembled predictions ready for deployment.
- `results.md`: Markdown summary of the best method and execution metadata.

---

## Configuration

Tasks are described using a `task_config.json` file in the task's source directory (`tasks/<task_name>/task_config.json`):

```json
{
  "schema_version": 2,
  "modality": "multimodal",
  "component_modalities": ["image", "tabular"],
  "problem_type": "classification",
  "inputs": {
    "image": {
      "modality": "image",
      "source": "input/entities.csv",
      "format": "file_manifest",
      "path_field": "image_path"
    },
    "metadata": {
      "modality": "tabular",
      "source": "input/entities.csv",
      "format": "csv",
      "feature_fields": ["age", "location", "device_type"]
    }
  },
  "sample_id_field": "entity_id",
  "entity_id_field": "entity_id",
  "target": {
    "source": "input/entities.csv",
    "field": "label"
  },
  "output": {
    "type": "class_probabilities"
  },
  "metrics": [
    {"name": "roc_auc", "direction": "maximize"}
  ],
  "primary_metric": "roc_auc",
  "resource_limits": {
    "preferred_accelerator": "auto",
    "max_ram_gb": 32
  }
}
```

`problem_type`, `output`, and `metrics` are independent of modality. If a
legacy task omits its metric, the resolver chooses a concrete task-aware
default (`accuracy`, `rmse`, `dice`, `box_iou`, `ndcg@10`, and so on);
the legacy word `score` is not treated as an evaluator name. Mixed legacy
datasets whose tabular IDs match image/audio/video filenames are indexed as
multimodal automatically. Ambiguous media layouts should use the explicit
schema-v2 form above.

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
