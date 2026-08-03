# AIBuildAI Architecture

AIBuildAI starts every task with a method tree. It does not generate, run, or
score a preliminary model before search.

## Runtime flow

```text
eval/run_search.py
  -> ManagerAgent(task_name, budget)
  -> TaskAnalyzer resolves TaskSpec and builds a lazy DatasetBundle index
  -> deterministic task_dataloader.py is written to the run directory
  -> TechniqueAgent proposes distinct root methods
  -> scheduler selects technique and implementation nodes
  -> ImplementationAgent generates each root method directly from:
       - the canonical task contract
       - the dataset profile and lazy sample index
       - the selected technique plan/artifact
  -> descendants refine, diversify, tune, or promote measured parents
  -> AggregatorAgent evaluates compatible OOF-backed merges
  -> the strongest method at the highest completed fidelity is selected
  -> submission.csv, method_tree.png, results.md, and token_usage.json
```

The initial task-analysis step is model-free. It only creates contracts and
data references needed by root methods.

## Main modules

```text
agent_system/
├── agents/
│   ├── manager_agent.py          # Method-tree orchestration and scheduling
│   ├── task_analyzer.py          # TaskSpec resolution and dataset indexing
│   ├── modality_scaffold.py      # Deterministic TaskDataLoader source
│   ├── technique_agent.py        # Root/follow-up method planning and retrieval
│   ├── implementation_agent.py   # Root method generation and parent evolution
│   ├── aggregator_agent.py       # OOF-backed prediction merging
│   ├── setup_agent.py            # Allowlisted dependency preparation
│   └── validation_guard.py       # Leakage and execution-contract checks
├── core/
│   ├── contracts.py              # TaskSpec and typed task configuration
│   ├── runtime_contracts.py      # Dataset, split, prediction, and model bundles
│   └── modality_registry.py      # Adapter registration
├── modalities/
│   ├── tabular.py
│   ├── text.py
│   ├── image.py
│   ├── audio.py
│   ├── video.py
│   └── multimodal.py
├── evaluation/
│   ├── metrics.py                # Task-aware metric registry
│   ├── splitters.py              # Leakage-unit-aware deterministic folds
│   ├── fidelity.py               # Screen, medium, and full profiles
│   ├── prediction_io.py          # Typed prediction payloads
│   └── runner.py                 # Typed evaluation lifecycle
├── eval/
│   └── run_search.py             # Direct method-tree CLI
├── memory_pool/                  # Verified reusable technique artifacts
├── search/                       # Evidence, promotion, pruning, tuning policies
├── tree/                         # Node state and UCB scheduler
└── evaluation_contract.py        # Generated-script evaluation facade
```

## Task contract

Schema-v2 tasks declare their modality, problem, inputs, output, and metrics:

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
      "format": "csv"
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
    {
      "name": "accuracy",
      "direction": "maximize"
    }
  ],
  "primary_metric": "accuracy"
}
```

Legacy tabular and safely discoverable table-plus-media tasks are normalized
into the same `TaskSpec`. Ambiguous layouts require an explicit schema-v2
configuration.

When a concrete metric is absent, the task contract selects a problem-aware
default:

- classification: `accuracy`
- regression: `rmse`
- multilabel classification: `f1_macro`
- segmentation: `dice`
- object detection: `box_iou`
- retrieval: `ndcg@10`
- captioning: `token_f1`
- temporal localization: `temporal_iou`
- clustering: `silhouette_score`

Metric direction comes from the registered metric definition.

## Direct root-method generation

`ManagerAgent.run_tree_search()` prepares these model-free assets:

- `resolved_task_spec.json`
- `dataset_profile.json`
- `dataset_analysis.md`
- `dataset_index.jsonl`
- `task_dataloader.py`

`dataset_profile.json` contains the runtime dataset-index contract plus bounded,
modality-neutral diagnostics for target topology, missingness, train/test KS
drift, dimensionality, and collinearity. `dataset_analysis.md` turns those
signals into concise modeling directives for the planning and implementation
agents. Direct tabular sources do not materialize `dataset_index.jsonl`.

It then asks `TechniqueAgent` for distinct root approaches. Each selected root
approach becomes an implementation node. `ImplementationAgent` receives no
starter algorithm for root nodes; it builds the complete executable method
from the selected plan and canonical task assets.

Descendant nodes receive only the code and measured artifacts of their
explicit parent. This keeps parent evolution meaningful while preventing
unrelated methods from silently reusing another branch.

## Evaluation protocol

Generated code must:

1. Load `train_data` and `test_data` through
   `task_dataloader.TaskDataLoader`.
2. Call `prepare_evaluation_data(train_data, fidelity)`.
3. Use the harness-provided rows and fold assignments exactly.
4. Fit learned preprocessing inside each training fold.
5. Compute fold values with `evaluation.metrics.metric_value`.
6. Write complete OOF outputs and a non-empty submission.
7. Write `result.json` with the requested metric, direction, and fidelity.

Scalar OOF data uses:

```text
row_id,target,prediction
```

Class-probability OOF data uses:

```text
row_id,target,prediction::<class-1>,prediction::<class-2>,...
```

Structured prediction bundles use numeric NPZ payloads or JSON payloads for
ragged/string structures.

## Search rewards and evidence

Scheduler rewards are a monotonic bounded transformation of the task metric
after the fold-uncertainty penalty. Maximization metrics preserve score order;
minimization metrics reverse it.

Statistical decisions compare a candidate to:

1. its measured parent, when it has one; otherwise
2. the strongest previously completed method at the same fidelity.

The first successful root method needs no synthetic reference. Once comparable
evidence exists, policies may prune, promote, tune, diversify, or create a
manager-owned ensemble action.

## Multi-fidelity and resource safety

The registered profiles are `screen`, `medium`, and `full`. They jointly cap
sample fraction, folds, epochs, estimator iterations, tuning trials, spatial
resolution, audio duration/sample rate, and video frame sampling.

Long jobs use a renewable progress lease: total runtime is unbounded while the
process continues to produce observable activity. Credentials are removed from
generated-code subprocess environments, task inputs are exposed through
read-only file links, and optional dependencies must match the project
allowlist.

## Run artifacts

```text
runs/<task>/
├── resolved_task_spec.json
├── dataset_profile.json
├── dataset_analysis.md
├── dataset_index.jsonl
├── task_dataloader.py
├── tree_state.json
├── search_trace.jsonl
├── provenance_graph.json
├── method_tree.png
├── node_<n>/
│   ├── technique_plan.md or algorithm.py
│   ├── result.json
│   ├── oof_predictions.csv
│   ├── prediction_bundle.json
│   └── submission/submission.csv
├── submission.csv
├── results.md
└── token_usage.json
```

Starting a new run archives the previous run and begins with fresh task assets
and a fresh method tree.

## Running

```bash
python eval/run_search.py <task-name> --budget 6
```

Only attempted implementation experiments consume the budget. Planning,
feasibility checks, and skipped incompatible actions do not.
