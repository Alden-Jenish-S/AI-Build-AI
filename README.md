# Evidence-Driven AIBuildAI Agent

This repository is a tabular-ML research prototype inspired by AIBuildAI-2. It evolves multiple runnable solutions, evaluates them under an implementation-experiment budget, reuses empirically validated techniques, and produces a diversity-aware ensemble.

## What changed from the original prototype

- **True solution lineage:** follow-up experiments start from the measured parent `algorithm.py` and inherit reusable support files instead of restarting from the baseline.
- **Budget-aware branching:** initial fan-out is `min(3, max(1, budget // 3))`. Every success creates cheap virtual `refine`, `tune`, and `diversify` slots, but the LLM materializes only a slot selected by the scheduler.
- **Experiment-based budgets:** only attempted implementation runs consume `--budget`; technique planning does not.
- **Harness-owned evaluation:** deterministic row subsets and folds enforce `screen`, `medium`, and `full` fidelity. Full fidelity restores loader-held validation rows instead of silently training on only 80% of the data.
- **Validation safeguards:** generated code is checked for leakage, and the harness recomputes fold scores and uncertainty from aligned OOF predictions.
- **Empirical memory:** bounded pool retrieval ranks scope-compatible artifacts with lexical fit, prior reward, improvement rate, and a UCB-style exploration bonus.
- **Feasibility-aware execution:** declared accelerator, RAM, and runtime requirements are checked before an implementation can consume experiment budget.
- **GPU-preferred nodes:** CUDA is selected ahead of MPS and CPU when available, propagated into every node subprocess, and used by compatible pool artifacts with a safe CPU fallback.
- **Effect-aware execution:** descendants with byte-identical parent OOF and test predictions are marked `no_effect`, penalized, deduplicated on disk, and prevented from spawning more branches.
- **Diversity-aware aggregation:** candidates at the best completed fidelity are filtered by prediction correlation and combined with rank averaging. OOF files enable deterministic hill-climbed weights.
- **Review-gated harness evolution:** completed traces can generate offline prompt/control-flow proposals without automatically rewriting the live harness.

## Search workflow

```mermaid
flowchart TD
    T[Task + data] --> B[Generated deterministic baseline]
    B --> I[Budget-scaled architecture approaches]
    I --> M[Pool retrieval with empirical history]
    M --> E[Implementation experiment]
    E --> G{Leakage, fidelity, OOF and effect contracts valid?}
    G -- no --> D[Bounded debugging or failed reward]
    G -- yes --> R[CV score, uncertainty, OOF predictions, runtime]
    R --> U[Uncertainty-discounted reward backpropagation]
    U --> P[Virtual refine / tune / diversify slots]
    P --> L[Materialize selected slot only]
    L --> M
    R --> A[Diversity-aware same-fidelity ensemble]
    A --> S[Final aligned submission]
```

The budget unit is one attempted implementation experiment. A default budget of six therefore runs up to six ML pipelines; it is no longer split between technique and implementation nodes.

## Configuration

Each task may define `tasks/<task_name>/task_config.json`:

```json
{
  "metric_name": "roc_auc",
  "metric_direction": "maximize",
  "subprocess_timeout": 300,
  "enable_multi_fidelity": true,
  "ensemble_top_k": 3,
  "ensemble_strategy": "rank_average",
  "uncertainty_weight": 1.0,
  "max_l1_categories": 8,
  "max_artifact_candidates": 5,
  "resource_limits": {
    "preferred_accelerator": "auto",
    "max_ram_gb": 32
  }
}
```

`preferred_accelerator` accepts `auto`, `gpu`, `cuda`, `mps`, or `cpu`. `auto`/`gpu` choose
CUDA first, then Apple MPS, then CPU. An optional `accelerators` list restricts
the detected devices but never fabricates an unavailable device. Each node receives
the selection through `AIBUILDAI_ACCELERATOR` and records it in
`execution_resource.json`; successful node results also report the backend actually
used. CatBoost, XGBoost, LightGBM, and PyTorch pool artifacts
enable their native backend when compatible and retry on CPU if the installed package
lacks GPU support. `max_ram_gb` is a cap on detected memory, so the example is sized
for a 32 GB worker without overstating smaller machines.

Configure one LLM provider:

```bash
export NVIDIA_API_KEY="..."
# or
export GEMINI_API_KEY="..."

export LLM_PROVIDER="nvidia"   # required only when both keys are present
export LLM_MODEL="..."         # optional override
```

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python eval/run_ablation.py playground-series-s6e2 --budget 6
```

Rerunning the same condition moves the previous condition directory under
`runs/<task>/archive/` before writing new node artifacts. Web-derived technique
dependencies can be installed before sandbox import verification only when the
package is already listed in `requirements.txt`; the project-pinned requirement is
used instead of a model-card-selected version.

The generated loader contract supports both `MyDataLoader()()` and immediate
`get_data()`, and exposes complete training rows plus stable row identifiers. The local
`evaluation_contract.py` deterministically creates the required data subset and folds.
If the initial baseline still crashes, times out, emits an invalid score,
or omits its submission, the harness records `baseline_debug.log` and gives
`InitialAgent` up to two deterministic repair attempts before stopping.

The command creates:

- `runs/<task>/baseline/`: the deterministic comparison baseline.
- `runs/<task>/complete_system/node_<n>/`: code, technique records, results, optional OOF predictions, and submissions for every node.
- Per-node `execution_resource.json`: selected/available accelerators and fallback policy.
- Per-node `evaluation_manifest.json` and `fold_assignments.csv`: enforced row/fold protocol.
- `tree_state.json` and `method_tree.png`: durable search state and visualization.
- `search_trace.jsonl`: frontier scores, exploration constant, selections, skips, no-effect decisions, and completed rewards.
- `ensemble_manifest.json`: selected same-fidelity ensemble members.
- `submission.csv`: final schema-aligned predictions.
- `eval/results.md`: score, normalized improvement, experiment count, fidelity, token, pool, and overcome metrics.
- `runs/<task>/token_usage.json`: aggregate and per-call input/output token usage, prompt sizes, and latency.

## Optional offline harness evolution

After collecting completed runs across several tasks and seeds:

```bash
python eval/evolve_harness.py \
  runs/task_a/complete_system \
  runs/task_b/complete_system \
  --output eval/harness_candidates.json
```

This produces review-gated candidate changes. It intentionally does not edit prompts or control flow automatically; candidates should be accepted only after held-out multi-task evaluation.

## Tests

```bash
python -m unittest tests.test_core_safety
```

The artifact sandbox is a compatibility check, not an operating-system security boundary. `contract-mock-data` verification checks output alignment and finiteness but remains separate from real-task validation history.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the complete contracts and scheduling details.
