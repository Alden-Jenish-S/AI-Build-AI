# AIBuildAI

AIBuildAI inspects an arbitrary task directory, asks an LLM to write direct
implementations for the observed data, executes and repairs them, and keeps the
best locally scored deliverable.

## Workflow

1. `TaskAnalyzer` recursively inventories the task's files and folders. For
   common table/JSON formats it records bounded samples, counts, columns, and
   keys. It also reads supplied task instructions and identifies the requested
   output/sample-submission shape.
2. `TechniqueAgent` proposes a small primary/backup root portfolio in one LLM
   call. Larger budgets execute at most two independent roots so most compute is
   retained for measured improvement.
3. Each root, tuning, merge, or recovery proposal is stored as a planning node;
   its following numbered child is the implementation node.
4. `ImplementationAgent` writes one self-contained `algorithm.py` per
   implementation node. Task files are exposed under `input/`; generated code
   uses those files directly and writes its deliverable under `submission/`.
5. Every deliverable passes a format-adaptive validator before its score is
   accepted. Supplied sample outputs define structure; otherwise arbitrary
   non-empty files or directory bundles are accepted, with safe semantic checks
   for recognized formats. Validation failures enter the repair loop.
6. A failed execution is repaired from its real code and stdout/stderr. Stale
   deliverables are removed between attempts, documentation search is available
   during repair, and transient provider responses are retried with backoff.
7. `ManagerAgent` uses a lineage-aware UCB frontier over lazy `refine`, `tune`,
   and `diversify` planning nodes. The strongest runnable node becomes the
   baseline. A displaced root receives one model-locked rescue tune and is
   pruned if it remains weak. Any later underperforming refinement, diversity,
   or merge candidate also receives one focused rescue tune before pruning.
   Improving descendants receive new actions; stale ancestors do not expand.
   Root, recovery, refinement, diversity, and merge implementations consume the
   configured new-idea budget. Tuning implementations are free and use separate
   depth/attempt safety limits.
8. Two competitive, independent lineages may be merged during the search. Both
   measured parent implementations are supplied to the merge, and an improving
   merge becomes the new baseline.
9. The selected node's native deliverable is validated again, then copied to
   `submission.csv` or `final_output/` at the run root.

Manager, planner, implementation-attempt, generated-program stdout/stderr,
tuning, pruning, merging, promotion, web-repair, and finalization progress are
printed live with node names and timestamps.

There are no task/runtime contract builders, modality adapters, artifact
manifests, repository evaluation harnesses, static generated-code gates, or
memory-pool verification jobs. The active tests cover submission validation,
but not the complete search workflow.
The generated implementation owns the task-appropriate local scoring method and
output construction.

## Run

```bash
python eval/run_search.py playground-series-s6e2 --budget 6
```

`--budget 6` funds six successfully scored new root/branch ideas. Tuning does
not decrement that budget; failed implementations also preserve it.

Configure any OpenAI-compatible provider through the environment, for example:

```bash
export LLM_PROVIDER=openai
export OPENAI_API_KEY=your-key
export LLM_MODEL=your-model
```

Web-assisted repair is enabled by default. Set `AIBUILDAI_WEB_SEARCH=0` to turn
it off. `LLM_MAX_RETRIES` and `LLM_TIMEOUT_SECONDS` control provider recovery.

## Output validation

Validation is evidence-driven rather than tied to a submission type:

- `sample_submission.*`, `sample_output.*`, output-template files, and sample
  output directories are discovered automatically;
- complete sample-submission tables enforce columns, row count, finite numeric
  values, and identifier membership without assuming identifier order;
- without a sample, recognized files receive safe semantic checks and unknown
  non-empty formats receive structural checks;
- arbitrary directory bundles are allowed, but empty directories, filesystem
  links, malformed recognized files, and artifacts outside the node are rejected;
- a specialized validator can be registered by file suffix without changing
  the manager or tree search.

Tasks that need stronger rules may add optional constraints to
`task_config.json`; no field is required:

```json
{
  "output_contract": {
    "reference": "example_output.jsonl",
    "kind": "file",
    "extension": ".jsonl",
    "primary_file": "predictions.jsonl",
    "row_count": 1000,
    "identifier_column": "record_id",
    "id_order_required": false
  }
}
```

## Run output

Each `runs/<task>/` contains only:

- `task_analysis.md`: observed task files/folders, data kinds/counts, goal,
  target, and expected output;
- `node1`, `node2`, … directly under the run root. Planning-node folders contain
  `node_state.json`; implementation-node folders additionally contain
  `algorithm.py`, `attempt_<n>.log`, the small `result.json`, inputs, and the
  native deliverable;
- `submission.csv` or `final_output/` from the selected node;
- `results.md`: score, pruning, runtime, and token summary;
- `tree_state.json`: compact final/progress tree state without source code,
  including new-idea budget and free-tuning counts;
- `token_usage.json`: aggregate and per-call LLM token metrics;
- `method_tree.png`: rendered node lineage, status, scores, pruning, merges, and
  the selected baseline. It uses the reference project's dynamically sized,
  leaf-span layout with descriptive planning and implementation boxes.

No dataset-profile or task-spec JSON is generated or placed in LLM context.
