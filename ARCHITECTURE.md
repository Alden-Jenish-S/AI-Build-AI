# Architecture

```text
eval/run_search.py
  -> ManagerAgent
     -> TaskAnalyzer: read-only directory inventory + goal/output discovery
     -> TechniqueAgent: compact initial plans, tuning plans, merge plans
     -> Planning Node -> Implementation Node: code -> execute -> error repair
     -> SubmissionValidator: infer sample/config contract -> validate artifact
     -> score comparison: promote best -> adaptive UCB frontier
        -> refine / tune / diversify -> prune or merge during search
     -> AggregatorAgent: copy the selected native output
```

The active source surface is intentionally small:

```text
agents/task_analyzer.py       observed files, folders, counts, goal, output
agents/technique_agent.py     root/refine/diversify/tuning/merge planning
agents/implementation_agent.py code generation, execution, error repair, web help
agents/submission_validator.py sample-aware, format-adaptive artifact validation
agents/manager_agent.py       baseline selection, tuning, pruning, mid-search merges
agents/aggregator_agent.py    final output copying
agents/llm_utils.py           provider selection, retries, caching, token accounting
agents/web_search.py          bounded cached documentation search
runtime_utils.py              task-file exposure, credential scrubbing, process lease
tree/node.py                  in-memory branch state
tree/scheduler.py             lineage-aware UCB selection of pending actions
eval/run_search.py            command-line entrypoint and Markdown result summary
```

## Search behavior

- A successful deliverable is accepted immediately; a later repair cannot
  overwrite it.
- Every root, tuning, merge, and recovery action is represented by a planning
  node followed by its implementation child.
- Every completed score is compared in the task's inferred direction.
- A node is completed only after its output passes validation. CSV/TSV, JSON,
  JSON Lines, NumPy arrays, images, ZIP files, and common text/bioinformatics
  formats receive safe semantic checks. Unknown formats receive structural
  checks and can add validators through the suffix registry.
- Only completed root, recovery, refinement, diversity, and merge
  implementations consume the configured budget. Tuning is budget-free and is
  bounded independently by tune depth and tuning/implementation attempt caps.
- Initial coverage is capped at two roots. A failed root promotes a pre-generated
  backup without spending completed-experiment budget.
- A displaced root receives one focused rescue tune. Any completed
  underperforming refinement, diversity, or merge receives one tuner attempt;
  if it still does not improve, that measured branch is pruned.
- Lazy refinement, tuning, and diversity actions are ranked by lineage reward,
  exploration value, and operator priority. Superseded ancestor actions are
  pruned before selection.
- Once two candidates are within the high-performance band, the manager may
  request a merge before remaining roots are finished.
- Remaining budget is spent on the current best implementation, not on weak
  lineages.

The manager snapshots `tree_state.json` after node transitions and refreshes it
at shutdown. The CLI always records `token_usage.json`, while finalization
renders the reference-style `method_tree.png` even when the search has no
successful node. These are observability outputs only and never validate or
reject an implementation.

## Failure behavior

Provider retries occur only in the shared LLM client, avoiding multiplied SDK
retries. Generated-program failures are repaired from captured process output;
library/API errors can be supplemented with web-search notes. Task inputs are
file-linked into each node (copied when links are unavailable), and credentials
are removed from child environments. A
renewable process lease watches output, deliverable changes, and CPU activity so
long active jobs are not killed by a fixed wall-clock timeout.
