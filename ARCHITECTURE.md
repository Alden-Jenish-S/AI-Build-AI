# Architecture

```text
eval/run_search.py
  -> ManagerAgent
     -> TaskAnalyzer: read-only directory inventory + goal/output discovery
     -> CouncilCoordinator
        -> bounded diagnostics + input allowlist + de-identified fingerprint
        -> adaptive member mandates -> focused diagnostic scripts
        -> precise primary-literature retrieval + query/provenance audit
        -> independent reports -> adversarial review -> chair synthesis
        -> ranked hypotheses + immutable evaluation protocol
     -> TechniqueAgent: council roots, tuning, architecture, and merge plans
     -> Planning Node -> Implementation Node: code -> execute -> error repair
     -> SubmissionValidator + evaluation-protocol check
     -> score comparison: promote best -> adaptive UCB frontier
        -> refine / tune / diversify / architect -> prune or merge during search
     -> AggregatorAgent: copy the selected native output
```

The active source surface is intentionally small:

```text
agents/task_analyzer.py       observed files, folders, counts, goal, output
agents/architecture_policy.py model-family coverage and material-plateau policy
agents/modality_policy.py     predictive-modality detection and ablation evidence
agents/council/contracts.py   stable brief and evaluation-protocol contracts
agents/council/diagnostics.py adaptive preflight and safe focused investigations
agents/council/research.py    de-identified query policy and source provenance
agents/council/coordinator.py member orchestration, peer review, synthesis
agents/technique_agent.py     root/refine/diversify/tuning/architect/merge planning
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
- Before any root is generated, the council produces an auditable brief. Its
  member count and mandates follow observed evidence gaps rather than a fixed
  roster or modality template.
- Search queries must be technically precise, de-identified, and targeted at a
  primary/authoritative source. Competition solutions, task identity, protected
  schema terms, notebooks, and copied code are rejected and audited.
- Likely held-out label/answer artifacts are excluded from every diagnostic and
  implementation node. The allowlist is enforced by physical file exposure, not
  only prompt instructions.
- All candidates use one hashed metric/split/seed/fold/leakage-unit protocol.
  Missing or mismatched evaluation evidence makes a node ineligible for ranking.
- Every root, tuning, architecture, merge, and recovery action is represented by a planning
  node followed by its implementation child.
- Every completed score is compared in the task's inferred direction.
- A node is completed only after its output passes validation. CSV/TSV, JSON,
  JSON Lines, NumPy arrays, images, ZIP files, and common text/bioinformatics
  formats receive safe semantic checks. Unknown formats receive structural
  checks and can add validators through the suffix registry.
- Only completed root, recovery, refinement, diversity, architecture, and merge
  implementations consume the configured budget. Tuning is budget-free and is
  bounded independently by tune depth and tuning/implementation attempt caps.
- Initial coverage follows the council's Pareto portfolio and remains bounded by
  the configured budget. A failed root promotes a pre-generated backup without
  spending completed-experiment budget.
- A displaced root receives one focused rescue tune. Any completed
  underperforming refinement, diversity, or merge receives one tuner attempt;
  if it still does not improve, that measured branch is pruned.
- Lazy refinement, tuning, and diversity actions are ranked by lineage reward,
  exploration value, and operator priority. Superseded ancestor actions are
  pruned before selection.
- Completed plans and code are classified into conventional, established-neural,
  custom-neural, or other coverage tracks. Plateau detection uses best-so-far
  material improvement with a relative tolerance; close but non-identical scores
  therefore count as saturation.
- With at least three idea slots, a missing custom-neural measurement is assigned
  one reserved `architect` experiment when conventional progress plateaus or the
  remaining budget reaches the reserve boundary. This intervention runs before
  another merge or finalization decision.
- An architecture experiment must derive its trainable graph from observed task
  evidence, implement it from primitive PyTorch operations, and compare the custom
  mechanism with both its measured parent and a plain neural ablation. It may test
  a task-invented composition but cannot assert global novelty without prior-art
  research and component ablations.
- Instruction files and sample-output templates are excluded from predictive
  modality detection. When two or more modalities remain, the council promotes
  a contribution audit comparing full fusion, per-modality controls, and every
  leave-one-modality-out variant on identical validation indices.
- A multimodal audit is rankable only when `result.json` contains finite
  `modality_ablation_scores` covering the full set, every single modality,
  and every leave-one-modality-out comparison. Every variant supplies finite fold
  scores and the same validation-index hash. Downstream decisions receive these
  scores and may intentionally retain only the strongest modality subset.
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
file-linked into each node from the council allowlist (copied when links are
unavailable), and credentials are removed from child environments. A
renewable process lease watches output, deliverable changes, and CPU activity so
long active jobs are not killed by a fixed wall-clock timeout.
