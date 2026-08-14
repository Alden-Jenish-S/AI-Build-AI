# Architecture

```text
eval/run_search.py
  -> ManagerAgent
     -> TaskAnalyzer: read-only directory inventory + goal/output discovery
     -> CouncilCoordinator
        -> bounded diagnostics + input allowlist + de-identified fingerprint
        -> adaptive member mandates -> focused diagnostic scripts
        -> precise primary-literature retrieval + query/provenance audit
        -> independent reports -> adversarial review + chair synthesis in one step
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
search_evidence.py            prediction signatures, noise estimates, family fingerprints
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
- A node is completed only after its native output passes validation or, for a
  supported numeric table task, its aligned OOF/test prediction bundle passes
  central validation and score recomputation. CSV/TSV, JSON, JSON Lines, NumPy
  arrays, images, ZIP files, and common text/bioinformatics formats receive safe
  semantic checks. Unknown formats receive structural checks and can add
  validators through the suffix registry.
- Only completed root, recovery, refinement, diversity, architecture, and merge
  implementations consume the configured budget. Tuning is budget-free and is
  bounded independently by tune depth and tuning/implementation attempt caps.
- Initial coverage follows the council's Pareto portfolio and remains bounded by
  the configured budget. The root portfolio is screened first by cheap probes
  (bounded subsets/epochs, same protocol, budget-free): the top-scoring plans
  are promoted to full data runs (successive halving), and a probe failure
  falls back to direct full roots instead of blocking the search.
- A failed root promotes a pre-generated backup without
  spending completed-experiment budget.
- Every improvement decision is evaluation-noise aware. Noise is estimated from
  `fold_scores` dispersion, then repeated `seed_scores` dispersion, then a
  relative floor (`1.5e-3` of the score) when neither exists; a candidate must
  exceed its parent by `improvement_noise_k` noise units up front. The
  high-performer band and plateau detection use the same noise-scaled margins.
- Completed implementations may store a fixed-size `prediction_signature` (and
  fold/seed scores) in `result.json`. Pending frontier actions whose measured
  parent is decorrelated with the incumbent best receive a complementarity
  bonus (`complementarity_weight x (1 - |pearson|)`), and merge pairs are
  chosen by score times the same complementarity term. Signature validation is
  strict; malformed or absent signatures simply disable the term.
- Model-family diversity is enforced as a hard fingerprint constraint: each
  plan's family fingerprint is stored in its node config, executed families
  collide structurally, and a `diversify` proposal repeating a measured family
  triggers exactly one re-plan before the action is discarded. A
  `diversify` action that passes the guard must also beat its base within
  evaluation noise in a cheap probe before the full run starts.
- Long iterative implementations receive an abort contract: after a warmup,
  checkpointed mid-run scores are compared with the incumbent best (margin
  `abort_margin_std` noise units, patience 2 checks). A trailing run may report
  `status: "truncated"` with an honest score; truncated nodes refund their idea
  budget and generate no follow-ups.
- Tuning proposals include an explicit search-space spec and budget (bounded
  config count, identical protocol, per-config early stopping), so tuning is a
  searchable dimension rather than a single configuration.
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
- The architect loop is iterative: when a completed custom-network run improved
  its control within noise and a later plateau retriggers, the manager queues
  another `architect` action (up to `max_architect_iterations`), passing the
  residual evidence from the prior custom-network result versus its control to
  the planner.
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
- Before finalization, a single budget-free numeric ensemble step may run when
  two or more unique nodes stored centrally scorable, aligned prediction bundles.
  The manager fits regularized non-negative weights with outer cross-fitting and
  accepts a non-trivial blend within an uncertainty-scaled robustness tolerance.
  Pruning controls expansion, not ensemble eligibility. Incompatible, task-native,
  generation, control, and structured-output tasks keep their declared evaluator
  and validated native-output fallback.
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
