# Ablation Study Results

Task evaluated: playground-series-s6e2
Derived baseline score: 0.9475 (direction: maximize)

| Condition | Best Score | Medal Rate | Gold Rate | Avg tokens/node | Pool-hit-rate | Overcome Rate |
|---|---|---|---|---|---|---|
| No-pool, single-agent (baseline) | 0.9539 | 100.0% | 0.0% | 5944 | n/a | 100.0% |
| No-pool, tree-search | 0.9539 | 25.0% | 0.0% | 112650 | n/a | 25.0% |
| Pool, single-agent | 0.9539 | 100.0% | 0.0% | 7598 | 1.0 | 100.0% |
| Pool, tree-search (full system) | 0.9475 | 0.0% | 0.0% | 108595 | 0.0 | 0.0% |
