# Results: house blueprint, 60 seeds per cell

Mean score (fraction of dependents rescued, 0 if the rescuer dies). Same seeds and same fires for every agent.
`oracle gap` = oracle - baseline: how much a correct rulebook is worth. With no novelty it must be +0.00.
`gap closed` = (adaptive - baseline) / (oracle - baseline); `n/a` when the oracle gap is under 0.05. Brackets: 95% paired-bootstrap intervals over seeds. Regenerate with `make results`;
ablations, detection delays and recovery curves: `python scripts/adaptive_study.py` (docs/adaptive_study.md).
`efficiency (o / a)` = fewest possible steps (ignoring fire and child pace) / steps taken, mean over each agent's own successes.
`steps vs oracle` = adaptive steps / oracle steps over the n seeds where BOTH got everyone out (1.00 = as fast as the oracle,
higher = slower). The oracle is the same planner given the true rules, not a proven shortest rescue.

| household | novelty | baseline | oracle | oracle gap | adaptive | adaptive - baseline | gap closed | all out (b / o / a) | efficiency (o / a) | steps vs oracle |
|---|---|---|---|---|---|---|---|---|---|---|
| single | none | 0.85 | 0.85 | +0.00 | 0.85 [0.75, 0.93] | +0.00 [+0.00, +0.00] | n/a | 85% / 85% / 85% | 0.89 / 0.89 | 1.00 [1.00, 1.00] (n=51) |
| single | door_jam | 0.57 | 0.82 | +0.25 | 0.77 [0.65, 0.87] | +0.20 [+0.08, +0.32] | 80% [53%, 100%] | 57% / 82% / 77% | 0.86 / 0.79 | 1.12 [1.03, 1.21] (n=46) |
| single | carpet_flashover | 0.93 | 0.93 | +0.00 | 0.93 [0.87, 0.98] | +0.00 [+0.00, +0.00] | n/a | 93% / 93% / 93% | 0.87 / 0.87 | 1.00 [1.00, 1.00] (n=56) |
| single | compound | 0.60 | 0.83 | +0.23 | 0.73 [0.62, 0.85] | +0.13 [+0.03, +0.23] | 57% [23%, 90%] | 60% / 83% / 73% | 0.84 / 0.80 | 1.07 [1.03, 1.12] (n=43) |
| single | hidden_dependent | 0.42 | 0.77 | +0.34 | 0.77 [0.70, 0.83] | +0.34 [+0.27, +0.42] | 100% [100%, 100%] | 0% / 55% / 55% | 0.81 / 0.81 | 1.00 [1.00, 1.00] (n=33) |
| married | none | 1.00 | 1.00 | +0.00 | 1.00 [1.00, 1.00] | +0.00 [+0.00, +0.00] | n/a | 100% / 100% / 100% | 0.18 / 0.18 | 1.00 [1.00, 1.00] (n=60) |
| married | door_jam | 1.00 | 1.00 | +0.00 | 1.00 [1.00, 1.00] | +0.00 [+0.00, +0.00] | n/a | 100% / 100% / 100% | 0.18 / 0.18 | 1.00 [1.00, 1.00] (n=60) |
| married | carpet_flashover | 1.00 | 1.00 | +0.00 | 1.00 [1.00, 1.00] | +0.00 [+0.00, +0.00] | n/a | 100% / 100% / 100% | 0.18 / 0.18 | 1.00 [1.00, 1.00] (n=60) |
| married | compound | 1.00 | 1.00 | +0.00 | 1.00 [1.00, 1.00] | +0.00 [+0.00, +0.00] | n/a | 100% / 100% / 100% | 0.18 / 0.18 | 1.00 [1.00, 1.00] (n=60) |
| married | hidden_dependent | 0.50 | 1.00 | +0.50 | 1.00 [1.00, 1.00] | +0.50 [+0.50, +0.50] | 100% [100%, 100%] | 0% / 100% / 100% | 0.04 / 0.03 | 1.21 [1.20, 1.21] (n=60) |
| kids | none | 0.74 | 0.74 | +0.00 | 0.74 [0.67, 0.81] | +0.00 [+0.00, +0.00] | n/a | 52% / 52% / 52% | 0.75 / 0.75 | 1.00 [1.00, 1.00] (n=31) |
| kids | door_jam | 0.47 | 0.68 | +0.20 | 0.65 [0.58, 0.72] | +0.18 [+0.10, +0.25] | 87% [65%, 127%] | 28% / 40% / 33% | 0.72 / 0.69 | 1.07 [1.00, 1.15] (n=17) |
| kids | carpet_flashover | 0.64 | 0.68 | +0.04 | 0.68 [0.60, 0.74] | +0.03 [-0.02, +0.08] | n/a | 35% / 38% / 40% | 0.64 / 0.64 | 1.00 [1.00, 1.01] (n=23) |
| kids | compound | 0.41 | 0.60 | +0.19 | 0.55 [0.47, 0.62] | +0.14 [+0.08, +0.21] | 74% [47%, 113%] | 20% / 30% / 22% | 0.71 / 0.70 | 0.95 [0.88, 1.02] (n=12) |
| kids | hidden_dependent | 0.49 | 0.57 | +0.08 | 0.58 [0.53, 0.62] | +0.08 [+0.02, +0.14] | 107% [50%, 182%] | 0% / 0% / 0% | n/a / n/a | n/a |
| kids_dog | none | 0.57 | 0.57 | +0.00 | 0.57 [0.51, 0.62] | +0.00 [+0.00, +0.00] | n/a | 10% / 10% / 10% | 0.77 / 0.77 | 1.00 [1.00, 1.00] (n=6) |
| kids_dog | door_jam | 0.35 | 0.51 | +0.16 | 0.49 [0.43, 0.55] | +0.14 [+0.08, +0.20] | 86% [58%, 126%] | 5% / 5% / 7% | 0.81 / 0.74 | 1.08 [1.00, 1.17] (n=2) |
| kids_dog | carpet_flashover | 0.44 | 0.47 | +0.03 | 0.47 [0.42, 0.53] | +0.03 [+0.00, +0.06] | n/a | 3% / 5% / 5% | 0.61 / 0.61 | 1.00 [1.00, 1.00] (n=3) |
| kids_dog | compound | 0.29 | 0.41 | +0.12 | 0.39 [0.33, 0.44] | +0.10 [+0.05, +0.16] | 86% [54%, 140%] | 3% / 3% / 3% | 0.74 / 0.72 | 1.05 [1.00, 1.08] (n=2) |
| kids_dog | hidden_dependent | 0.52 | 0.52 | +0.00 | 0.52 [0.47, 0.56] | +0.00 [+0.00, +0.00] | n/a | 2% / 2% / 2% | 0.66 / 0.66 | 1.00 [1.00, 1.00] (n=1) |
| family | none | 1.00 | 1.00 | +0.00 | 1.00 [1.00, 1.00] | +0.00 [+0.00, +0.00] | n/a | 100% / 100% / 100% | 0.05 / 0.05 | 1.00 [1.00, 1.00] (n=60) |
| family | door_jam | 0.82 | 0.91 | +0.09 | 0.88 [0.81, 0.94] | +0.06 [+0.00, +0.12] | 65% [0%, 143%] | 63% / 80% / 78% | 0.04 / 0.04 | 1.07 [1.02, 1.13] (n=45) |
| family | carpet_flashover | 1.00 | 1.00 | +0.00 | 1.00 [1.00, 1.00] | +0.00 [+0.00, +0.00] | n/a | 100% / 100% / 100% | 0.05 / 0.05 | 1.00 [1.00, 1.00] (n=60) |
| family | compound | 0.79 | 0.84 | +0.04 | 0.78 [0.71, 0.85] | -0.01 [-0.06, +0.03] | n/a | 52% / 67% / 55% | 0.05 / 0.05 | 1.01 [1.00, 1.02] (n=33) |
| family | hidden_dependent | 0.75 | 0.99 | +0.24 | 0.99 [0.97, 1.00] | +0.24 [+0.22, +0.25] | 100% [100%, 100%] | 0% / 95% / 95% | 0.04 / 0.04 | 1.00 [1.00, 1.00] (n=57) |
| family_dog | none | 0.99 | 0.99 | +0.00 | 0.99 [0.97, 1.00] | +0.00 [+0.00, +0.00] | n/a | 95% / 95% / 95% | 0.03 / 0.03 | 1.00 [1.00, 1.00] (n=57) |
| family_dog | door_jam | 0.81 | 0.90 | +0.09 | 0.88 [0.82, 0.93] | +0.07 [+0.02, +0.12] | 73% [26%, 138%] | 50% / 77% / 68% | 0.03 / 0.03 | 1.05 [1.01, 1.09] (n=41) |
| family_dog | carpet_flashover | 0.90 | 0.94 | +0.04 | 0.94 [0.89, 0.98] | +0.04 [+0.00, +0.09] | n/a | 82% / 87% / 87% | 0.03 / 0.03 | 1.00 [1.00, 1.00] (n=52) |
| family_dog | compound | 0.67 | 0.70 | +0.03 | 0.71 [0.62, 0.78] | +0.04 [-0.03, +0.12] | n/a | 33% / 37% / 37% | 0.03 / 0.03 | 1.00 [1.00, 1.01] (n=21) |
| family_dog | hidden_dependent | n/a: the house cannot host this (too few slots) |||||||||
