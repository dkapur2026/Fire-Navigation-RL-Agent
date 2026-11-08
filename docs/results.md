# Results: ranch blueprint, 60 seeds per cell

Mean score (fraction of dependents rescued, 0 if the rescuer dies). Same seeds and same fires for every agent.
`oracle gap` = oracle - baseline: how much a correct rulebook is worth. With no novelty it must be +0.00.
`gap closed` = (adaptive - baseline) / (oracle - baseline); `n/a` when the oracle gap is under 0.05. Brackets: 95% paired-bootstrap intervals over seeds. Regenerate with `make results`;
ablations, detection delays and recovery curves: `python scripts/adaptive_study.py` (docs/adaptive_study.md).

| household | novelty | baseline | oracle | oracle gap | adaptive | adaptive - baseline | gap closed | all out (b / o / a) |
|---|---|---|---|---|---|---|---|---|
| single | none | 0.85 | 0.85 | +0.00 | 0.85 [0.77, 0.93] | +0.00 [+0.00, +0.00] | n/a | 85% / 85% / 85% |
| single | door_jam | 0.03 | 0.82 | +0.78 | 0.82 [0.72, 0.92] | +0.78 [+0.67, +0.90] | 100% [88%, 114%] | 3% / 82% / 82% |
| single | carpet_flashover | 0.88 | 0.88 | +0.00 | 0.88 [0.80, 0.95] | +0.00 [+0.00, +0.00] | n/a | 88% / 88% / 88% |
| single | compound | 0.02 | 0.20 | +0.18 | 0.20 [0.10, 0.30] | +0.18 [+0.08, +0.28] | 100% [67%, 150%] | 2% / 20% / 20% |
| single | hidden_dependent | 0.42 | 0.83 | +0.41 | 0.93 [0.87, 0.97] | +0.50 [+0.45, +0.55] | 122% [100%, 157%] | 0% / 77% / 88% |
| married | none | 0.95 | 0.95 | +0.00 | 0.95 [0.88, 1.00] | +0.00 [+0.00, +0.00] | n/a | 95% / 95% / 95% |
| married | door_jam | 0.90 | 0.90 | +0.00 | 0.92 [0.85, 0.98] | +0.02 [-0.03, +0.07] | n/a | 68% / 90% / 92% |
| married | carpet_flashover | 0.98 | 0.98 | +0.00 | 0.98 [0.95, 1.00] | +0.00 [+0.00, +0.00] | n/a | 98% / 98% / 98% |
| married | compound | 0.77 | 0.82 | +0.05 | 0.80 [0.70, 0.90] | +0.03 [-0.03, +0.10] | n/a | 58% / 82% / 80% |
| married | hidden_dependent | 0.73 | 0.88 | +0.14 | 0.93 [0.87, 0.97] | +0.19 [+0.12, +0.26] | 135% [92%, 230%] | 53% / 82% / 88% |
| kids | none | 0.93 | 0.93 | +0.00 | 0.93 [0.85, 0.98] | +0.00 [+0.00, +0.00] | n/a | 92% / 92% / 92% |
| kids | door_jam | 0.47 | 0.75 | +0.28 | 0.67 [0.58, 0.75] | +0.20 [+0.10, +0.28] | 71% [42%, 109%] | 0% / 63% / 43% |
| kids | carpet_flashover | 0.96 | 0.96 | +0.00 | 0.96 [0.92, 0.99] | +0.00 [+0.00, +0.00] | n/a | 92% / 92% / 92% |
| kids | compound | 0.39 | 0.40 | +0.01 | 0.39 [0.34, 0.44] | +0.00 [-0.07, +0.07] | n/a | 0% / 0% / 0% |
| kids | hidden_dependent | 0.62 | 0.92 | +0.30 | 0.92 [0.84, 0.98] | +0.30 [+0.26, +0.33] | 100% [100%, 100%] | 0% / 90% / 90% |
| kids_dog | none | 0.92 | 0.92 | +0.00 | 0.92 [0.84, 0.98] | +0.00 [+0.00, +0.00] | n/a | 90% / 90% / 90% |
| kids_dog | door_jam | 0.64 | 0.84 | +0.19 | 0.81 [0.74, 0.86] | +0.16 [+0.09, +0.22] | 83% [52%, 126%] | 0% / 68% / 50% |
| kids_dog | carpet_flashover | 0.96 | 0.96 | +0.00 | 0.96 [0.91, 0.99] | +0.00 [+0.00, +0.00] | n/a | 90% / 90% / 90% |
| kids_dog | compound | 0.56 | 0.58 | +0.02 | 0.59 [0.54, 0.63] | +0.03 [-0.04, +0.09] | n/a | 0% / 0% / 0% |
| kids_dog | hidden_dependent | 0.69 | 0.99 | +0.30 | 0.99 [0.96, 1.00] | +0.30 [+0.25, +0.36] | 100% [100%, 100%] | 0% / 98% / 98% |
| family | none | 0.93 | 0.93 | +0.00 | 0.93 [0.87, 0.98] | +0.00 [+0.00, +0.00] | n/a | 88% / 88% / 88% |
| family | door_jam | 0.93 | 0.93 | +0.00 | 0.93 [0.87, 0.98] | +0.00 [+0.00, +0.00] | n/a | 88% / 88% / 88% |
| family | carpet_flashover | 0.88 | 0.88 | +0.00 | 0.88 [0.82, 0.94] | +0.00 [+0.00, +0.00] | n/a | 78% / 78% / 78% |
| family | compound | 0.79 | 0.79 | +0.00 | 0.79 [0.71, 0.87] | +0.00 [+0.00, +0.00] | n/a | 63% / 63% / 63% |
| family | hidden_dependent | 0.70 | 0.88 | +0.18 | 0.88 [0.80, 0.95] | +0.18 [+0.11, +0.25] | 100% [100%, 100%] | 0% / 80% / 80% |
| family_dog | none | 0.89 | 0.89 | +0.00 | 0.89 [0.81, 0.96] | +0.00 [+0.00, +0.00] | n/a | 83% / 83% / 83% |
| family_dog | door_jam | 0.91 | 0.88 | -0.03 | 0.88 [0.80, 0.95] | -0.03 [-0.11, +0.06] | n/a | 88% / 85% / 85% |
| family_dog | carpet_flashover | 0.93 | 0.93 | +0.00 | 0.93 [0.86, 0.98] | +0.00 [+0.00, +0.00] | n/a | 90% / 90% / 90% |
| family_dog | compound | 0.61 | 0.87 | +0.26 | 0.87 [0.79, 0.94] | +0.26 [+0.15, +0.38] | 100% [100%, 100%] | 37% / 80% / 80% |
| family_dog | hidden_dependent | n/a: the house cannot host this (too few slots) |||||||
