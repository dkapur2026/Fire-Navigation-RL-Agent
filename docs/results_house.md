# Results: house blueprint, 60 seeds per cell

Mean score (fraction of dependents rescued, 0 if the rescuer dies). Same seeds and same fires for every agent.
`oracle gap` = oracle - baseline: how much a correct rulebook is worth. With no novelty it must be +0.00.
`gap closed` = (adaptive - baseline) / (oracle - baseline); `n/a` when the oracle gap is under 0.05. Brackets: 95% paired-bootstrap intervals over seeds. Regenerate with `make results`;
ablations, detection delays and recovery curves: `python scripts/adaptive_study.py` (docs/adaptive_study.md).

| household | novelty | baseline | oracle | oracle gap | adaptive | adaptive - baseline | gap closed | all out (b / o / a) |
|---|---|---|---|---|---|---|---|---|
| single | none | 1.00 | 1.00 | +0.00 | 1.00 [1.00, 1.00] | +0.00 [+0.00, +0.00] | n/a | 100% / 100% / 100% |
| single | door_jam | 0.03 | 1.00 | +0.97 | 1.00 [1.00, 1.00] | +0.97 [+0.92, +1.00] | 100% [100%, 100%] | 3% / 100% / 100% |
| single | carpet_flashover | 1.00 | 1.00 | +0.00 | 1.00 [1.00, 1.00] | +0.00 [+0.00, +0.00] | n/a | 100% / 100% / 100% |
| single | compound | 0.05 | 1.00 | +0.95 | 1.00 [1.00, 1.00] | +0.95 [+0.88, +1.00] | 100% [100%, 100%] | 5% / 100% / 100% |
| single | hidden_dependent | 0.50 | 1.00 | +0.50 | 1.00 [1.00, 1.00] | +0.50 [+0.50, +0.50] | 100% [100%, 100%] | 0% / 100% / 100% |
| married | none | 1.00 | 1.00 | +0.00 | 1.00 [1.00, 1.00] | +0.00 [+0.00, +0.00] | n/a | 100% / 100% / 100% |
| married | door_jam | 1.00 | 1.00 | +0.00 | 1.00 [1.00, 1.00] | +0.00 [+0.00, +0.00] | n/a | 100% / 100% / 100% |
| married | carpet_flashover | 1.00 | 1.00 | +0.00 | 1.00 [1.00, 1.00] | +0.00 [+0.00, +0.00] | n/a | 100% / 100% / 100% |
| married | compound | 1.00 | 1.00 | +0.00 | 1.00 [1.00, 1.00] | +0.00 [+0.00, +0.00] | n/a | 100% / 100% / 100% |
| married | hidden_dependent | 0.50 | 1.00 | +0.50 | 1.00 [1.00, 1.00] | +0.50 [+0.50, +0.50] | 100% [100%, 100%] | 0% / 100% / 100% |
| kids | none | 1.00 | 1.00 | +0.00 | 1.00 [1.00, 1.00] | +0.00 [+0.00, +0.00] | n/a | 100% / 100% / 100% |
| kids | door_jam | 0.01 | 1.00 | +0.99 | 1.00 [1.00, 1.00] | +0.99 [+0.97, +1.00] | 100% [100%, 100%] | 0% / 100% / 100% |
| kids | carpet_flashover | 1.00 | 1.00 | +0.00 | 1.00 [1.00, 1.00] | +0.00 [+0.00, +0.00] | n/a | 100% / 100% / 100% |
| kids | compound | 0.05 | 1.00 | +0.95 | 1.00 [1.00, 1.00] | +0.95 [+0.88, +1.00] | 100% [100%, 100%] | 5% / 100% / 100% |
| kids | hidden_dependent | 0.67 | 1.00 | +0.33 | 0.87 [0.81, 0.93] | +0.21 [+0.14, +0.26] | 62% [42%, 78%] | 0% / 100% / 72% |
| kids_dog | none | 1.00 | 1.00 | +0.00 | 1.00 [1.00, 1.00] | +0.00 [+0.00, +0.00] | n/a | 100% / 100% / 100% |
| kids_dog | door_jam | 1.00 | 1.00 | +0.00 | 1.00 [1.00, 1.00] | +0.00 [+0.00, +0.00] | n/a | 100% / 100% / 100% |
| kids_dog | carpet_flashover | 0.07 | 0.07 | +0.00 | 0.07 [0.02, 0.13] | +0.00 [+0.00, +0.00] | n/a | 3% / 3% / 3% |
| kids_dog | compound | 0.87 | 0.87 | +0.00 | 0.87 [0.81, 0.93] | +0.00 [+0.00, +0.00] | n/a | 72% / 72% / 72% |
| kids_dog | hidden_dependent | 0.75 | 1.00 | +0.25 | 1.00 [1.00, 1.00] | +0.25 [+0.25, +0.25] | 100% [100%, 100%] | 0% / 100% / 100% |
| family | none | 0.96 | 0.96 | +0.00 | 0.96 [0.93, 0.99] | +0.00 [+0.00, +0.00] | n/a | 88% / 88% / 88% |
| family | door_jam | 0.96 | 0.96 | +0.00 | 0.96 [0.93, 0.99] | +0.00 [+0.00, +0.00] | n/a | 88% / 88% / 88% |
| family | carpet_flashover | 0.06 | 0.06 | +0.00 | 0.06 [0.01, 0.10] | +0.00 [+0.00, +0.00] | n/a | 0% / 0% / 0% |
| family | compound | 0.68 | 0.68 | +0.00 | 0.68 [0.62, 0.73] | +0.00 [+0.00, +0.00] | n/a | 20% / 20% / 20% |
| family | hidden_dependent | 0.72 | 0.92 | +0.20 | 0.92 [0.85, 0.97] | +0.20 [+0.14, +0.24] | 100% [100%, 100%] | 0% / 90% / 90% |
| family_dog | none | 0.97 | 0.97 | +0.00 | 0.97 [0.95, 1.00] | +0.00 [+0.00, +0.00] | n/a | 92% / 92% / 92% |
| family_dog | door_jam | 0.97 | 0.99 | +0.01 | 0.99 [0.97, 1.00] | +0.01 [-0.00, +0.03] | n/a | 92% / 95% / 95% |
| family_dog | carpet_flashover | 0.06 | 0.35 | +0.29 | 0.06 [0.01, 0.11] | +0.00 [+0.00, +0.00] | 0% [0%, 0%] | 0% / 0% / 0% |
| family_dog | compound | 0.40 | 0.76 | +0.36 | 0.78 [0.72, 0.83] | +0.38 [+0.30, +0.45] | 105% [100%, 114%] | 22% / 27% / 27% |
| family_dog | hidden_dependent | n/a: the house cannot host this (too few slots) |||||||
