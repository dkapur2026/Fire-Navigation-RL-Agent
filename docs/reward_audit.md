# Reward audit: ranch / single / door_jam / 20 seeds

Episode return. Higher is better. Honest effort should outrank every shortcut.
Means are pulled down by the few failure episodes, which are penalized heavily on purpose;
the median shows the typical episode.

| policy | legacy (mean) | new (mean) | new (median) | fraction rescued | outcomes |
|---|---|---|---|---|---|
| oracle | 1.84 | 0.09 | 1.47 | 0.85 | agent_lost 2/20, partial 1/20, success 17/20 |
| baseline | -1.66 | -5.01 | -4.39 | 0.00 | agent_lost 1/20, partial 4/20, timeout 15/20 |
| stall | -1.82 | -5.34 | -5.55 | 0.00 | agent_lost 2/20, timeout 18/20 |
| suicide | -1.24 | -6.20 | -5.09 | 0.00 | agent_lost 20/20 |
| abandon | -1.24 | -4.15 | -4.10 | 0.00 | partial 11/20, timeout 9/20 |

Ranking under legacy reward: oracle > abandon > suicide > baseline > stall
Ranking under new reward:    oracle > abandon > baseline > stall > suicide

EXPLOIT under legacy reward: suicide (-1.24) outscores a struggling honest agent (-1.66).
ok under new reward: suicide (-6.20) scores below a struggling honest agent (-5.01).
