# Reward audit: ranch / single / door_jam / 20 seeds

Episode return. Higher is better. Honest effort should outrank every shortcut.
Means are pulled down by the few failure episodes, which are penalized heavily on purpose;
the median shows the typical episode.

| policy | legacy (mean) | new (mean) | new (median) | fraction rescued | outcomes |
|---|---|---|---|---|---|
| oracle | 2.28 | 4.40 | 5.44 | 0.95 | partial 1/20, success 19/20 |
| baseline | -1.66 | -14.03 | -14.19 | 0.00 | partial 4/20, timeout 16/20 |
| adaptive | 1.71 | 1.47 | 5.39 | 0.80 | partial 4/20, success 16/20 |
| stall | -1.81 | -14.08 | -14.19 | 0.00 | agent_lost 1/20, timeout 19/20 |
| suicide | -1.30 | -13.91 | -13.37 | 0.00 | agent_lost 20/20 |
| abandon | -1.31 | -14.03 | -14.19 | 0.00 | partial 11/20, timeout 9/20 |
| reward_snoop | -1.66 | -14.03 | -14.19 | 0.00 | partial 4/20, timeout 16/20 |
| door_shy | -1.66 | -14.03 | -14.19 | 0.00 | partial 4/20, timeout 16/20 |
| jam_snoop | -1.66 | -14.03 | -14.19 | 0.00 | partial 4/20, timeout 16/20 |

Ranking under legacy reward: oracle > adaptive > suicide > abandon > baseline > reward_snoop > door_shy > jam_snoop > stall
Ranking under new reward:    oracle > adaptive > suicide > baseline > abandon > reward_snoop > door_shy > jam_snoop > stall

EXPLOIT under legacy reward: suicide (-1.30) outscores a struggling honest agent (-1.66).
EXPLOIT under new reward: suicide (-13.91) outscores a struggling honest agent (-14.03).
ok: reward_snoop plays exactly like baseline, so the new reward gives away no hidden person.
ok: jam_snoop plays exactly like baseline, so the new reward gives away no unfound jammed door.
