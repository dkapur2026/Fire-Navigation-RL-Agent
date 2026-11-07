# WP-H: Technical report  (E7)

**Goal.** A two-page write-up in the shape of a short R&D report, where every number can be regenerated from a command.

## Read first
`docs/results.md`, `docs/REWARD.md`, `docs/ARCHITECTURE.md`, `docs/EXPLAINING_IT.md`, the output of WP-F. Needs WP-F.

## Deliverables: `docs/REPORT.md`*
1. **Problem** (a paragraph a non-expert can follow, then the precise statement).
2. **Method:** the rule/patch representation, the planner, the detector, the repair search, the explanation layer. One diagram.
3. **Experimental setup:** blueprints, households, novelties, seeds, metrics, oracle and control. State what is held fixed.
4. **Results:** the main table (gap closed with confidence intervals), the recovery curve, the false-alarm vs delay trade-off,
   the ablations. Figures saved under `docs/media/`.
5. **Reward and benchmark validity:** the reward-hacking audit and the `wood_flashover` negative result, summarized honestly.
6. **Limitations and next steps:** what the scaffold does not show (toy fire, parameter novelties only, no structural novelty),
   and the two or three experiments that would most change your mind.
7. **Reproducibility appendix:** a table mapping each figure and number to the exact command that produces it.

## Done when
```bash
python scripts/check_report.py     # (create it) every `command:` line in the appendix runs, and every table number matches its source file
```
At minimum, someone should be able to copy each command from the appendix and get the number in the report.

## Constraints
Two pages means about 900 to 1200 words plus figures. Plain language first, equations only where needed. No claim the data does
not support; if a result is weaker than hoped, say so in the abstract, not just in a footnote.

## Docs
Link the report from `README.md`. Flip E7 in `docs/SPEC.md`.
