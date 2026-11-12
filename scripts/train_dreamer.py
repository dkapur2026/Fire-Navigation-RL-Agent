"""Train the DreamerV3 agent (see docs/DREAMER.md for the runs reported). Same as `python -m emberpath.dreamer`.

    python scripts/train_dreamer.py --blueprint studio --profiles single --novelties none --steps 60000 \
        --seed 0 --out runs/dreamer_studio
    python scripts/train_dreamer.py --blueprint ranch --profiles single kids \
        --novelties none door_jam carpet_flashover hidden_dependent --steps 300000 --seed 0 \
        --init runs/dreamer_studio/final.pt --out runs/dreamer_ranch

Training fires come from seeds 0..999 only; evaluation uses 1000..1099 (disjointness asserted in the module).
"""

import sys

from emberpath.dreamer import main

if __name__ == "__main__":
    sys.exit(main())
