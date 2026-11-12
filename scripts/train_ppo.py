"""Train the PPO baseline (WP-C). Same as `python -m emberpath.rl_baseline`; see docs/RL.md for the runs reported.

    python scripts/train_ppo.py --blueprint ranch --profiles single kids \
        --novelties none door_jam carpet_flashover hidden_dependent --steps 1000000 --seed 0 --out runs/ppo
"""

import sys

from emberpath.rl_baseline import main

if __name__ == "__main__":
    sys.exit(main())
