"""Phase 4: a pure deep-RL baseline to compare against.

STATUS: interface only (needs `pip install -e .[rl]`). See docs/ROADMAP.md, Phase 4.

Purpose: answer the question "why not just use deep RL?" with data. Train a PPO policy on the
nominal world (no novelty), then evaluate it two ways on a world that changes mid-episode:
    (a) frozen policy                         -> how badly does it break?
    (b) keeps fine-tuning after the change    -> how many episodes/samples until it recovers?
Compare (b)'s sample cost against the adaptive agent's recovery time (ticks until the repaired
belief produces good behavior). The expected story: the symbolic repair needs dozens of
observations; RL needs thousands of episodes. Report whatever you find.

Suggested design
----------------
* Policy: small CNN over the (H, W, 10) grid from ``env.encode`` -> 5 action logits + value.
* Use ``emberpath.env.EmberPathEnv``. Seeds: train on 0..999, evaluate on 1000..1099.
* Log: success rate vs. env steps, and "episodes to recover to 90% of pre-novelty score".
* Implement PPO yourself (a good PyTorch exercise, ~150 lines) or use stable-baselines3,
  but keep it behind the ``rl`` extra so the core install stays light.
"""

from __future__ import annotations


def train_ppo(*args, **kwargs):
    raise NotImplementedError("Phase 4: PPO baseline is not built yet.")


def evaluate_policy(*args, **kwargs):
    raise NotImplementedError("Phase 4: PPO baseline is not built yet.")
