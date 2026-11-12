"""A compact DreamerV3 (Hafner et al., 2023) for the burning house: a world model with memory, and a policy trained
inside it. Needs ``pip install -e .[rl]``. Results, setup and hyperparameters are in docs/DREAMER.md.

Why try it after PPO: PPO's policy has no memory, but the rescuer often needs one (a door it bumped, a person it
heard, how the fire has moved). Dreamer's recurrent latent state carries that, and learning a model of the house
lets every real step be replayed many times, which should help with a CPU budget.

What is here (the DreamerV3 essentials, kept small enough to train on a CPU)
------------------------------------------------------------------------------
* World model (``WorldModel``): a CNN encoder over the ``env.encode`` grid (smoke dropped, exactly the PPO input);
  an RSSM with a GRU deterministic state and a categorical stochastic state (``stoch`` x ``classes``, 1% unimix,
  straight-through samples); a CNN decoder; a reward head and a value head that predict a two-hot distribution over
  symexp-spaced buckets; a continue head. Inputs are symlog-squashed. KL balancing: 0.5 x KL(sg(post) || prior) +
  0.1 x KL(post || sg(prior)), each clipped below at 1 nat (free bits).
* Actor-critic trained in imagination: rollouts of ``horizon`` steps from every replayed posterior state, lambda-
  returns, advantages divided by max(1, EMA of the 5-95 percentile range of returns), REINFORCE for the discrete
  actor with entropy bonus 3e-4 and 1% unimix, and a two-hot critic regularised toward a slow EMA copy of itself.
* A replay buffer of per-env streams with uniform sequence sampling, and train-ratio control (replayed steps per
  env step).
* ``DreamerAgent``: the ``Agent`` interface. Evidence discipline: it builds its input with
  ``rl_baseline.encode_observation`` from the ``Observation`` and its own ``Event``s, exactly as ``PPOAgent`` does,
  never from the world, and carries its recurrent state across ticks.

Deviations from the paper, all for CPU cost or because the input is a grid of flags rather than an image: the
decoder is trained with a per-cell binary cross-entropy on the 0/1 channels (symlog-MSE on the clock channel)
instead of MSE on pixels; imagination starts from a random 512 of the 1024 replayed states
per update; a plain ``nn.GRUCell`` (no layer norm inside the GRU); Adam instead of the paper's
optimiser settings per network otherwise as in the paper (1e-4 model, 3e-5 actor and critic); continue
probabilities (not their mode) discount imagined returns.

Optional, off by default (docs/DREAMER.md, "Planner-guided Dreamer"): a demo-seeded replay (``collect_demos``,
``demo_fraction``, ``sample_mixed``) that mixes AdaptiveAgent episodes from training seeds into world-model batches
and tapers their share to exactly 0, and a hearing plane in the input (``heard``). The agent itself never calls a
planner.

torch is imported inside the functions that need it (core stays light).

Command line: ``python scripts/train_dreamer.py --help``.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from emberpath.blueprint import Blueprint
from emberpath.env import TIME_CHANNEL
from emberpath.planner import Decision
from emberpath.rl_baseline import (
    ACTION_WORDS,
    EVAL_SEEDS,
    HEARD_CHANNEL,
    POLICY_CHANNELS,
    TRAIN_SEEDS,
    Phase,
    Task,
    canvas_for,
    default_phases,
    encode_observation,
    pad_to,
    update_heard,
    with_heard,
)
from emberpath.rules import key_cell, parse_key
from emberpath.world import Event, Observation

assert not set(TRAIN_SEEDS) & set(EVAL_SEEDS), "train and eval seeds must be disjoint"
N_ACTIONS = 5
N_IN = len(POLICY_CHANNELS)  # 10: every env.encode channel except smoke; the clock is the last one
N_BINARY = N_IN - 1
assert POLICY_CHANNELS[-1] == TIME_CHANNEL
# With ``heard`` the rescuer's "someone is calling here" plane (rl_baseline.with_heard, built from its own hearing
# events, as PPO's variant uses) goes in just before the clock, so the clock stays the last input channel.
HEARD_CHANNELS = tuple(c for c in POLICY_CHANNELS if c != TIME_CHANNEL) + (HEARD_CHANNEL, TIME_CHANNEL)


def channels_for(cfg) -> tuple[int, ...]:
    """The input planes (indices into env.encode, plus HEARD_CHANNEL) this config's model reads, clock last."""
    return HEARD_CHANNELS if getattr(cfg, "heard", False) else POLICY_CHANNELS


def frame(padded: np.ndarray, channels: Sequence[int], heard: dict | None = None) -> np.ndarray:
    """The model input from a padded (H, W, N_CHANNELS) encoding (and the hearing map, if the model reads it)."""
    if HEARD_CHANNEL in channels:
        padded = with_heard(padded, heard or {})
    return padded[..., list(channels)]


@dataclass(frozen=True)
class DreamerConfig:
    n_envs: int = 16
    batch: int = 16
    seq_len: int = 64
    train_ratio: float = 64.0  # replayed steps (batch x seq_len per update) per env step
    prefill: int = 5000  # env steps collected before the first update (random actions unless --init)
    capacity: int = 1_000_000  # replay size in steps (all envs together)
    deter: int = 256
    stoch: int = 16
    classes: int = 16
    hidden: int = 256
    cnn: int = 16  # channels of the first conv (the second has twice as many)
    unimix: float = 0.01
    bins: int = 255
    bin_range: float = 20.0  # buckets are uniform in symlog space over [-bin_range, bin_range]
    free_nats: float = 1.0
    dyn_scale: float = 0.5
    rep_scale: float = 0.1
    horizon: int = 15
    imag_starts: int = 512  # imagine from this many of the batch x seq_len replayed states (0 = all), for CPU
    gamma: float = 0.997
    lam: float = 0.95
    entropy: float = 3e-4
    ret_norm_decay: float = 0.99
    ret_norm_limit: float = 1.0
    slow_critic_decay: float = 0.98
    slow_reg: float = 1.0
    model_lr: float = 1e-4
    actor_lr: float = 3e-5
    critic_lr: float = 3e-5
    model_clip: float = 1000.0
    ac_clip: float = 100.0
    max_steps: int = 150
    seed: int = 0
    torch_threads: int = 4
    # Demo-seeded replay (docs/DREAMER.md, "Planner-guided Dreamer"). The defaults (0) reproduce unguided Dreamer
    # exactly. A fraction ``demo_frac`` of each world-model batch's sequences comes from a separate replay of
    # AdaptiveAgent episodes on training seeds; it falls linearly to exactly 0 at env step ``demo_until`` of the run
    # (0 = 60% of the run's planned steps) and stays 0, so the end of training is the agent's own experience only.
    demo_frac: float = 0.0
    demo_until: int = 0
    heard: bool = False  # add the hearing plane to the input (HEARD_CHANNELS); False = the original 10 planes
    # Actor-side imitation ("DreamerV3 + imitation", runs/formal_eval). The default (0) leaves the actor untouched. With
    # ``actor_bc_lr > 0`` the demo rows of each world-model batch also train the actor by behaviour cloning: cross-
    # entropy of the AdaptiveAgent's next action given the (detached) posterior feature of the frame it acted on, with
    # its own Adam whose learning rate falls linearly from ``actor_bc_lr`` to exactly 0 at env step ``actor_bc_until``
    # (0 = the demo schedule's end, ``demo_until`` or 60% of the run). Needs ``demo_frac > 0``. The imagination
    # (REINFORCE) actor loss is unchanged, so after the anneal the actor learns only from its own world model.
    actor_bc_lr: float = 0.0
    actor_bc_until: int = 0


def actor_bc_scale(step: int, cfg: DreamerConfig, total_steps: int) -> float:
    """The fraction of ``actor_bc_lr`` used at env step ``step``: 1 at the start, falling linearly to exactly 0 at
    ``actor_bc_until`` (default: where the demo share ends) and 0 from then on; 0 throughout when imitation is off."""
    if cfg.actor_bc_lr <= 0:
        return 0.0
    until = cfg.actor_bc_until or cfg.demo_until or int(0.6 * total_steps)
    if until <= 0 or step >= until:
        return 0.0
    return float(1.0 - step / until)


def actor_bc_loss(model, feat, act, first, term):
    """Behaviour cloning of the actor on demo sequences: (B, T, F) posterior features, and the replay's (B, T)
    ``act`` (the action that led to each frame), ``first`` and ``term``. The teacher's action on frame t is act[t+1],
    valid unless frame t is terminal or t+1 starts a new episode. Returns (loss, accuracy, number of targets)."""
    import torch

    f = feat[:, :-1].detach()
    target = act[:, 1:].long()
    valid = (1.0 - term[:, :-1]) * (1.0 - first[:, 1:])
    n = valid.sum()
    logp = model.ac.policy_logp(f.reshape(-1, f.shape[-1])).view(*target.shape, -1)
    nll = -logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    loss = (nll * valid).sum() / n.clamp(min=1.0)
    with torch.no_grad():
        acc = (((logp.argmax(-1) == target).float() * valid).sum() / n.clamp(min=1.0)).item()
    return loss, acc, int(n.item())


def demo_fraction(step: int, cfg: DreamerConfig, total_steps: int) -> float:
    """The share of a world-model batch drawn from planner demos at env step ``step`` of the run: ``demo_frac``
    falling linearly to exactly 0 at ``demo_until`` (default 60% of ``total_steps``), and 0 from then on."""
    if cfg.demo_frac <= 0:
        return 0.0
    until = cfg.demo_until or int(0.6 * total_steps)
    if until <= 0 or step >= until:
        return 0.0
    return float(cfg.demo_frac * (1.0 - step / until))


def demo_rows(batch: int, frac: float) -> int:
    """How many of the ``batch`` sequences come from the demo replay."""
    return int(min(batch, max(0, round(batch * frac))))


# --------------------------------------------------------------------------- maths (pure functions, tested)


def symlog(x):
    import torch

    return torch.sign(x) * torch.log1p(x.abs())


def symexp(x):
    import torch

    return torch.sign(x) * torch.expm1(x.abs())


def make_bins(n: int = 255, limit: float = 20.0):
    import torch

    return torch.linspace(-limit, limit, n)


def twohot(x, bins):
    """Two-hot encoding of raw values ``x`` (any shape) over buckets ``bins`` in symlog space: the weight is split
    between the two buckets either side of symlog(x), so the expectation in symlog space is exact."""
    import torch

    y = symlog(x).clamp(bins[0], bins[-1])
    k = (torch.searchsorted(bins, y.contiguous(), right=True) - 1).clamp(0, len(bins) - 2)
    lo, hi = bins[k], bins[k + 1]
    w_hi = ((y - lo) / (hi - lo)).clamp(0, 1)
    out = torch.zeros(*x.shape, len(bins), dtype=x.dtype)
    out.scatter_(-1, k.unsqueeze(-1), (1 - w_hi).unsqueeze(-1))
    out.scatter_add_(-1, (k + 1).unsqueeze(-1), w_hi.unsqueeze(-1))
    return out


def twohot_mean(probs, bins):
    """The decoded value of a distribution over buckets: symexp of its expectation in symlog space."""
    return symexp((probs * bins).sum(-1))


def twohot_nll(logits, target, bins):
    """Cross-entropy of a bucket distribution (logits) against the two-hot encoding of ``target``."""
    import torch

    return -(twohot(target, bins) * torch.log_softmax(logits, -1)).sum(-1)


def unimix_logp(logits, mix: float):
    """log of (1 - mix) * softmax(logits) + mix / K: every class keeps some probability, so KLs stay finite."""
    import torch

    probs = torch.softmax(logits, -1)
    if mix > 0:
        probs = (1 - mix) * probs + mix / logits.shape[-1]
    return torch.log(probs)


def categorical_kl(logp, logq):
    """KL(p || q) for (..., S, K) log-probabilities, summed over the S latents."""
    return (logp.exp() * (logp - logq)).sum((-2, -1))


def kl_balance(post_logp, prior_logp, free: float = 1.0, dyn_scale: float = 0.5, rep_scale: float = 0.1):
    """DreamerV3's KL term. The dynamics loss trains the prior toward the (frozen) posterior; the representation
    loss trains the posterior toward the (frozen) prior, five times more weakly. Each is clipped below at ``free``
    nats, so below that there is no gradient. Returns (loss per element, raw dyn KL, raw rep KL)."""
    import torch

    dyn = categorical_kl(post_logp.detach(), prior_logp)
    rep = categorical_kl(post_logp, prior_logp.detach())
    loss = dyn_scale * torch.clamp(dyn, min=free) + rep_scale * torch.clamp(rep, min=free)
    return loss, dyn, rep


def lambda_return(rewards, conts, values, gamma: float, lam: float):
    """TD(lambda) returns for an imagined trajectory. ``rewards[t]`` and ``conts[t]`` are the reward and continue
    probability on arriving at state t+1; ``values`` has one more entry (v_0 .. v_H). Recursion:
    R_t = r_{t+1} + gamma c_{t+1} ((1 - lam) v_{t+1} + lam R_{t+1}),  R_H = v_H."""
    import torch

    H = rewards.shape[0]
    out = []
    R = values[H]
    for t in reversed(range(H)):
        R = rewards[t] + gamma * conts[t] * ((1 - lam) * values[t + 1] + lam * R)
        out.append(R)
    return torch.stack(out[::-1])


# --------------------------------------------------------------------------- networks


@lru_cache(maxsize=None)
def _modules():
    """The network classes, built on first use so that importing this module does not import torch."""
    import torch
    from torch import nn

    def mlp(n_in: int, hidden: int, n_out: int, layers: int = 2, zero_last: bool = False) -> nn.Sequential:
        mods: list = []
        d = n_in
        for _ in range(layers):
            mods += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.SiLU()]
            d = hidden
        last = nn.Linear(d, n_out)
        if zero_last:
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
        return nn.Sequential(*mods, last)

    class RSSM(nn.Module):
        def __init__(self, embed: int, cfg: DreamerConfig) -> None:
            super().__init__()
            self.S, self.K, self.D, self.mix = cfg.stoch, cfg.classes, cfg.deter, cfg.unimix
            z = self.S * self.K
            self.img_in = nn.Sequential(nn.Linear(z + N_ACTIONS, cfg.hidden), nn.LayerNorm(cfg.hidden), nn.SiLU())
            self.gru = nn.GRUCell(cfg.hidden, self.D)
            self.prior_net = mlp(self.D, cfg.hidden, z, layers=1)
            self.post_net = mlp(self.D + embed, cfg.hidden, z, layers=1)

        def initial(self, n: int):
            return torch.zeros(n, self.D), torch.zeros(n, self.S * self.K)

        def dist(self, logits):
            return unimix_logp(logits.view(*logits.shape[:-1], self.S, self.K), self.mix)

        def sample(self, logp, gen=None):
            """A one-hot sample per latent with straight-through gradients."""
            probs = logp.exp()
            idx = torch.multinomial(probs.reshape(-1, self.K), 1, generator=gen).view(*probs.shape[:-1])
            onehot = nn.functional.one_hot(idx, self.K).to(probs.dtype)
            return (onehot + probs - probs.detach()).flatten(-2)

        def mode(self, logp):
            return nn.functional.one_hot(logp.argmax(-1), self.K).to(logp.dtype).flatten(-2)

        def img_step(self, h, z, action_onehot):
            h = self.gru(self.img_in(torch.cat([z, action_onehot], -1)), h)
            return h, self.dist(self.prior_net(h))

        def obs_step(self, h, z, action_onehot, embed, is_first, sample: bool = True, gen=None):
            keep = (1.0 - is_first).unsqueeze(-1)  # a new episode starts from the zero state
            h, z, action_onehot = h * keep, z * keep, action_onehot * keep
            h, prior = self.img_step(h, z, action_onehot)
            post = self.dist(self.post_net(torch.cat([h, embed], -1)))
            z = self.sample(post, gen) if sample else self.mode(post)
            return h, z, post, prior

    class WorldModel(nn.Module):
        def __init__(self, canvas: tuple[int, int], cfg: DreamerConfig) -> None:
            super().__init__()
            H, W = canvas
            c = cfg.cnn
            self.canvas, self.h2, self.w2, self.c2 = canvas, (H + 1) // 2, (W + 1) // 2, 2 * c
            flat = self.c2 * self.h2 * self.w2
            n_in = len(channels_for(cfg))
            self.encoder = nn.Sequential(
                nn.Conv2d(n_in, c, 3, padding=1), nn.SiLU(),
                nn.Conv2d(c, 2 * c, 3, stride=2, padding=1), nn.SiLU(),
                nn.Flatten(), nn.Linear(flat, cfg.hidden), nn.LayerNorm(cfg.hidden), nn.SiLU(),
            )
            self.rssm = RSSM(cfg.hidden, cfg)
            feat = cfg.deter + cfg.stoch * cfg.classes
            self.feat = feat
            self.dec_in = nn.Sequential(nn.Linear(feat, flat), nn.SiLU())
            self.dec = nn.Sequential(
                nn.ConvTranspose2d(2 * c, c, 3, stride=2, padding=1, output_padding=1), nn.SiLU(),
                nn.Conv2d(c, n_in, 3, padding=1),
            )
            self.reward = mlp(feat, cfg.hidden, cfg.bins, zero_last=True)
            self.cont = mlp(feat, cfg.hidden, 1)

        def embed(self, x):
            """x: (..., H, W, N_IN) policy channels -> (..., hidden)."""
            lead = x.shape[:-3]
            z = self.encoder(symlog(x.reshape(-1, *x.shape[-3:])).permute(0, 3, 1, 2))
            return z.view(*lead, -1)

        def decode(self, feat):
            """feat (N, F) -> (N, H, W, N_IN) logits for the 0/1 channels, symlog value for the clock."""
            H, W = self.canvas
            y = self.dec(self.dec_in(feat).view(-1, self.c2, self.h2, self.w2))
            return y[:, :, :H, :W].permute(0, 2, 3, 1)

    class ActorCritic(nn.Module):
        def __init__(self, feat: int, cfg: DreamerConfig) -> None:
            super().__init__()
            self.mix = cfg.unimix
            self.actor = mlp(feat, cfg.hidden, N_ACTIONS)
            self.critic = mlp(feat, cfg.hidden, cfg.bins, zero_last=True)
            self.slow = mlp(feat, cfg.hidden, cfg.bins, zero_last=True)
            self.slow.load_state_dict(self.critic.state_dict())
            for p in self.slow.parameters():
                p.requires_grad_(False)

        def policy_logp(self, feat):
            return unimix_logp(self.actor(feat), self.mix)

    class Dreamer(nn.Module):
        """World model + actor-critic + the return-normalisation statistics (buffers, so they are checkpointed)."""

        def __init__(self, canvas: tuple[int, int], cfg: DreamerConfig) -> None:
            super().__init__()
            self.cfg, self.canvas = cfg, tuple(canvas)
            self.wm = WorldModel(self.canvas, cfg)
            self.ac = ActorCritic(self.wm.feat, cfg)
            self.register_buffer("bins", make_bins(cfg.bins, cfg.bin_range))
            self.register_buffer("ret_lo", torch.zeros(()))
            self.register_buffer("ret_hi", torch.zeros(()))
            self.register_buffer("updates", torch.zeros((), dtype=torch.long))

    return Dreamer


def make_dreamer(canvas: tuple[int, int], cfg: DreamerConfig = DreamerConfig()):
    return _modules()(canvas, cfg)


# --------------------------------------------------------------------------- one update


def _onehot(a):
    import torch

    return torch.nn.functional.one_hot(a.long(), N_ACTIONS).float()


def world_model_loss(model, obs, prev_act, reward, first, term):
    """Observe a replayed batch (B, T, ...) through the RSSM. Returns (loss, metrics, posterior features (B, T, F))."""
    import torch
    import torch.nn.functional as F

    wm, cfg = model.wm, model.cfg
    B, T = obs.shape[:2]
    embed = wm.embed(obs)
    h, z = wm.rssm.initial(B)
    feats, posts, priors = [], [], []
    for t in range(T):
        h, z, post, prior = wm.rssm.obs_step(h, z, _onehot(prev_act[:, t]), embed[:, t], first[:, t])
        feats.append(torch.cat([h, z], -1))
        posts.append(post)
        priors.append(prior)
    feat = torch.stack(feats, 1)
    post, prior = torch.stack(posts, 1), torch.stack(priors, 1)
    flat = feat.reshape(B * T, -1)
    rec = wm.decode(flat).view(B, T, *obs.shape[2:])
    nb = obs.shape[-1] - 1  # the clock is the last channel
    bce = F.binary_cross_entropy_with_logits(rec[..., :nb], obs[..., :nb], reduction="none")
    mse = (rec[..., nb] - symlog(obs[..., nb])).pow(2)
    recon = bce.sum((-3, -2, -1)) + mse.sum((-2, -1))
    rew_logits = wm.reward(flat).view(B, T, -1)
    rew_loss = twohot_nll(rew_logits, reward, model.bins)
    cont_logit = wm.cont(flat).view(B, T)
    cont_loss = F.binary_cross_entropy_with_logits(cont_logit, 1.0 - term, reduction="none")
    kl, dyn, rep = kl_balance(post, prior, cfg.free_nats, cfg.dyn_scale, cfg.rep_scale)
    loss = (recon + rew_loss + cont_loss + kl).mean()
    with torch.no_grad():
        rew_pred = twohot_mean(torch.softmax(rew_logits, -1), model.bins)
        big = reward.abs() > 0.5
        metrics = {
            "wm_loss": loss.item(), "recon": recon.mean().item(), "reward_nll": rew_loss.mean().item(),
            "cont_loss": cont_loss.mean().item(), "kl_dyn": dyn.mean().item(), "kl_rep": rep.mean().item(),
            "reward_abs_err": (rew_pred - reward).abs().mean().item(),
            "reward_abs_err_big": (rew_pred - reward)[big].abs().mean().item() if big.any() else float("nan"),
            "agent_cell_acc": _agent_cell_acc(rec, obs),
        }
    return loss, metrics, feat


def _agent_cell_acc(rec, obs) -> float:
    """How often the decoder's most likely rescuer cell is the true one (channel 3 of the policy input)."""
    import torch

    r = rec[..., 3].flatten(-2).argmax(-1)
    o = obs[..., 3].flatten(-2).argmax(-1)
    return float((r == o).float().mean())


def imagine(model, start_feat, horizon: int):
    """Roll the actor through the world model from (N, F) start features, without gradients (the discrete actor
    learns by REINFORCE, so nothing needs to flow through the dynamics). Returns features (H+1, N, F) and actions
    (H, N)."""
    import torch

    wm, ac = model.wm, model.ac
    D = model.cfg.deter
    h, z = start_feat[:, :D], start_feat[:, D:]
    feats, acts = [start_feat], []
    with torch.no_grad():
        for _ in range(horizon):
            a = torch.multinomial(ac.policy_logp(feats[-1]).exp(), 1).squeeze(-1)
            h, prior = wm.rssm.img_step(h, z, _onehot(a))
            z = wm.rssm.sample(prior)
            feats.append(torch.cat([h, z], -1))
            acts.append(a)
    return torch.stack(feats), torch.stack(acts)


def actor_critic_loss(model, start_feat, start_cont):
    """Imagine from every replayed state; lambda-returns; normalised advantages; REINFORCE actor; two-hot critic
    regularised toward its slow copy. ``start_cont`` (N,) is 0 for a terminal start state (nothing to imagine)."""
    import torch

    cfg, wm, ac = model.cfg, model.wm, model.ac
    feats, acts = imagine(model, start_feat, cfg.horizon)
    with torch.no_grad():
        Hp1, N, Fd = feats.shape
        flat = feats.view(Hp1 * N, Fd)
        rew = twohot_mean(torch.softmax(wm.reward(flat), -1), model.bins).view(Hp1, N)[1:]
        cont = torch.sigmoid(wm.cont(flat)).view(Hp1, N)[1:]
        values = twohot_mean(torch.softmax(ac.critic(flat), -1), model.bins).view(Hp1, N)
        slow = twohot_mean(torch.softmax(ac.slow(flat), -1), model.bins).view(Hp1, N)[:-1]
        ret = lambda_return(rew, cont, values, cfg.gamma, cfg.lam)
        weight = torch.cumprod(torch.cat([start_cont[None], cont[:-1]], 0), 0)  # (H, N)
        lo, hi = torch.quantile(ret, 0.05), torch.quantile(ret, 0.95)
        d = cfg.ret_norm_decay
        if int(model.updates) == 0:
            model.ret_lo.copy_(lo)
            model.ret_hi.copy_(hi)
        else:
            model.ret_lo.mul_(d).add_((1 - d) * lo)
            model.ret_hi.mul_(d).add_((1 - d) * hi)
        scale = torch.clamp(model.ret_hi - model.ret_lo, min=cfg.ret_norm_limit)
        adv = (ret - values[:-1]) / scale
    f = feats[:-1].reshape(-1, Fd)
    logp_all = ac.policy_logp(f).view(cfg.horizon, N, N_ACTIONS)
    logp = logp_all.gather(-1, acts.unsqueeze(-1)).squeeze(-1)
    ent = -(logp_all.exp() * logp_all).sum(-1)
    actor_loss = -(weight * (logp * adv + cfg.entropy * ent)).mean()
    crit_logits = ac.critic(f).view(cfg.horizon, N, -1)
    critic_loss = (weight * (twohot_nll(crit_logits, ret, model.bins)
                             + cfg.slow_reg * twohot_nll(crit_logits, slow, model.bins))).mean()
    metrics = {"actor_loss": actor_loss.item(), "critic_loss": critic_loss.item(), "entropy": ent.mean().item(),
               "imag_return": ret[0].mean().item(), "imag_reward": rew.mean().item(), "value": values[0].mean().item(),
               "ret_scale": float(scale), "imag_cont": cont.mean().item()}
    return actor_loss, critic_loss, metrics


class Learner:
    """The three optimisers and one training step (world model, then actor and critic in imagination)."""

    def __init__(self, model) -> None:
        import torch

        cfg = model.cfg
        self.model = model
        self.opt_wm = torch.optim.Adam(model.wm.parameters(), lr=cfg.model_lr, eps=1e-8)
        self.opt_actor = torch.optim.Adam(model.ac.actor.parameters(), lr=cfg.actor_lr, eps=1e-5)
        self.opt_critic = torch.optim.Adam(model.ac.critic.parameters(), lr=cfg.critic_lr, eps=1e-5)
        # actor-side imitation (off by default: no optimiser is built, so unguided runs are unchanged)
        self.opt_bc = (torch.optim.Adam(model.ac.actor.parameters(), lr=cfg.actor_bc_lr, eps=1e-5)
                       if getattr(cfg, "actor_bc_lr", 0.0) > 0 else None)

    def step(self, batch: dict, bc_scale: float = 0.0, demo_rows: int = 0) -> dict:
        """One update. With imitation on, ``demo_rows`` (the last rows of ``batch``, as ``sample_mixed`` builds it) and
        ``bc_scale`` > 0 also take one behaviour-cloning step on the actor at ``actor_bc_lr * bc_scale``."""
        import torch

        m, cfg = self.model, self.model.cfg
        b = {k: torch.as_tensor(v) for k, v in batch.items()}
        loss, metrics, feat = world_model_loss(m, b["obs"], b["act"], b["reward"], b["first"], b["term"])
        self.opt_wm.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.wm.parameters(), cfg.model_clip)
        self.opt_wm.step()
        start = feat.detach().reshape(-1, feat.shape[-1])
        start_cont = 1.0 - b["term"].reshape(-1)
        if 0 < cfg.imag_starts < len(start):
            pick = torch.randperm(len(start))[: cfg.imag_starts]
            start, start_cont = start[pick], start_cont[pick]
        actor_loss, critic_loss, ac_metrics = actor_critic_loss(m, start, start_cont)
        self.opt_actor.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(m.ac.actor.parameters(), cfg.ac_clip)
        self.opt_actor.step()
        self.opt_critic.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(m.ac.critic.parameters(), cfg.ac_clip)
        self.opt_critic.step()
        bc_metrics = {}
        if self.opt_bc is not None:
            bc_metrics = {"bc_scale": float(bc_scale), "bc_loss": float("nan"), "bc_acc": float("nan"), "bc_n": 0}
            if bc_scale > 0 and demo_rows > 0:
                d = slice(len(b["act"]) - demo_rows, None)
                bc_loss, bc_acc, bc_n = actor_bc_loss(m, feat[d], b["act"][d], b["first"][d], b["term"][d])
                if bc_n:
                    for g in self.opt_bc.param_groups:
                        g["lr"] = cfg.actor_bc_lr * bc_scale
                    self.opt_bc.zero_grad()
                    bc_loss.backward()
                    torch.nn.utils.clip_grad_norm_(m.ac.actor.parameters(), cfg.ac_clip)
                    self.opt_bc.step()
                bc_metrics.update(bc_loss=bc_loss.item(), bc_acc=bc_acc, bc_n=bc_n)
        with torch.no_grad():
            for ps, pf in zip(m.ac.slow.parameters(), m.ac.critic.parameters()):
                ps.mul_(cfg.slow_critic_decay).add_((1 - cfg.slow_critic_decay) * pf)
        m.updates += 1
        return {**metrics, **ac_metrics, **bc_metrics}


# --------------------------------------------------------------------------- replay


class Replay:
    """One stream per env slot, in arrival order (episodes back to back, marked by ``first``). Sequences are drawn
    uniformly from all stored windows of ``seq_len`` consecutive steps; a window may cross an episode boundary, where
    ``first`` resets the model state. The 0/1 grid channels are bit-packed (351 bytes a step on the ranch)."""

    def __init__(self, n_streams: int, capacity: int, canvas: tuple[int, int], seed: int, n_in: int = N_IN) -> None:
        self.canvas, self.n_in, self.nb = canvas, n_in, n_in - 1
        self.cap = max(1, capacity // n_streams)
        self.nbits = canvas[0] * canvas[1] * self.nb
        nbytes = (self.nbits + 7) // 8
        self.bits = np.zeros((n_streams, self.cap, nbytes), np.uint8)
        self.clock = np.zeros((n_streams, self.cap), np.float32)
        self.act = np.zeros((n_streams, self.cap), np.int64)
        self.reward = np.zeros((n_streams, self.cap), np.float32)
        self.first = np.zeros((n_streams, self.cap), np.float32)
        self.term = np.zeros((n_streams, self.cap), np.float32)
        self.pos = np.zeros(n_streams, np.int64)
        self.count = np.zeros(n_streams, np.int64)
        self.rng = np.random.default_rng([seed, 7])

    def __len__(self) -> int:
        return int(self.count.sum())

    def add(self, k: int, x: np.ndarray, prev_act: int, reward: float, first: bool, term: bool) -> None:
        """``x`` (H, W, N_IN): the policy channels; the clock is stored once (it is the same on every cell)."""
        p = self.pos[k]
        self.bits[k, p] = np.packbits(x[..., :self.nb].astype(np.uint8).ravel())
        self.clock[k, p] = x[0, 0, self.nb]
        self.act[k, p], self.reward[k, p], self.first[k, p], self.term[k, p] = prev_act, reward, first, term
        self.pos[k] = (p + 1) % self.cap
        self.count[k] = min(self.count[k] + 1, self.cap)

    def add_episode(self, k: int, ep: dict) -> None:
        """Append a whole pre-packed episode (from ``collect_demos``) to stream ``k``."""
        for i in range(len(ep["act"])):
            p = self.pos[k]
            self.bits[k, p], self.clock[k, p] = ep["bits"][i], ep["clock"][i]
            self.act[k, p], self.reward[k, p] = ep["act"][i], ep["reward"][i]
            self.first[k, p], self.term[k, p] = ep["first"][i], ep["term"][i]
            self.pos[k] = (p + 1) % self.cap
            self.count[k] = min(self.count[k] + 1, self.cap)

    def sample(self, batch: int, seq_len: int) -> dict:
        windows = np.maximum(self.count - seq_len + 1, 0)
        if windows.sum() == 0:
            raise ValueError("not enough data in the replay buffer yet")
        ks = self.rng.choice(len(windows), size=batch, p=windows / windows.sum())
        idx = np.empty((batch, seq_len), np.int64)
        for i, k in enumerate(ks):
            oldest = (self.pos[k] - self.count[k]) % self.cap
            start = oldest + self.rng.integers(windows[k])
            idx[i] = (start + np.arange(seq_len)) % self.cap
        rows = ks[:, None]
        H, W = self.canvas
        bits = np.unpackbits(self.bits[rows, idx], axis=-1, count=self.nbits)
        obs = np.empty((batch, seq_len, H, W, self.n_in), np.float32)
        obs[..., :self.nb] = bits.reshape(batch, seq_len, H, W, self.nb)
        obs[..., self.nb] = self.clock[rows, idx][..., None, None]
        return {"obs": obs, "act": self.act[rows, idx], "reward": self.reward[rows, idx],
                "first": self.first[rows, idx], "term": self.term[rows, idx]}


# --------------------------------------------------------------------------- envs (keeping the terminal frame)


class _Envs:
    """Env slots stepped in this process, with the same per-slot RNG scheme as the PPO runs (seeded by run seed and
    slot index; tasks drawn uniformly, fire seeds from TRAIN_SEEDS only). Unlike ``rl_baseline._Slots`` it returns
    the terminal frame, which the world model needs to learn the final reward and the episode end."""

    def __init__(self, n: int, seed: int, canvas: tuple[int, int], max_steps: int,
                 channels: Sequence[int] = POLICY_CHANNELS) -> None:
        from emberpath.rl_baseline import _make_env

        self.channels = tuple(channels)
        self.heard: list[dict] = [{} for _ in range(n)]  # per slot, from that episode's own hearing events

        self._make_env = _make_env
        self.n, self.canvas, self.max_steps = n, canvas, max_steps
        self.rngs = [np.random.default_rng([seed, i]) for i in range(n)]
        self.tasks: tuple[Task, ...] = ()
        self.envs: dict[Task, list] = {}
        self.cur: list[Any] = [None] * n
        self.meta: list[dict] = [{} for _ in range(n)]

    def set_tasks(self, tasks: Sequence[Task]) -> None:
        self.tasks = tuple(tasks)

    def reset(self, k: int) -> np.ndarray:
        rng = self.rngs[k]
        task = self.tasks[int(rng.integers(len(self.tasks)))]
        seed = int(TRAIN_SEEDS[int(rng.integers(len(TRAIN_SEEDS)))])
        if task not in self.envs:
            self.envs[task] = [self._make_env(task, self.max_steps) for _ in range(self.n)]
        env = self.envs[task][k]
        obs, _ = env.reset(seed=seed)
        self.cur[k] = env
        self.meta[k] = {"task": task.label, "seed": seed, "comp": {}}
        self.heard[k] = {}
        return frame(pad_to(obs, self.canvas), self.channels, self.heard[k])

    def step(self, k: int, action: int):
        o, r, term, trunc, info = self.cur[k].step(int(action))
        comp = self.meta[k]["comp"]
        for name, v in info["reward_components"].items():
            comp[name] = comp.get(name, 0.0) + v
        done = term or trunc
        fin = None
        if done:
            fin = {"task": self.meta[k]["task"], "seed": self.meta[k]["seed"], "outcome": info["outcome"],
                   "return": info["episode_return"], "steps": self.cur[k].world.t, "components": dict(comp)}
        update_heard(self.heard[k], info["events"])
        return frame(pad_to(o, self.canvas), self.channels, self.heard[k]), float(r), done, fin


# --------------------------------------------------------------------------- planner demos (demo-seeded replay)

DEMO_TEACHERS = ("adaptive",)  # the evidence-disciplined planner agent; never the oracle


def _demo_job(args) -> list[dict]:
    """Play ``teacher`` on (task, training seeds) and record each episode in the replay's own format: the policy
    channels of every frame including the terminal one, the action that led to it, the reward on arriving, and the
    first / terminal flags (exactly what ``train_dreamer`` stores for its own steps). The teacher is built without
    the world and gets only what ``evaluate.run_episode`` hands an agent: the ``Observation`` and the ``Event``s."""
    task, seeds, teacher, canvas, max_steps, *rest = args
    chans = tuple(rest[0]) if rest else POLICY_CHANNELS
    from emberpath.agents import AGENTS
    from emberpath.rl_baseline import _make_env

    cls = AGENTS[teacher]
    if teacher not in DEMO_TEACHERS or getattr(cls, "privileged", False):
        raise ValueError(f"{teacher!r} is not an evidence-disciplined demo teacher (choose from {DEMO_TEACHERS})")
    if set(seeds) - set(TRAIN_SEEDS):
        raise ValueError("demo episodes must come from training seeds only")
    env = _make_env(task, max_steps)
    out = []
    for s in seeds:
        x, _ = env.reset(seed=s)
        agent = cls(env.bp)  # never handed the world
        obs, events, heard = env.world.observe(), [], {}
        frames, acts, rews, firsts, terms = [frame(pad_to(x, canvas), chans, heard)], [0], [0.0], [True], [False]
        while True:
            d = agent.act(obs, events)
            x, r, term, trunc, info = env.step(d.action)
            update_heard(heard, info["events"])
            frames.append(frame(pad_to(x, canvas), chans, heard))
            acts.append(int(d.action))
            rews.append(float(r))
            firsts.append(False)
            terms.append(bool(term or trunc))
            if term or trunc:
                break
            obs, events = env.world.observe(), info["events"]
        f = np.stack(frames)
        nb = f.shape[-1] - 1
        out.append({"bits": np.packbits(f[..., :nb].astype(np.uint8).reshape(len(f), -1), axis=1),
                    "clock": f[:, 0, 0, nb].astype(np.float32), "act": np.array(acts, np.int64),
                    "reward": np.array(rews, np.float32), "first": np.array(firsts, np.float32),
                    "term": np.array(terms, np.float32), "seed": int(s), "task": task.label,
                    "outcome": info["outcome"], "return": float(info["episode_return"]), "teacher": teacher})
    return out


def collect_demos(tasks: Sequence[Task], episodes_per_task: int, teacher: str = "adaptive", seed: int = 0,
                  canvas: tuple[int, int] = (13, 24), max_steps: int = 150, workers: int = 0,
                  channels: Sequence[int] = POLICY_CHANNELS) -> dict:
    """AdaptiveAgent episodes on TRAINING seeds (drawn without replacement per task with ``seed``), packed for the
    demo replay. Returns {"episodes", "teacher", "canvas", "transitions", "seeds"}."""
    if teacher not in DEMO_TEACHERS:
        raise ValueError(f"{teacher!r} is not an evidence-disciplined demo teacher (choose from {DEMO_TEACHERS})")
    rng = np.random.default_rng([seed, 13])
    jobs = []
    for task in tasks:
        picks = [int(TRAIN_SEEDS[i]) for i in rng.choice(len(TRAIN_SEEDS), episodes_per_task, replace=False)]
        jobs += [(task, picks[k:k + 25], teacher, tuple(canvas), max_steps, tuple(channels))
                 for k in range(0, len(picks), 25)]
    if workers > 0:
        import multiprocessing as mp

        with mp.get_context("spawn").Pool(workers) as pool:
            parts = pool.map(_demo_job, jobs)
    else:
        parts = [_demo_job(j) for j in jobs]
    eps = [e for part in parts for e in part]
    seeds = {e["seed"] for e in eps}
    assert seeds <= set(TRAIN_SEEDS) and not seeds & set(EVAL_SEEDS)
    return {"episodes": eps, "teacher": teacher, "canvas": tuple(canvas), "channels": tuple(channels),
            "transitions": int(sum(len(e["act"]) - 1 for e in eps)), "seeds": sorted(seeds)}


def demo_replay(demos: dict, n_streams: int, canvas: tuple[int, int], seed: int,
                channels: Sequence[int] | None = None) -> Replay:
    """A separate replay holding every demo episode (round-robin over streams, sized so nothing is overwritten)."""
    if tuple(demos["canvas"]) != tuple(canvas):
        raise ValueError(f"demos were recorded on a {demos['canvas']} canvas, the model uses {canvas}")
    if demos["teacher"] not in DEMO_TEACHERS:
        raise ValueError(f"demo teacher {demos['teacher']!r} is not allowed")
    chans = tuple(demos.get("channels", POLICY_CHANNELS))
    if channels is not None and chans != tuple(channels):
        raise ValueError(f"demos carry channels {chans}, the model reads {tuple(channels)}")
    eps = demos["episodes"]
    per = [0] * n_streams
    for i, e in enumerate(eps):
        per[i % n_streams] += len(e["act"])
    rep = Replay(n_streams, max(per) * n_streams + n_streams, canvas, seed, len(chans))
    for i, e in enumerate(eps):
        rep.add_episode(i % n_streams, e)
    return rep


def sample_mixed(replay: Replay, demo: Replay | None, batch: int, seq_len: int, frac: float) -> tuple[dict, int]:
    """A batch whose last ``demo_rows(batch, frac)`` sequences come from ``demo``; returns (batch, demo rows)."""
    nd = demo_rows(batch, frac) if demo is not None else 0
    if nd == 0:
        return replay.sample(batch, seq_len), 0
    d = demo.sample(nd, seq_len)
    if nd == batch:
        return d, nd
    o = replay.sample(batch - nd, seq_len)
    return {k: np.concatenate([o[k], d[k]]) for k in o}, nd


# --------------------------------------------------------------------------- training loop


def _log(msg: str) -> None:
    print(msg, flush=True)


def train_dreamer(phases: Sequence[Phase], config: DreamerConfig = DreamerConfig(), out: str | Path | None = None,
                  init: str | Path | None = None, log=_log, checkpoint_every: int = 0,
                  canvas: tuple[int, int] | None = None, log_every: int = 50, max_seconds: float = math.inf,
                  demos: dict | None = None, save_init: str | Path | None = None) -> dict:
    """Train Dreamer over the curriculum ``phases``; returns {"model", "history", "episodes", "steps", ...}.

    Deterministic for a given config on CPU with a fixed thread count. ``init`` continues from a checkpoint (its
    weights; the replay starts empty and is prefilled with the loaded policy rather than random actions).
    ``out``: a directory for ``final.pt``, ``step_<n>.pt`` every ``checkpoint_every`` env steps, ``history.json``.
    ``max_seconds`` stops early (at the end of an env step) so a run fits a wall-clock budget.

    ``demos`` (from ``collect_demos``) with ``config.demo_frac > 0`` turns on the demo-seeded replay: the schedule
    ``demo_fraction`` picks how many sequences of each world-model batch come from the demos; when it first reaches
    0 the model is saved as ``demo_zero.pt``. Demos only shape what is replayed; acting is always the policy's own.
    With ``config.actor_bc_lr > 0`` the demo rows also train the actor by behaviour cloning (``actor_bc_scale``).
    ``save_init``: also save the model before any update (the 0% point of a learning curve); changes nothing else."""
    import torch

    from emberpath.cli import load_blueprint

    torch.manual_seed(config.seed)
    torch.set_num_threads(config.torch_threads)
    blueprints = {t.blueprint: load_blueprint(t.blueprint) for p in phases for t in p.tasks}
    if init:
        model = load_dreamer(init, config)
        canvas = model.canvas
    else:
        canvas = tuple(canvas or canvas_for(list(blueprints.values())))
        model = make_dreamer(canvas, config)
    model.train()
    if save_init:
        Path(save_init).parent.mkdir(parents=True, exist_ok=True)
        save_dreamer(model, save_init, phases)
    learner = Learner(model)
    out = Path(out) if out else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
    n, cfg = config.n_envs, config
    chans = channels_for(cfg)
    replay = Replay(n, cfg.capacity, canvas, cfg.seed, len(chans))
    demo = None
    if cfg.demo_frac > 0:
        if not demos or not demos["episodes"]:
            raise ValueError("demo_frac > 0 needs demos (collect_demos)")
        demo = demo_replay(demos, n, canvas, cfg.seed + 1_000_003, chans)
    elif demos:
        raise ValueError("demos were given but demo_frac is 0")
    if cfg.actor_bc_lr > 0 and demo is None:
        raise ValueError("actor imitation (actor_bc_lr > 0) needs the demo replay (demo_frac > 0 and demos)")
    total_steps = sum(p.steps for p in phases)
    demo_info = {"demo_steps_replayed": 0, "demo_zero": None}
    envs = _Envs(n, cfg.seed, canvas, cfg.max_steps, chans)
    per_update = cfg.batch * cfg.seq_len
    history: list[dict] = []
    episodes: list[dict] = []
    step = updates = 0
    t0 = time.time()
    next_ckpt = checkpoint_every or math.inf
    random_prefill = init is None
    metrics_acc: list[dict] = []
    for pi, phase in enumerate(phases):
        envs.set_tasks(phase.tasks)
        cur = [envs.reset(k) for k in range(n)]
        rew, first, term, prev = [0.0] * n, [True] * n, [False] * n, [0] * n
        h, z = model.wm.rssm.initial(n)
        phase_end = step + phase.steps
        while step < phase_end and time.time() - t0 < max_seconds:
            for k in range(n):
                replay.add(k, cur[k], prev[k], rew[k], first[k], term[k])
            with torch.no_grad():
                x = torch.as_tensor(np.stack(cur))
                emb = model.wm.embed(x)
                h, z, _, _ = model.wm.rssm.obs_step(h, z, _onehot(torch.as_tensor(prev)), emb,
                                                     torch.as_tensor(first, dtype=torch.float32))
                probs = model.ac.policy_logp(torch.cat([h, z], -1)).exp()
                acts = torch.multinomial(probs, 1).squeeze(-1).numpy()
            if random_prefill and step < cfg.prefill:
                acts = np.array([int(envs.rngs[k].integers(N_ACTIONS)) for k in range(n)])
            for k in range(n):
                if term[k]:  # the terminal frame was stored; start the next episode
                    cur[k], rew[k], first[k], term[k], prev[k] = envs.reset(k), 0.0, True, False, 0
                    continue
                cur[k], rew[k], term[k], fin = envs.step(k, int(acts[k]))
                first[k], prev[k] = False, int(acts[k])
                step += 1
                if fin:
                    fin.update(phase=pi, step=step)
                    episodes.append(fin)
            if step >= cfg.prefill and len(replay) >= per_update:
                due = int((step - cfg.prefill) * cfg.train_ratio / per_update) + 1
                while updates < due:
                    frac = demo_fraction(step, cfg, total_steps)
                    batch, nd = sample_mixed(replay, demo, cfg.batch, cfg.seq_len, frac)
                    if cfg.actor_bc_lr > 0:
                        bcs = actor_bc_scale(step, cfg, total_steps)
                        metrics_acc.append({**learner.step(batch, bcs, nd), "demo_frac": frac, "demo_rows": nd})
                    else:
                        metrics_acc.append({**learner.step(batch), "demo_frac": frac, "demo_rows": nd})
                    demo_info["demo_steps_replayed"] += nd * cfg.seq_len
                    updates += 1
                    if demo is not None and nd == 0 and demo_info["demo_zero"] is None:
                        demo_info["demo_zero"] = {"step": step, "updates": updates - 1,
                                                  "time": round(time.time() - t0, 1)}
                        if out:
                            save_dreamer(model, out / "demo_zero.pt", phases)
                    if updates % log_every == 0:
                        recent = [e for e in episodes[-300:] if e["phase"] == pi][-100:]
                        row = {"step": step, "updates": updates, "phase": pi, "time": round(time.time() - t0, 1),
                               **{k: float(np.nanmean([m[k] for m in metrics_acc])) for k in metrics_acc[0]},
                               "train_success": float(np.mean([e["outcome"] == "success" for e in recent]))
                               if recent else None,
                               "train_return": float(np.mean([e["return"] for e in recent])) if recent else None}
                        metrics_acc = []
                        history.append(row)
                        if log:
                            log(f"phase {pi} step {step:>7} upd {updates:>5}  succ {row['train_success'] or 0:.2f}  "
                                f"ret {row['train_return'] or 0:6.2f}  recon {row['recon']:6.1f}  "
                                f"rew_nll {row['reward_nll']:.2f}  kl {row['kl_dyn']:.2f}  "
                                f"imag_ret {row['imag_return']:6.2f}  ent {row['entropy']:.2f}  demo {row['demo_frac']:.2f}  "
                                f"bigerr {row['reward_abs_err_big']:.2f}  cell {row['agent_cell_acc']:.2f}"
                                + (f"  bc {row['bc_scale']:.2f} acc {row['bc_acc']:.2f}" if "bc_scale" in row else "")
                                + f"  ({row['time']:.0f}s)")
            if out and step >= next_ckpt:
                save_dreamer(model, out / f"step_{step}.pt", phases)
                next_ckpt += checkpoint_every
    touched = {e["seed"] for e in episodes}
    assert not touched & set(EVAL_SEEDS), "a training episode used an evaluation seed"
    model.eval()
    result = {"model": model, "history": history, "episodes": episodes, "canvas": canvas,
              "wall_clock": time.time() - t0, "steps": step, "updates": updates}
    if demo is not None:
        demo_info.update(teacher=demos["teacher"], demo_episodes=len(demos["episodes"]),
                         demo_transitions=demos["transitions"], demo_seeds=demos["seeds"],
                         schedule={"start": cfg.demo_frac, "until_step": cfg.demo_until or int(0.6 * total_steps),
                                   "planned_steps": total_steps, "shape": "linear"})
        if cfg.actor_bc_lr > 0:
            demo_info["actor_bc"] = {"lr": cfg.actor_bc_lr, "planned_steps": total_steps, "shape": "linear lr to 0",
                                     "until_step": cfg.actor_bc_until or cfg.demo_until or int(0.6 * total_steps)}
        assert not set(demos["seeds"]) & set(EVAL_SEEDS), "a demo episode used an evaluation seed"
        result["demo"] = demo_info
    if out:
        save_dreamer(model, out / "final.pt", phases)
        (out / "history.json").write_text(json.dumps(
            {"config": asdict(config), "phases": _phases_json(phases), "wall_clock": result["wall_clock"],
             "steps": step, "updates": updates, "history": history, "episodes": episodes,
             **({"demo": demo_info} if demo is not None else {})}))
    return result


def _phases_json(phases: Sequence[Phase]) -> list:
    return [{"steps": p.steps, "tasks": [asdict(t) for t in p.tasks]} for p in phases]


def save_dreamer(model, path: str | Path, phases: Sequence[Phase] = ()) -> None:
    import torch

    torch.save({"state_dict": model.state_dict(), "canvas": list(model.canvas), "config": asdict(model.cfg),
                "phases": _phases_json(phases), "channels": list(channels_for(model.cfg))}, path)


def load_dreamer(path: str | Path, config: DreamerConfig | None = None):
    """A model from a checkpoint written by ``train_dreamer``. ``config`` may change training settings (rates,
    ratio, seed) but not the network sizes, which always come from the checkpoint."""
    import torch

    meta = torch.load(path, map_location="cpu", weights_only=False)
    saved = DreamerConfig(**meta["config"])
    if tuple(meta["channels"]) != channels_for(saved):
        raise ValueError(f"{path}: trained on channels {meta['channels']}, this code feeds {channels_for(saved)}")
    widen = False
    if config is not None:
        sizes = ("deter", "stoch", "classes", "hidden", "cnn", "bins", "bin_range")
        if any(getattr(config, s) != getattr(saved, s) for s in sizes):
            raise ValueError(f"{path}: network sizes differ from the requested config")
        if config.heard != saved.heard:
            if saved.heard:
                raise ValueError(f"{path}: reads the hearing plane; the requested config does not")
            widen = True
        saved = config
    model = make_dreamer(tuple(meta["canvas"]), saved)
    state = meta["state_dict"]
    if widen:
        state = _widen_state(state, model.state_dict(), meta["channels"], channels_for(saved))
    model.load_state_dict(state)
    model.eval()
    return model


def _widen_state(old: dict, new: dict, old_ch: Sequence[int], new_ch: Sequence[int]) -> dict:
    """Start a model with extra input planes from one without them: every weight carries over; the encoder's
    weights on a new plane start at zero (so it initially behaves exactly as before) and the decoder keeps its fresh
    initialisation for the new plane."""
    out = dict(old)
    pos = [list(new_ch).index(c) for c in old_ch]
    enc, dec_w, dec_b = "wm.encoder.0.weight", "wm.dec.2.weight", "wm.dec.2.bias"
    w = new[enc].clone().zero_()
    w[:, pos] = old[enc]
    out[enc] = w
    for k in (dec_w, dec_b):
        t = new[k].clone()
        t[pos] = old[k]
        out[k] = t
    return out


# --------------------------------------------------------------------------- the agent


class DreamerAgent:
    """A trained Dreamer behind the ``Agent`` interface. Its input is rebuilt from the observation and its own
    events (exactly as ``PPOAgent``), its recurrent state carries over from tick to tick, and each move is explained
    with the policy's confidence in it.

    Latents are always sampled (as in DreamerV3; the posterior's single most likely latent turned out to be nearly
    constant across a whole episode, see docs/DREAMER.md), from a generator seeded by ``seed``, so a run is
    reproducible. ``greedy`` takes the most likely action; otherwise the action is sampled too."""

    def __init__(self, model, blueprint: Blueprint, greedy: bool = True, max_steps: int = 150, seed: int = 0):
        import torch

        self.model, self.bp, self.greedy, self.max_steps = model, blueprint, greedy, max_steps
        self.canvas = model.canvas
        self.name = "dreamer" if greedy else "dreamer_sampled"
        self.gen = torch.Generator().manual_seed(seed)
        self.h, self.z = model.wm.rssm.initial(1)
        self.prev, self.first = 0, True
        self.bumped: set = set()
        self.heard: dict = {}  # hiding places where its own hearing events last said someone is calling
        self.channels = channels_for(model.cfg)

    def features(self, obs: Observation, events: list[Event]) -> np.ndarray:
        for e in events:
            if e.outcome and parse_key(e.key)[0] == "door_jammed":
                self.bumped.add(key_cell(e.key))
        update_heard(self.heard, events)
        x = pad_to(encode_observation(self.bp, obs, self.bumped, self.max_steps), self.canvas)
        return frame(x, self.channels, self.heard)

    def act(self, obs: Observation, events: list[Event]) -> Decision:
        import torch

        x = torch.as_tensor(self.features(obs, events)[None])
        m = self.model
        with torch.no_grad():
            emb = m.wm.embed(x)
            self.h, self.z, _, _ = m.wm.rssm.obs_step(
                self.h, self.z, _onehot(torch.tensor([self.prev])), emb, torch.tensor([float(self.first)]),
                sample=True, gen=self.gen)
            p = m.ac.policy_logp(torch.cat([self.h, self.z], -1))[0].exp()
        if self.greedy:
            a = int(p.argmax())
        else:
            a = int(torch.multinomial(p, 1, generator=self.gen))
        self.prev, self.first = a, False
        how = "" if self.greedy else ", sampled"
        return Decision(a, f"World model: {ACTION_WORDS[a]} ({float(p[a]):.0%} confident{how}).")


def dreamer_factory(model, greedy: bool = True, max_steps: int = 150, seed: int = 0):
    """An ``Agent`` factory for ``evaluate.run_episode``: the world is never handed to the agent."""

    def make(bp: Blueprint, _world) -> DreamerAgent:
        return DreamerAgent(model, bp, greedy=greedy, max_steps=max_steps, seed=seed)

    return make


def evaluate_dreamer(model, blueprint: Blueprint, household, novelty: str | None,
                     seeds: Sequence[int] = EVAL_SEEDS, greedy: bool = True, max_steps: int = 150,
                     sample_seed: int = 0) -> list:
    """Run the agent through the shared harness on held-out seeds only (refuses training seeds)."""
    from emberpath.evaluate import run_episode

    seeds = list(seeds)
    leaked = set(seeds) & set(TRAIN_SEEDS)
    if leaked:
        raise ValueError(f"evaluate_dreamer uses held-out seeds only; got training seeds {sorted(leaked)[:5]}...")
    return [run_episode(blueprint, household, dreamer_factory(model, greedy, max_steps, sample_seed * 100_003 + s),
                        novelty, s, max_steps) for s in seeds]


# --------------------------------------------------------------------------- command line


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="train_dreamer", description="Train DreamerV3 (docs/DREAMER.md).")
    p.add_argument("--blueprint", default="ranch")
    p.add_argument("--profiles", nargs="+", default=["single", "kids"])
    p.add_argument("--novelties", nargs="+", default=["none", "door_jam", "carpet_flashover", "hidden_dependent"],
                   help="'none' for the nominal world; each episode samples one")
    p.add_argument("--steps", type=int, default=200_000, help="env steps")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs/dreamer", help="checkpoint directory (gitignored)")
    p.add_argument("--init", default=None, help="continue from this checkpoint")
    p.add_argument("--canvas", default="ranch", help="pad every house to this blueprint's size (so checkpoints "
                                                       "carry over between houses)")
    p.add_argument("--envs", type=int, default=16)
    p.add_argument("--train-ratio", type=float, default=DreamerConfig.train_ratio)
    p.add_argument("--prefill", type=int, default=DreamerConfig.prefill)
    p.add_argument("--batch", type=int, default=DreamerConfig.batch)
    p.add_argument("--seq-len", type=int, default=DreamerConfig.seq_len)
    p.add_argument("--actor-lr", type=float, default=DreamerConfig.actor_lr)
    p.add_argument("--critic-lr", type=float, default=DreamerConfig.critic_lr)
    p.add_argument("--model-lr", type=float, default=DreamerConfig.model_lr)
    p.add_argument("--gamma", type=float, default=DreamerConfig.gamma)
    p.add_argument("--entropy", type=float, default=DreamerConfig.entropy)
    p.add_argument("--imag-starts", type=int, default=DreamerConfig.imag_starts)
    p.add_argument("--threads", type=int, default=4, help="torch threads (fix it: results depend on it)")
    p.add_argument("--checkpoint-every", type=int, default=25_000)
    p.add_argument("--max-minutes", type=float, default=math.inf, help="stop early after this wall clock")
    p.add_argument("--demo-frac", type=float, default=0.0,
                   help="share of world-model batches from AdaptiveAgent demos at the start (0 = unguided)")
    p.add_argument("--demo-until", type=int, default=0,
                   help="env step at which the demo share reaches exactly 0 (0 = 60%% of --steps)")
    p.add_argument("--demo-episodes", type=int, default=100, help="demo episodes per training setting")
    p.add_argument("--demo-workers", type=int, default=0, help="processes for collecting demos")
    p.add_argument("--demo-data", default=None, help="cache file for the demos (reused if it exists)")
    p.add_argument("--heard", action="store_true", help="add the 'someone is calling here' plane to the input")
    p.add_argument("--actor-bc-lr", type=float, default=0.0,
                   help="actor-side imitation of the demos: behaviour-cloning learning rate at the start, annealed "
                        "linearly to 0 (0 = off; needs --demo-frac > 0)")
    p.add_argument("--actor-bc-until", type=int, default=0,
                   help="env step at which the imitation rate reaches exactly 0 (0 = where the demo share ends)")
    p.add_argument("--free-nats", type=float, default=DreamerConfig.free_nats)
    p.add_argument("--dyn-scale", type=float, default=DreamerConfig.dyn_scale)
    p.add_argument("--rep-scale", type=float, default=DreamerConfig.rep_scale)
    p.add_argument("--save-init", default=None, help="also save the model before any update to this path")
    a = p.parse_args(argv)
    from emberpath.cli import load_blueprint
    from emberpath.dependents import HOUSEHOLDS
    from emberpath.scenarios import applicable

    bp = load_blueprint(a.blueprint)
    for prof in a.profiles:
        for nov in a.novelties:
            if not applicable(bp, HOUSEHOLDS[prof], None if nov == "none" else nov):
                raise SystemExit(f"{a.blueprint}: {prof} / {nov} does not apply")
    cfg = DreamerConfig(n_envs=a.envs, seed=a.seed, torch_threads=a.threads, train_ratio=a.train_ratio,
                        prefill=a.prefill, batch=a.batch, seq_len=a.seq_len, actor_lr=a.actor_lr,
                        critic_lr=a.critic_lr, model_lr=a.model_lr, gamma=a.gamma, entropy=a.entropy,
                        imag_starts=a.imag_starts, demo_frac=a.demo_frac, demo_until=a.demo_until, heard=a.heard,
                        actor_bc_lr=a.actor_bc_lr, actor_bc_until=a.actor_bc_until, free_nats=a.free_nats,
                        dyn_scale=a.dyn_scale, rep_scale=a.rep_scale)
    phases = default_phases(a.blueprint, a.profiles, a.novelties, a.steps, 0)
    canvas = canvas_for([bp, load_blueprint(a.canvas)])
    demos = None
    if a.demo_frac > 0:
        import pickle

        cache = Path(a.demo_data) if a.demo_data else None
        if cache and cache.exists():
            demos = pickle.loads(cache.read_bytes())
        else:
            t0 = time.time()
            tasks = [t for ph in phases for t in ph.tasks]
            demos = collect_demos(tasks, a.demo_episodes, "adaptive", a.seed, canvas, cfg.max_steps, a.demo_workers,
                                  channels_for(cfg))
            demos["collect_seconds"] = round(time.time() - t0, 1)
            if cache:
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_bytes(pickle.dumps(demos))
        oc = {}
        for e in demos["episodes"]:
            oc[e["outcome"]] = oc.get(e["outcome"], 0) + 1
        print(f"demos: {len(demos['episodes'])} adaptive episodes, {demos['transitions']} transitions, outcomes "
              f"{oc}, collected in {demos.get('collect_seconds', '?')}s", flush=True)
    res = train_dreamer(phases, cfg, out=a.out, init=a.init, checkpoint_every=a.checkpoint_every, canvas=canvas,
                        max_seconds=a.max_minutes * 60, demos=demos, save_init=a.save_init)
    if "demo" in res:
        print(f"demo share hit 0 at {res['demo']['demo_zero']}; {res['demo']['demo_steps_replayed']} demo steps "
              f"replayed", flush=True)
    print(f"done: {res['steps']} env steps, {res['updates']} updates, {len(res['episodes'])} episodes, "
          f"{res['wall_clock']:.0f}s -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
