"""Phase 4: notice when the world stops behaving the way the agent believes.

Every rule is a coin (``Event.key`` + ``Event.outcome`` are flips of that coin). The agent believes
the coin has bias ``p0 = belief.prob(key)``. Per key we keep (successes, failures) and score

    log BF = log P(data | p ~ Beta(a, a)) - log P(data | p = p0)

and flag the key when the score exceeds ``threshold``.

Design choices (why, not what):

* **Peeking every tick is safe without a growing threshold.** A Bayes factor against a point null,
  with an alternative prior fixed in advance, is a nonnegative martingale under the null (a mixture
  sequential probability ratio test). Ville's inequality then gives P(score ever >= threshold) <=
  exp(-threshold) per key, however often we look. At the default 6.0 that is 0.25% per stochastic
  key, about 1.7% for all seven of them by a union bound. ``min_trials`` is an extra guard for
  stochastic keys only. Forgetting (``decay < 1``) breaks the martingale argument; it is off by
  default and trades false alarms for a shorter delay on late changes (see scripts/false_alarms.py).
* **One near-deterministic path for every boolean family** (``rules.BOOLEAN_FAMILIES``: jammed doors,
  hidden occupants). Their alternative prior is a U-shaped ``Beta(boolean_prior, boolean_prior)``:
  "the flag is either on or off". Under the null, outcomes that agree with the 0/1 belief have
  likelihood 1, and a contrary outcome has likelihood ``noise`` (a sensor glitch). So one contrary
  outcome scores log(0.5 / noise) = 8.5 > 6 and flags at once, while agreeing outcomes can only lower
  the score (marginal likelihood <= 1): a stream of agreeing events never flags. ``min_trials`` does
  not apply to them, because a contradiction of a deterministic rule is decisive by itself; the
  trial floor exists to stop small samples of a *noisy* rule from looking surprising.
* **Stochastic keys** use ``Beta(1, 1)``; the same prior gives the posterior for the reported mean and
  90% interval, so a flagged finding describes "where the rule is, given that it changed".
* **Cost.** Scores are cached per key and recomputed (two ``lgamma`` calls) only for keys that got new
  events. The 1001-point grid posterior is computed lazily, only for flagged keys, and cached until
  that key sees new evidence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from emberpath.rules import BOOLEAN_FAMILIES, Rules, parse_key
from emberpath.world import Event

_GRID = (np.arange(1001) + 0.5) / 1001  # open-interval midpoints: U-shaped priors are infinite at 0 and 1
_LOG_GRID, _LOG_1M_GRID = np.log(_GRID), np.log1p(-_GRID)


@dataclass(frozen=True)
class Finding:
    """One suspected rule change."""

    key: str
    believed: float
    posterior_mean: float
    interval90: tuple[float, float]
    n_trials: int
    log_bayes_factor: float
    first_flagged_step: int


class NoveltyDetector(Protocol):
    def observe(self, events: list[Event]) -> None: ...
    def findings(self) -> list[Finding]: ...
    def reset_belief(self, belief: Rules) -> None:
        """Called after a repair so the detector measures surprise against the new belief."""
        ...


def _log_beta(a: float, b: float) -> float:
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


@dataclass
class _KeyStats:
    prior: float  # a in Beta(a, a)
    boolean: bool
    believed: float
    s: float = 0.0  # (possibly decayed) successes
    f: float = 0.0  # (possibly decayed) failures
    n: int = 0  # raw trial count
    last_step: int = 0
    score: float = 0.0
    flagged_at: int | None = None
    finding: Finding | None = None  # lazily built, dropped when the counts change


class BetaBernoulliDetector:
    """Per-rule Beta-Bernoulli mixture-SPRT detector (see module docstring)."""

    def __init__(
        self,
        belief: Rules,
        threshold: float = 6.0,
        min_trials: int = 5,
        decay: float = 1.0,
        boolean_prior: float = 0.1,
        noise: float = 1e-4,
    ):
        if not 0.0 < decay <= 1.0:
            raise ValueError("decay must be in (0, 1]")
        self.belief = belief
        self.threshold = threshold
        self.min_trials = min_trials
        self.decay = decay
        self.boolean_prior = boolean_prior
        self.noise = noise
        self._stats: dict[str, _KeyStats] = {}

    def observe(self, events: list[Event]) -> None:
        touched: dict[str, _KeyStats] = {}
        for e in events:
            st = self._stats.get(e.key)
            if st is None:
                boolean = parse_key(e.key)[0] in BOOLEAN_FAMILIES
                st = self._stats[e.key] = _KeyStats(
                    self.boolean_prior if boolean else 1.0, boolean, self.belief.prob(e.key), last_step=e.step
                )
            if self.decay < 1.0 and e.step > st.last_step:
                w = self.decay ** (e.step - st.last_step)
                st.s *= w
                st.f *= w
            st.last_step = max(st.last_step, e.step)
            if e.outcome:
                st.s += 1
            else:
                st.f += 1
            st.n += 1
            touched[e.key] = st
        for st in touched.values():
            self._rescore(st)

    def _rescore(self, st: _KeyStats) -> None:
        a, p0 = st.prior, st.believed
        log_alt = _log_beta(a + st.s, a + st.f) - _log_beta(a, a)
        log_null = st.s * math.log(max(p0, self.noise)) + st.f * math.log(max(1.0 - p0, self.noise))
        st.score = log_alt - log_null
        st.finding = None
        flagged = st.score > self.threshold and (st.boolean or st.n >= self.min_trials)
        if not flagged:
            st.flagged_at = None
        elif st.flagged_at is None:
            st.flagged_at = st.last_step

    def findings(self) -> list[Finding]:
        out = []
        for key, st in self._stats.items():
            if st.flagged_at is None:
                continue
            if st.finding is None:
                st.finding = self._finding(key, st)
            out.append(st.finding)
        return sorted(out, key=lambda f: -f.log_bayes_factor)

    def _finding(self, key: str, st: _KeyStats) -> Finding:
        a, b = st.prior + st.s, st.prior + st.f
        log_post = (a - 1) * _LOG_GRID + (b - 1) * _LOG_1M_GRID
        cdf = np.cumsum(np.exp(log_post - log_post.max()))
        cdf /= cdf[-1]
        lo, hi = _GRID[np.searchsorted(cdf, 0.05)], _GRID[min(np.searchsorted(cdf, 0.95), len(_GRID) - 1)]
        return Finding(key, st.believed, a / (a + b), (float(lo), float(hi)), st.n, st.score, st.flagged_at)

    def reset_belief(self, belief: Rules) -> None:
        """Keep the evidence, change the null: surprise is now measured against the repaired belief."""
        self.belief = belief
        for key, st in self._stats.items():
            st.believed = belief.prob(key)
            st.flagged_at = None
            self._rescore(st)
