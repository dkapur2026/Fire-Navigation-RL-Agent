"""Phase 4/5: notice when the world stops behaving the way the agent believes.

STATUS: interface only. Build this next (see docs/ROADMAP.md, Phases 4-5).

Idea in one sentence: every rule is a coin (``Event.key`` + ``Event.outcome`` are flips of that
coin). The agent believes the coin has bias p. Keep a Beta posterior over each coin's true
bias, and raise a flag when the recent flips are far less likely under the belief than under
what the data says.

Suggested design
----------------
* Per key, keep counts (successes, trials). Prior: Beta centered on the believed p with a
  tunable ``prior_strength`` (pseudo-counts); doors use a near-deterministic prior.
* Score each key with a log Bayes factor:  log P(data | free p ~ Beta(1,1)) - log P(data | p = belief).
  Flag when it exceeds ``threshold`` AND there are at least ``min_trials`` flips.
* Report calibrated uncertainty (posterior mean + 90% interval), not just a boolean.
* Two false-alarm controls: a minimum trial count, and a sequential test (so peeking every tick
  does not inflate the false-positive rate). Measure false alarms on the no-novelty scenario.
* Pyro/NumPyro is optional here. Closed-form Beta-Bernoulli is exact and fast. Use PPLs for the
  stretch goals (hierarchical priors across keys, change-point detection on *when* it changed).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from emberpath.rules import Rules
from emberpath.world import Event


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


class BetaBernoulliDetector:
    """TODO(Phase 4): implement per the module docstring."""

    def __init__(self, belief: Rules, prior_strength: float = 20.0, threshold: float = 6.0, min_trials: int = 5):
        self.belief = belief
        self.prior_strength = prior_strength
        self.threshold = threshold
        self.min_trials = min_trials

    def observe(self, events: list[Event]) -> None:
        raise NotImplementedError("Phase 4: BetaBernoulliDetector.observe")

    def findings(self) -> list[Finding]:
        raise NotImplementedError("Phase 4: BetaBernoulliDetector.findings")

    def reset_belief(self, belief: Rules) -> None:
        raise NotImplementedError("Phase 4: BetaBernoulliDetector.reset_belief")
