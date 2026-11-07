"""Acceptance tests for features that are NOT BUILT YET (Phases 4-6).

They are marked xfail(raises=NotImplementedError): they pass-as-expected-failure today. When you
implement a feature, its test should start passing (XPASS). Then delete the xfail marker so
it guards against regressions. If it fails with an AssertionError, the implementation is wrong.
"""

import numpy as np
import pytest

from emberpath.dependents import HOUSEHOLDS
from emberpath.evaluate import evaluate, gap_closed, summarize
from emberpath.novelty import BetaBernoulliDetector
from emberpath.repair import RepairSearch
from emberpath.rules import RulePatch, door_key, nominal_rules
from emberpath.world import Event

pending = pytest.mark.xfail(raises=NotImplementedError, strict=False, reason="feature not built yet")
pytestmark = pytest.mark.acceptance


def bernoulli_events(key, p, n, seed=0, start=0):
    rng = np.random.default_rng(seed)
    return [Event(start + i, key, bool(rng.random() < p)) for i in range(n)]


@pending
def test_detector_flags_a_jammed_door():
    det = BetaBernoulliDetector(nominal_rules())
    det.observe([Event(t, door_key((2, 7)), True) for t in range(6)])
    assert door_key((2, 7)) in {f.key for f in det.findings()}


@pending
def test_detector_flags_a_faster_carpet_fire_and_estimates_it():
    det = BetaBernoulliDetector(nominal_rules())
    det.observe(bernoulli_events("spread_prob:carpet", 0.60, 80, seed=1))
    (f,) = [f for f in det.findings() if f.key == "spread_prob:carpet"]
    assert abs(f.posterior_mean - 0.60) < 0.15
    assert f.interval90[0] < f.posterior_mean < f.interval90[1]


@pytest.mark.slow
@pending
def test_detector_has_few_false_alarms_when_nothing_changed():
    """Peeking after every batch must not inflate false alarms: <= 5% of 40 quiet worlds."""
    nominal = nominal_rules()
    false_alarms = 0
    for seed in range(40):
        det = BetaBernoulliDetector(nominal)
        for step in range(40):  # observe in small batches, like a real episode
            for key in ("spread_prob:carpet", "spread_prob:wood", "follow_prob:dog"):
                det.observe(bernoulli_events(key, nominal.prob(key), 2, seed=seed * 1000 + step, start=step))
        false_alarms += bool(det.findings())
    assert false_alarms <= 2


@pending
def test_repair_proposes_the_jam_patch():
    events = [Event(t, door_key((2, 7)), True) for t in range(6)]
    det = BetaBernoulliDetector(nominal_rules())
    det.observe(events)
    result = RepairSearch().repair(nominal_rules(), det.findings(), events)
    assert RulePatch(door_key((2, 7)), True) in result.patches


@pending
def test_repair_prefers_the_smallest_explanation():
    """Evidence about one rule should produce one patch, not a pile of them."""
    events = bernoulli_events("spread_prob:carpet", 0.6, 80, seed=4)
    det = BetaBernoulliDetector(nominal_rules())
    det.observe(events)
    result = RepairSearch().repair(nominal_rules(), det.findings(), events)
    assert [p.key for p in result.patches] == ["spread_prob:carpet"]


@pytest.mark.slow
@pending
def test_adaptive_agent_closes_most_of_the_gap_on_door_jam(ranch):
    res = evaluate(ranch, HOUSEHOLDS["single"], ["baseline", "oracle", "adaptive"], "door_jam", range(20))
    assert gap_closed(summarize(res)) >= 0.5


@pending
def test_adaptive_agent_does_not_hurt_when_nothing_changes(ranch):
    res = evaluate(ranch, HOUSEHOLDS["single"], ["baseline", "adaptive"], None, range(20))
    s = summarize(res)
    assert s["adaptive"]["mean_score"] >= s["baseline"]["mean_score"] - 0.05
