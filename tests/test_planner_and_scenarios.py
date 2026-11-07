import pytest

from emberpath.dependents import HOUSEHOLDS
from emberpath.evaluate import evaluate, gap_closed, run_episode, summarize
from emberpath.rules import door_cell
from emberpath.scenarios import NOVELTIES, door_jam


def mean_score(results, agent):
    rows = [r.score for r in results if r.agent == agent]
    return sum(rows) / len(rows)


def test_baseline_is_competent_without_novelty(studio):
    res = evaluate(studio, HOUSEHOLDS["married"], ["baseline"], None, range(10))
    assert summarize(res)["baseline"]["success_rate"] >= 0.9


def test_oracle_equals_baseline_when_nothing_changes(ranch):
    """Control experiment: with correct beliefs the two agents must behave identically."""
    res = evaluate(ranch, HOUSEHOLDS["single"], ["baseline", "oracle"], None, range(6))
    base = [(r.outcome, r.steps) for r in res if r.agent == "baseline"]
    orac = [(r.outcome, r.steps) for r in res if r.agent == "oracle"]
    assert base == orac


def test_every_decision_has_a_reason(studio):
    seen = []
    run_episode(studio, HOUSEHOLDS["single"], "baseline", None, 0, record=lambda w, d: seen.append(d))
    decisions = [d for d in seen if d is not None]
    assert decisions and all(d.reason for d in decisions)


def test_door_jam_wrecks_baseline_but_not_oracle(ranch):
    res = evaluate(ranch, HOUSEHOLDS["single"], ["baseline", "oracle"], "door_jam", range(8))
    assert mean_score(res, "oracle") - mean_score(res, "baseline") > 0.5


def test_carpet_flashover_has_an_oracle_gap_for_kids(ranch):
    res = evaluate(ranch, HOUSEHOLDS["kids"], ["baseline", "oracle"], "carpet_flashover", range(20))
    assert mean_score(res, "oracle") > mean_score(res, "baseline")


@pytest.mark.parametrize("bp_name", ["studio", "ranch"])
def test_door_jam_keeps_house_solvable(bp_name, request):
    bp = request.getfixturevalue(bp_name)
    hh = HOUSEHOLDS["single"]
    (spec,) = door_jam(bp, hh)
    bp.validate(1, frozenset({door_cell(spec.patch.key)}))


def test_all_novelties_build_for_all_households(ranch):
    for build in NOVELTIES.values():
        for hh in HOUSEHOLDS.values():
            assert build(ranch, hh)


def test_gap_closed_formula():
    summ = {k: {"mean_score": v} for k, v in {"baseline": 0.2, "oracle": 0.8, "adaptive": 0.5}.items()}
    assert gap_closed(summ) == pytest.approx(0.5)
    assert gap_closed({"baseline": {"mean_score": 0.5}, "oracle": {"mean_score": 0.5}, "adaptive": {"mean_score": 0.5}}) is None
