import pytest

from emberpath.blueprint import Blueprint
from emberpath.rules import RulePatch, Rules, door_key, nominal_rules, parse_key


def test_shipped_blueprints_parse_and_are_solvable(studio, ranch):
    studio.validate(2)
    ranch.validate(3)
    assert len(ranch.exits) == 2 and len(studio.slots) == 2


def test_rows_must_be_rectangular():
    with pytest.raises(ValueError, match="row 1"):
        Blueprint.parse("####\n###\n")


def test_needs_exactly_one_agent_and_an_exit():
    with pytest.raises(ValueError, match="'A'"):
        Blueprint.parse("#####\n#.E.#\n#####")
    with pytest.raises(ValueError, match="exit"):
        Blueprint.parse("#####\n#A..#\n#####")


def test_validate_catches_unreachable_exit():
    bp = Blueprint.parse("#####\n#A#E#\n#####")
    with pytest.raises(ValueError, match="no exit reachable"):
        bp.validate()


def test_patch_is_immutable_and_roundtrips():
    base = nominal_rules()
    patched = base.with_patch(RulePatch("spread_prob:carpet", 0.9))
    assert base.spread_prob["carpet"] == 0.16  # untouched
    assert patched.spread_prob["carpet"] == 0.9
    assert base.diff(patched) == [RulePatch("spread_prob:carpet", 0.9)]


def test_door_patch_and_prob():
    r = Rules().with_patch(RulePatch(door_key((2, 7)), True))
    assert r.prob(door_key((2, 7))) == 1.0
    assert r.prob(door_key((3, 7))) == 0.0
    assert not r.with_patch(RulePatch(door_key((2, 7)), False)).jammed_doors


@pytest.mark.parametrize("bad", ["spread_prob:lava", "nonsense", "follow_prob:cat", "door_jammed:x,y"])
def test_bad_keys_rejected(bad):
    with pytest.raises(ValueError):
        parse_key(bad)


def test_probabilities_must_be_in_range():
    with pytest.raises(ValueError):
        RulePatch("spread_prob:wood", 1.5)
