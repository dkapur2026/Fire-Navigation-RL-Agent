import numpy as np

from emberpath.dependents import HOUSEHOLDS
from emberpath.grid import MOVES, WAIT
from emberpath.rules import RulePatch, Rules, door_key, parse_key
from emberpath.world import FOLLOWING, RESCUED, NoveltySpec, World

RIGHT = MOVES.index((0, 1))


def make(bp, profile="single", **kw):
    return World(bp, HOUSEHOLDS[profile], **kw)


def test_same_seed_same_fire(studio):
    a, b = make(studio, seed=3), make(studio, seed=3)
    for _ in range(15):
        a.step(WAIT), b.step(WAIT)
        assert np.array_equal(a.burn, b.burn)


def test_fire_is_independent_of_what_the_agent_does(ranch):
    """Fire has its own random stream, so every agent faces the same fire for a seed."""
    a, b = make(ranch, seed=5, max_steps=40), make(ranch, seed=5, max_steps=40)
    for i in range(25):
        a.step(WAIT)
        b.step(MOVES.index((0, 1)) if i % 2 else MOVES.index((1, 0)))
        assert np.array_equal(a.burn, b.burn)


def test_fire_never_enters_walls_or_exits(ranch):
    w = make(ranch, seed=1, max_steps=200)
    for _ in range(120):
        if w.done:
            break
        w.step(WAIT)
    for cell in list(ranch.walls) + list(ranch.exits):
        assert w.burn[cell] == 0


def test_events_use_valid_rule_keys(studio):
    w = make(studio, seed=0)
    seen = []
    for _ in range(10):
        seen += w.step(WAIT).events
    assert seen, "fire should have produced rule trials"
    for e in seen:
        parse_key(e.key)


def test_jammed_door_blocks_and_is_logged(studio):
    jam = Rules(jammed_doors=frozenset({(2, 7), (3, 7)}))
    w = make(studio, rules=jam)
    w.agent_pos = (2, 6)
    res = w.step(RIGHT)
    assert w.agent_pos == (2, 6)
    assert any(e.key == door_key((2, 7)) and e.outcome for e in res.events)


def test_open_door_lets_agent_through_and_is_logged(studio):
    w = make(studio)
    w.agent_pos = (2, 6)
    res = w.step(RIGHT)
    assert w.agent_pos == (2, 7)
    assert any(e.key == door_key((2, 7)) and not e.outcome for e in res.events)


def test_novelty_hits_the_truth_at_the_scheduled_tick(studio):
    spec = NoveltySpec(2, RulePatch("spread_prob:carpet", 0.9))
    w = make(studio, novelties=[spec])
    w.step(WAIT)
    w.step(WAIT)
    assert w.rules.spread_prob["carpet"] == 0.16  # tick 2 not yet processed
    w.step(WAIT)
    assert w.rules.spread_prob["carpet"] == 0.9
    assert w.novelty_log == [(2, spec.patch)]


def test_follower_catches_up_and_gets_rescued(studio):
    always = Rules(follow_prob={"dog": 1.0, "spouse": 1.0, "child": 1.0})
    w = make(studio, "married", rules=always, seed=2, max_steps=60)
    w.agent_pos = (4, 10)  # next to the exit
    w.dependents[0].pos = (4, 9)
    w.step(WAIT)
    assert w.dependents[0].status == FOLLOWING
    for _ in range(8):
        if w.done:
            break
        w.step(RIGHT if w.agent_pos != (4, 11) else WAIT)
    assert w.dependents[0].status == RESCUED
    assert w.outcome == "success"  # rescuer is on the exit and everyone is resolved


def test_success_outcome(studio):
    w = make(studio, "married", max_steps=60)
    w.agent_pos = (4, 11)
    w.dependents[0].pos = (4, 11)
    res = w.step(WAIT)
    assert res.done and res.info["outcome"] == "success"


def test_agent_dies_in_fire(studio):
    w = make(studio, max_steps=60)
    fire = next(iter(w.burning_cells()))
    w.agent_pos = (fire[0], fire[1] + 1)
    w.step(MOVES.index((0, -1)))
    assert not w.agent_alive and w.outcome == "agent_lost"


def test_partial_observability_shrinks_view(ranch):
    w = make(ranch)
    far = w.observe(full=False)
    assert far.visibility_radius is not None
    assert all(abs(c[0] - w.agent_pos[0]) + abs(c[1] - w.agent_pos[1]) <= far.visibility_radius for c in far.burning)
