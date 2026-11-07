"""Tests for the reward module, written as a reward-hacking audit: each exploit test asks
'could an agent score more by doing something we do not want?' and checks the answer is no."""

import dataclasses

import pytest

from emberpath.agents import BaselineAgent
from emberpath.dependents import HOUSEHOLDS
from emberpath.evaluate import run_episode
from emberpath.grid import MOVES, WAIT
from emberpath.planner import Decision, HazardAwarePlanner
from emberpath.reward import COMPONENTS, RewardConfig, RewardTracker, mission_lower_bound, proximity
from emberpath.rules import Rules, nominal_rules
from emberpath.world import FOLLOWING, LOST, WAITING, World

DOWN = MOVES.index((1, 0))


def calm(burn_steps: int = 10) -> Rules:
    """Fire that never spreads, so exploit tests isolate the reward from the physics."""
    return Rules(spread_prob={"wood": 0.0, "carpet": 0.0, "tile": 0.0, "door": 0.0}, burn_steps=burn_steps)


def play(world: World, agent) -> tuple[RewardTracker, World]:
    tracker = RewardTracker(world)
    obs, events = world.observe(), []
    while not world.done:
        d = agent.act(obs, events)
        res = world.step(d.action)
        tracker.update(world)
        obs, events = res.obs, res.events
    return tracker, world


# ------------------------------------------------------------------ basics
def test_breakdown_has_every_component_and_totals_correctly(studio):
    w = World(studio, HOUSEHOLDS["married"], seed=0)
    t = RewardTracker(w)
    w.step(WAIT)
    b = t.update(w)
    assert set(b.components) == set(COMPONENTS)
    assert b.total == pytest.approx(sum(b.components.values()))


def test_every_tick_costs_time(studio):
    w = World(studio, HOUSEHOLDS["married"], rules=calm(), seed=0)
    t = RewardTracker(w)
    w.step(WAIT)
    assert t.update(w).components["time"] == pytest.approx(-RewardConfig().step_cost)


def test_proximity_shape():
    assert proximity(0, 4) == 1.0
    assert [proximity(d, 4) for d in (1, 2, 3, 4, 9)] == [0.75, 0.5, 0.25, 0.0, 0.0]


# ------------------------------------------------- the fire terms (the brief)
def test_fire_proximity_grows_as_a_dependent_nears_the_flames(studio):
    w = World(studio, HOUSEHOLDS["married"], rules=calm(), seed=0)
    t = RewardTracker(w)
    assert w.burning_cells() == frozenset({(4, 4)})
    readings = []
    for row in (1, 2, 3):  # 3, 2, 1 steps from the fire at (4,4)
        w.dependents[0].pos = (row, 4)
        readings.append(t.fire_proximity(w))
    assert readings == pytest.approx([0.25, 0.5, 0.75])
    w.dependents[0].pos = (2, 10)  # other room, far along any walkable path
    assert t.fire_proximity(w) == 0.0


def test_proximity_is_charged_each_tick_and_scaled(studio):
    w = World(studio, HOUSEHOLDS["married"], rules=calm(burn_steps=50), seed=0)
    w.dependents[0].pos = (3, 4)  # adjacent to the fire
    t = RewardTracker(w)
    w.step(WAIT)
    b = t.update(w)
    assert b.components["fire_proximity"] == pytest.approx(-RewardConfig().proximity_weight * 0.75)


def test_touching_flames_loses_the_dependent_and_is_penalized(studio):
    w = World(studio, HOUSEHOLDS["married"], rules=calm(burn_steps=50), seed=0)
    t = RewardTracker(w)
    w.dependents[0].pos = (4, 4)  # standing in the fire
    w.step(WAIT)
    b = t.update(w)
    assert w.dependents[0].status == LOST
    assert b.components["fire_contact"] == pytest.approx(-RewardConfig().contact_penalty)


def test_touching_costs_far_more_than_being_near(studio):
    cfg = RewardConfig()
    worst_near = cfg.proximity_weight * proximity(1, cfg.proximity_radius)
    assert cfg.contact_penalty > 10 * worst_near


# ------------------------------------------------------- reward-hacking audit
def test_exploit_suicide_is_never_cheaper_than_stalling(studio):
    """The old reward paid -1 for dying but -0.01 x 150 for a timeout, so a hopeless agent preferred death."""
    horizon = 60
    for stall_for in (0, 20, 45):  # die immediately / after dithering for a while
        # world A: wander safely for a while, then walk into the fire
        a = World(studio, HOUSEHOLDS["single"], rules=calm(burn_steps=500), seed=0, max_steps=horizon)
        ta = RewardTracker(a)
        for _ in range(stall_for):
            a.step(WAIT)
            ta.update(a)
        a.agent_pos = (3, 4)
        a.step(DOWN)
        ta.update(a)
        assert a.outcome == "agent_lost"

        # world B: just wait out the clock
        b = World(studio, HOUSEHOLDS["single"], rules=calm(burn_steps=500), seed=0, max_steps=horizon)
        tb = RewardTracker(b)
        while not b.done:
            b.step(WAIT)
            tb.update(b)
        assert b.outcome == "timeout"
        assert ta.total < tb.total, f"dying after {stall_for} ticks ({ta.total:.2f}) beat stalling ({tb.total:.2f})"


def test_exploit_losing_a_dependent_never_raises_the_potential(studio):
    """Otherwise the agent is paid for letting someone burn: their distance term would vanish."""
    for status_before in (WAITING, FOLLOWING):
        for agent_pos in ((1, 1), (2, 6), (3, 9)):
            w = World(studio, HOUSEHOLDS["kids"], rules=calm(), seed=0)
            t = RewardTracker(w)
            w.agent_pos = agent_pos
            for i, d in enumerate(w.dependents):
                d.status = status_before
                before = t.potential(w)
                d.status = LOST
                assert t.potential(w) <= before + 1e-9
                d.status = status_before


def test_exploit_progress_reward_cannot_be_farmed_by_looping(studio):
    """Potential-based shaping telescopes: summed progress = potential(end) - potential(start)."""
    w = World(studio, HOUSEHOLDS["married"], seed=1)
    t = RewardTracker(w)
    phi0 = t.potential(w)
    agent = BaselineAgent(studio)
    obs, events, progress = w.observe(), [], 0.0
    while not w.done:
        res = w.step(agent.act(obs, events).action)
        progress += t.update(w).components["progress"]
        obs, events = res.obs, res.events
    assert progress == pytest.approx(t.cfg.progress_weight * (t.potential(w) - phi0))


def test_exploit_back_and_forth_pays_nothing(studio):
    """Walk away and back with nothing else changing: net progress is exactly zero."""
    w = World(studio, HOUSEHOLDS["married"], rules=calm(), seed=0)
    w.dependents[0].pos = (3, 9)  # parked in the far room; the agent never reaches it
    w.agent_pos = (1, 3)
    t = RewardTracker(w)  # created after the setup so its starting potential is the real one
    left, right = MOVES.index((0, -1)), MOVES.index((0, 1))
    steps_toward = 0.0
    for action in (right, left, right, left, right, left):
        w.step(action)
        steps_toward += t.update(w).components["progress"]
    assert w.agent_pos == (1, 3)
    assert steps_toward == pytest.approx(0.0, abs=1e-12)


class AbandonAgent:
    """Hides every dependent from the planner, so it just runs to the exit alone and waits there."""

    def __init__(self, blueprint):
        self.planner = HazardAwarePlanner(blueprint, nominal_rules())

    def act(self, obs, events):
        return self.planner.act(dataclasses.replace(obs, dependents=()))


def test_exploit_abandoning_the_dependent_scores_worse_than_rescuing(ranch):
    """Ranch, not studio: in the studio the dog is on the way to the exit, so 'abandoning' it rescues it anyway
    (it bonds when the rescuer walks past). In the ranch the dog is in a side room, off the route."""
    wins = 0
    for seed in range(8):
        honest, hw = play(World(ranch, HOUSEHOLDS["single"], seed=seed), BaselineAgent(ranch))
        abandon, aw = play(World(ranch, HOUSEHOLDS["single"], seed=seed), AbandonAgent(ranch))
        assert aw.counts()["rescued"] == 0
        if hw.outcome == "success":
            wins += 1
            assert honest.total > abandon.total + 1.0
    assert wins >= 4


class Delayed:
    def __init__(self, inner, ticks):
        self.inner, self.ticks = inner, ticks

    def act(self, obs, events):
        if self.ticks > 0:
            self.ticks -= 1
            return Decision(WAIT, "dawdling")
        return self.inner.act(obs, events)


def test_faster_rescue_scores_higher(studio):
    """Same outcome, less time, more reward."""
    pairs = []
    for seed in range(14):
        ft, fw = play(World(studio, HOUSEHOLDS["married"], seed=seed), BaselineAgent(studio))
        st, sw = play(World(studio, HOUSEHOLDS["married"], seed=seed), Delayed(BaselineAgent(studio), 6))
        if fw.outcome == sw.outcome == "success":
            pairs.append((ft.total, st.total))
            assert sw.t > fw.t
    assert len(pairs) >= 4
    assert all(fast > slow for fast, slow in pairs)


def test_a_clean_success_is_clearly_positive_and_a_failure_clearly_negative(studio, ranch):
    good = run_episode(studio, HOUSEHOLDS["married"], "baseline", None, seed=0)
    assert good.outcome == "success" and good.reward > 1.0
    stuck = run_episode(ranch, HOUSEHOLDS["single"], "baseline", "door_jam", seed=1)
    assert stuck.outcome == "timeout" and stuck.reward < -1.0


# ------------------------------------------------------------- efficiency
def test_mission_lower_bound_on_a_known_layout(studio):
    w = World(studio, HOUSEHOLDS["married"], seed=0)
    assert mission_lower_bound(w) == 13  # 3 steps to the spouse at (2,3), then 10 to the exit at (4,11)


def test_efficiency_is_a_fraction_for_successes_and_none_otherwise(studio, ranch):
    ok = run_episode(studio, HOUSEHOLDS["married"], "oracle", None, seed=0)
    assert ok.efficiency is not None and 0 < ok.efficiency <= 1
    stuck = run_episode(ranch, HOUSEHOLDS["single"], "baseline", "door_jam", seed=1)
    assert stuck.efficiency is None


def test_control_oracle_and_baseline_get_identical_reward_when_nothing_changes(studio):
    a = run_episode(studio, HOUSEHOLDS["married"], "baseline", None, seed=0)
    b = run_episode(studio, HOUSEHOLDS["married"], "oracle", None, seed=0)
    assert (a.reward, a.efficiency) == (b.reward, b.efficiency)  # control: identical when nothing changes


# --------------------------------------------------------------- gymnasium
def test_env_reports_components_that_sum_to_the_reward(studio):
    pytest.importorskip("gymnasium")
    from emberpath.env import EmberPathEnv

    env = EmberPathEnv(studio, HOUSEHOLDS["kids"])
    env.reset(seed=0)
    for _ in range(5):
        _, reward, term, trunc, info = env.step(WAIT)
        assert reward == pytest.approx(sum(info["reward_components"].values()))
        if term or trunc:
            break
