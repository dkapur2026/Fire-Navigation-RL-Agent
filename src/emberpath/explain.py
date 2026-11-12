"""Phase 6: turn what the agent knows and did into words a person can read.

``explain_adaptation`` is called by the adaptive agent on the tick it repairs its beliefs. It answers four
questions in plain words, using only the agent's own beliefs and evidence (it is never handed the world):

1. **What changed?** "Door 2 (between the bedroom and the hallway) no longer opens." Doors are numbered and rooms
   named by ``places.py``; several changes at once are grouped ("Door 2 (...) and door 5 (...) no longer open.").
2. **How does that change the plan?** The planner is asked what it would do this tick with the OLD belief and
   with the NEW one (``plan_with``), and the two routes are compared: "Rerouting via the study", "Forcing door 3
   from the hallway (about 10 kicks)", "Going for Max first".
3. **What does it cost?** Added steps, and an estimated added fire risk (formula below).
4. **How sure am I?** When the repair search found a rival explanation almost as good, say so and name it.

Fire risk of a route (``fire_reach_chance``)
--------------------------------------------
"The chance the fire gets to my path before I do": the share of simulated fires in which some cell of the route
catches fire no later than the tick the rescuer would reach it. The simulated fires start from the flames the
agent can see now and spread by the same rule as the simulator, but with the BELIEVED rates: each tick every
burning cell tries once to ignite each 4-neighbouring floor cell with the believed ``spread_prob`` of that cell's
material (so a cell with k burning neighbours catches with 1 - (1 - p)^k); fire never enters exits or burnt-out
cells; a cell burns for the believed ``burn_steps`` and then burns out (cells already burning are assumed to have
their whole burn ahead of them: the agent does not see how long they have burnt). The fire takes a turn only on the
ticks the public fire clock allows (``world.FireClock``: slowly at first, then every tick), counted from the
observation's step, exactly as in the simulator. The rescuer reaches route cell i
at tick ``i * pace`` (the slowest follower's pace, as the planner uses), plus ``1 / force_prob`` ticks of kicking
for every believed-jammed door on the route before it. Over ``R`` simulated fires (default 200, a fixed seed of
its own: the world's fire stream is never touched):

    risk(route) = (1/R) * #{ fires : some route cell i has  ignition_tick(cell_i) <= arrival_tick(i) }

    added fire risk = risk(new route) - risk(old route), both under the NEW belief, on the same simulated fires,

in percentage points. Both routes are scored against the same simulated fires so the difference is not noise
from two separate samples. The "old route" is the route the agent was on before the repair, timed as it was
planned (a door then believed open costs no kicks), so for a jammed door the comparison is "the detour versus the
way I had planned to go". It is a model estimate, not a measured
frequency: it is only as good as the believed rates, and it ignores that the rescuer would react to the fire.

The text is checked against ``JARGON`` in the tests: no "posterior", "Bayes factor", "patch" or rule keys in the
plain part. The numbers behind it (detector scores, repair margins) go in the separate technical appendix.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

import numpy as np

from emberpath.blueprint import Blueprint
from emberpath.grid import Cell, dijkstra, path_to
from emberpath.planner import Decision, HazardAwarePlanner
from emberpath.rules import RulePatch, Rules, door_cell, key_cell, parse_key
from emberpath.world import FOLLOWING, WAITING, Observation, is_fire_tick

_MATERIAL_WORDS = {"wood": "Wood floors", "carpet": "Carpet", "tile": "Tile floors", "door": "Doors"}
_MATERIAL_VERB = {"wood": "are", "carpet": "is", "tile": "are", "door": "are"}
_KIND_WORDS = {"dog": "The dog", "spouse": "Your spouse", "child": "The children"}
_KIND_VERB = {"dog": "is", "spouse": "is", "child": "are"}

# Words that must never reach the plain-language text (tests/test_explain.py checks every explanation).
JARGON = ("posterior", "bayes", "patch", "likelihood", "hypothes", "prior", "spread_prob", "follow_prob",
          "force_prob", "door_jammed", "occupied:", "log ", "sprt", "runner-up", "margin", "belief state")


# ------------------------------------------------------------------------------------------- small phrases
def rate_words(p: float) -> str:
    """A probability as a frequency people can picture: 'about 1 in 6', 'about 5 in 10', 'every time'."""
    if p <= 0.005:
        return "never"
    if p >= 0.995:
        return "every time"
    if p <= 0.5:
        n = round(1 / p)
        if abs(1 / n - p) <= 0.03:
            return f"about 1 in {n}"
    return f"about {int(p * 10 + 0.5)} in 10"


def _join(items: list[str]) -> str:
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:]


def _times(n: int) -> str:
    return "once" if n == 1 else "twice" if n == 2 else f"{n} times"


def _steps(n: float) -> str:
    n = int(round(n))
    return f"{n} step" if n == 1 else f"{n} steps"


def describe_patch(patch: RulePatch, before: Rules | None = None, blueprint: Blueprint | None = None) -> str:
    """One rule change as a sentence. With a blueprint, doors and rooms get their human names."""
    family, name = parse_key(patch.key)
    if family == "spread_prob":
        was = f" (was {before.spread_prob[name]:.0%})" if before else ""
        return (
            f"{_MATERIAL_WORDS[name]} now catch{'es' if name == 'carpet' else ''} fire {float(patch.value):.0%} of the time "
            f"next to a flame{was}."
        )
    if family == "force_prob":
        was = f" (was {before.force_prob[name]:.0%})" if before else ""
        return f"Forcing a jammed door now works {float(patch.value):.0%} of the time per try{was}."
    if family == "follow_prob":
        was = f" (was {before.follow_prob[name]:.0%})" if before else ""
        return f"{_KIND_WORDS[name]} now follows only {float(patch.value):.0%} of the time{was}."
    if family == "occupied":
        cell = key_cell(patch.key)
        where = blueprint.places.room(cell) if blueprint else None
        if where:
            return (f"Someone not on the household list is hiding in {where}." if patch.value
                    else f"Nobody is hiding in {where} after all.")
        return f"Someone not on the household list is {'hiding at' if patch.value else 'no longer at'} {cell}."
    cell = door_cell(patch.key)
    if blueprint:
        return f"{blueprint.places.describe_door(cell)} {'is jammed shut' if patch.value else 'opens again'}."
    return f"The door at {cell} is {'jammed shut' if patch.value else 'working again'}."


# ---------------------------------------------------------------------------------------- what changed
def describe_changes(
    patches: list[RulePatch] | tuple[RulePatch, ...],
    before: Rules,
    blueprint: Blueprint,
    evidence: dict[str, tuple[int, int]] | None = None,
    obs: Observation | None = None,
) -> str:
    """What changed, grouped: all newly jammed doors in one sentence, then the rest, one sentence each.

    `evidence` maps a rule key to (times it happened, times it was tried), from the agent's own event log."""
    places, evidence = blueprint.places, evidence or {}
    out: list[str] = []
    jams = sorted((p for p in patches if parse_key(p.key)[0] == "door_jammed" and p.value),
                  key=lambda p: places.door_numbers.get(door_cell(p.key), 0))
    if jams:
        names = [places.describe_door(door_cell(p.key), capital=i == 0) for i, p in enumerate(jams)]
        stuck = [evidence.get(p.key, (0, 0))[0] for p in jams]
        if len(jams) == 1:
            seen = f" (it was stuck when tried{', ' + _times(stuck[0]) if stuck[0] > 1 else ''})" if stuck[0] else ""
            out.append(f"{names[0]} no longer opens{seen}.")
        else:
            out.append(f"{_join(names)} no longer open (each was stuck when tried).")
    for p in patches:
        family, name = parse_key(p.key)
        s, n = evidence.get(p.key, (0, 0))
        if family == "door_jammed":
            if not p.value:
                out.append(f"{places.describe_door(door_cell(p.key))} opens again.")
        elif family == "spread_prob":
            old, new = before.spread_prob[name], float(p.value)
            pace = "faster" if new > old else "more slowly"
            saw = f" I saw it catch {s} of the {n} times a flame was next to it." if n else ""
            out.append(f"{_MATERIAL_WORDS[name]} {_MATERIAL_VERB[name]} catching fire {pace} than I expected "
                       f"({rate_words(new)} per step next to a flame, instead of {rate_words(old)}).{saw}")
        elif family == "follow_prob":
            old, new = before.follow_prob[name], float(p.value)
            who = _KIND_WORDS[name]
            if obs is not None:
                named = [d.name for d in obs.dependents if d.kind == name and d.id >= 0]
                if len(named) == 1:
                    who = named[0]
            verb = "is" if who not in _KIND_WORDS.values() else _KIND_VERB[name]
            how = "less" if new < old else "more"
            out.append(f"{who} {verb} following {how} reliably than I expected "
                       f"({rate_words(new)} tries instead of {rate_words(old)}).")
        elif family == "force_prob":
            old, new = before.force_prob[name], float(p.value)
            how = "harder" if new < old else "easier"
            out.append(f"Jammed doors are {how} to force than I expected (one gives way {rate_words(new)} kicks, "
                       f"not {rate_words(old)}).")
        elif family == "occupied":
            where = places.room(key_cell(p.key)) or "the house"
            if p.value:
                out.append(f"Someone not on the household list is hiding in {where}: I can hear them calling.")
            else:
                out.append(f"Nobody is hiding in {where} after all.")
    return " ".join(out)


def _rival_words(runner_up: tuple[RulePatch, ...], before: Rules, blueprint: Blueprint) -> str:
    if not runner_up:
        return "that nothing has changed and I was just unlucky"
    return "this: " + describe_changes(runner_up, before, blueprint).rstrip(".")


def describe_doubt(margin: float, runner_up: tuple[RulePatch, ...], before: Rules, blueprint: Blueprint) -> str:
    """The ambiguity sentence. `margin` is the repair search's score gap (a log ratio), shown as 'N times'."""
    ratio = math.exp(max(0.0, margin))
    times = f"{ratio:.1f}" if ratio < 10 else f"{ratio:.0f}"
    return (f"I am not sure about this: it could also be {_rival_words(runner_up, before, blueprint)}. "
            f"My best guess is only about {times} times more believable, so I am acting on it and will keep checking.")


# -------------------------------------------------------------------------------------------- the plan
def plan_with(planner: HazardAwarePlanner, belief: Rules, obs: Observation) -> Decision:
    """What `planner` would decide this tick if it believed `belief`, without touching the planner itself: a copy
    makes the decision (its remembered state, such as cells already searched, is copied, not shared)."""
    twin = copy.copy(planner)
    twin._searched = set(planner._searched)
    twin._crossing = None
    twin.belief = belief
    return twin.act(obs)


def plan_route(planner: HazardAwarePlanner, belief: Rules, obs: Observation, target: Cell | None) -> list[Cell]:
    """The route the planner prices for `target` under `belief` (its own cost of each cell: flames and doors I
    cannot force are walls, a door I can force costs the expected kicks, risky cells cost extra), continued from
    the target by the shortest way to an exit (the planner re-prices that leg when it gets there, so pricing it
    now would compare guesses). Read-only use of the planner's pricing."""
    me = obs.agent_pos
    if target is None or target == me:
        return [me]
    twin = copy.copy(planner)
    twin._crossing = None
    twin.belief = belief
    bp = planner.bp
    eta = twin.fire_eta(obs.burning, obs.burnt, step=obs.step)
    cost = twin._enter_cost(obs, eta, twin._steps_from(me, obs.burning, obs.broken_doors))
    dist, parent = dijkstra(me, cost, bp.height, bp.width)
    path = path_to(parent, me, target)
    if not path:
        return [me]
    if target not in bp.exits:  # then the shortest way out from there: the planner will re-price that leg later
        jammed = belief.jammed_doors - obs.broken_doors
        unit = lambda c: 1.0 if bp.passable(c) and c not in obs.burning and c not in jammed else math.inf  # noqa: E731
        dist2, parent2 = dijkstra(target, unit, bp.height, bp.width)
        exits = [e for e in bp.exits if e in dist2]
        if exits:
            path += path_to(parent2, target, min(exits, key=dist2.get))[1:]
    return path


def _forced(route: list[Cell], belief: Rules, obs: Observation) -> list[int]:
    """Indices of route cells that are doors I believe jammed and have not broken: I must kick them in."""
    jammed = belief.jammed_doors - obs.broken_doors
    return [i for i, c in enumerate(route) if i > 0 and c in jammed]


def arrival_ticks(route: list[Cell], belief: Rules, obs: Observation) -> list[tuple[Cell, float]]:
    """(cell, tick I get there) along a route: walking at the group's pace, plus expected kicks at forced doors."""
    mine = [d for d in obs.dependents if d.status == FOLLOWING and d.leader is None]
    pace = HazardAwarePlanner._pace(mine)
    kick = 1.0 / max(belief.force_prob["door"], 1e-3)
    forced, delay, out = set(_forced(route, belief, obs)), 0.0, []
    for i, cell in enumerate(route):
        if i in forced:
            delay += kick
        if i > 0:
            out.append((cell, i * pace + delay))
    return out


def fire_reach_chance(
    blueprint: Blueprint,
    belief: Rules,
    obs: Observation,
    timed_routes: list[list[tuple[Cell, float]]],
    rollouts: int = 200,
    seed: int = 0,
) -> list[float]:
    """For each timed route, the share of simulated fires (believed rates; see module docstring) in which some
    route cell catches fire no later than the rescuer reaches it. All routes share the same simulated fires."""
    horizon = int(math.ceil(max((a for r in timed_routes for _, a in r), default=0)))
    if horizon <= 0 or not timed_routes:
        return [0.0 for _ in timed_routes]
    bp = blueprint
    h, w = bp.height, bp.width
    floor = np.zeros((h, w), bool)
    p = np.zeros((h, w))
    for cell, material in bp.materials.items():
        if material not in (None, "exit") and bp.passable(cell):
            floor[cell] = True
            p[cell] = belief.spread_prob[material]
    keep = 1.0 - p  # chance one burning neighbour fails to ignite the cell
    burn = np.zeros((rollouts, h, w), np.int16)
    first = np.full((rollouts, h, w), np.iinfo(np.int32).max, np.int32)
    for cell in obs.burnt:
        burn[:, cell[0], cell[1]] = -1
    for cell in obs.burning:
        burn[:, cell[0], cell[1]] = belief.burn_steps
        first[:, cell[0], cell[1]] = 0
    rng = np.random.default_rng(seed)
    k = np.zeros((rollouts, h, w), np.int8)
    for t in range(1, horizon + 1):
        on = burn > 0
        if not on.any():
            break
        if not is_fire_tick(obs.step + t - 1):  # the public fire clock: no fire turn this tick (slow start)
            continue
        k[:] = 0
        k[:, 1:, :] += on[:, :-1, :]
        k[:, :-1, :] += on[:, 1:, :]
        k[:, :, 1:] += on[:, :, :-1]
        k[:, :, :-1] += on[:, :, 1:]
        n, r, c = np.nonzero((burn == 0) & floor & (k > 0))  # cells next to fire that can still catch
        caught = rng.random(len(n)) < 1.0 - keep[r, c] ** k[n, r, c]
        n, r, c = n[caught], r[caught], c[caught]
        burn[on] -= 1
        burn[on & (burn == 0)] = -1
        burn[n, r, c] = belief.burn_steps
        first[n, r, c] = t
    out = []
    for route in timed_routes:
        hit = np.zeros(rollouts, bool)
        for (r, c), arrive in route:
            hit |= first[:, r, c] <= arrive
        out.append(float(hit.mean()))
    return out


def _who(cell: Cell | None, obs: Observation) -> str | None:
    """The person waiting at `cell`, by name, or 'the person I heard' for a believed hiding place."""
    if cell is None:
        return None
    for d in obs.dependents:
        if d.pos == cell and d.id >= 0 and d.status in (WAITING, FOLLOWING):
            return d.name
    return None


def describe_plan_change(
    blueprint: Blueprint,
    obs: Observation,
    before: Rules,
    after: Rules,
    old_plan: Decision,
    new_plan: Decision,
    old_route: list[Cell],
    new_route: list[Cell],
) -> str:
    places, exits = blueprint.places, set(blueprint.exits)
    old_t, new_t = old_plan.target, new_plan.target
    out: list[str] = []
    if new_t != old_t:
        person = _who(new_t, obs)
        if new_t in after.occupied and person is None:
            out.append(f"Going to find whoever is hiding in {places.room(new_t) or 'there'}")
        elif person and _who(old_t, obs):
            out.append(f"Going for {person} first")
        elif person:
            out.append(f"Going to get {person}")
        elif new_t in exits and old_t in exits:
            out.append(f"Heading for {places.exit(new_t)} instead of {places.exit(old_t)}")
        elif new_t in exits:
            out.append(f"Heading for {places.exit(new_t)} now")
        elif new_t is None or new_t == obs.agent_pos:
            out.append("Holding still for now")
        else:
            out.append(f"Heading for {places.place(new_t)} instead")
    was_forced = {old_route[i] for i in _forced(old_route, before, obs)}
    forced_new = [i for i in _forced(new_route, after, obs) if new_route[i] not in was_forced]
    if forced_new:
        i = forced_new[0]
        side = places.room(new_route[i - 1]) or "this side"
        kicks = round(1.0 / max(after.force_prob["door"], 1e-3))
        out.append(f"{'forcing' if out else 'Forcing'} {places.door(new_route[i])} from {side} (about {kicks} kicks)")
    elif new_t == old_t and new_route != old_route:
        fresh_rooms = [r for r in places.rooms_along(new_route) if r not in places.rooms_along(old_route)]
        fresh_doors = [c for c in new_route if c in places.door_numbers and c not in old_route]
        if fresh_rooms:
            out.append(f"Rerouting via {_join(fresh_rooms)}")
        elif fresh_doors:
            out.append(f"Rerouting through {places.door(fresh_doors[0])}")
        else:
            out.append("Taking a slightly different route")
    if not out:
        return "My plan stays the same."
    return ", ".join(out) + "."


# ---------------------------------------------------------------------------------------- the report
@dataclass(frozen=True)
class Explanation:
    """One adaptation, explained. `text` is for anyone; `appendix` holds the numbers behind it."""

    step: int
    changed: str  # what changed
    plan: str = ""  # how the plan changes
    cost: str = ""  # added steps and fire risk
    doubt: str = ""  # uncertainty, when the repair was a close call
    appendix: tuple[str, ...] = ()
    added_steps: float | None = None
    risk_old: float | None = None  # chance the fire reaches the old route first, under the new belief
    risk_new: float | None = None  # same for the new route
    risk_expected: float | None = None  # the old route under the OLD belief (what I thought before)

    @property
    def text(self) -> str:
        return " ".join(s for s in (self.changed, self.plan, self.cost, self.doubt) if s)

    @property
    def headline(self) -> str:
        """Short form for a GIF banner: what changed and what I am doing about it."""
        return " ".join(s for s in (self.changed, self.plan) if s)

    def __str__(self) -> str:
        return self.text


def _pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def describe_cost(
    old_route: list[Cell], new_route: list[Cell], old_times, new_times,
    risk_old: float, risk_new: float, risk_expected: float, spread_changed: bool,
) -> tuple[str, float | None]:
    """'Compared with the way I had planned, this adds about 6 steps and an estimated 12% added fire risk (...)'."""
    if len(new_route) < 2 or len(old_route) < 2:
        return "", None
    if new_route == old_route and old_times == new_times:
        if spread_changed and abs(risk_new - risk_expected) >= 0.01:
            return (f"The chance the fire gets to my path before I do is now about {_pct(risk_new)}, "
                    f"not the {_pct(risk_expected)} I expected."), 0.0
        return "", 0.0
    added = (new_times[-1][1] if new_times else 0.0) - (old_times[-1][1] if old_times else 0.0)
    if added >= 0.5:
        steps = f"adds about {_steps(added)}"
    elif added <= -0.5:
        steps = f"saves about {_steps(-added)}"
    else:
        steps = "takes about as long"
    delta = round(100 * risk_new) - round(100 * risk_old)
    span = f"(the chance the fire gets to my path before I do goes from {_pct(risk_old)} to {_pct(risk_new)})"
    if delta >= 1:
        risk = f", with an estimated {delta}% added fire risk {span}"
    elif delta <= -1:
        risk = f", and cuts the fire risk by about {-delta}% {span}"
    else:
        chance = "almost no" if risk_new < 0.01 else f"about {_pct(risk_new)}"
        risk = f", with no added fire risk ({chance} chance the fire gets to my path before I do)"
    return f"Compared with the way I had planned, this {steps}{risk}.", added


def technical_appendix(
    findings=(),
    result=None,
    evidence: dict[str, tuple[int, int]] | None = None,
    risk: tuple[float, float, float] | None = None,
    rollouts: int = 200,
) -> tuple[str, ...]:
    """The numbers behind an explanation, for an expert: detector scores, the repair search's choice and margin,
    and the fire-risk estimate. Jargon is allowed here; it is never in the plain text."""
    evidence = evidence or {}
    lines = []
    for f in findings:
        s, n = evidence.get(f.key, (None, f.n_trials))
        seen = f"{s}/{n}" if s is not None else f"{n} trials"
        lines.append(
            f"{f.key}: believed {f.believed:.2f}, seen {seen}, posterior mean {f.posterior_mean:.2f} "
            f"(90% interval {f.interval90[0]:.2f}-{f.interval90[1]:.2f}), log Bayes factor {f.log_bayes_factor:.1f}, "
            f"flagged at t={f.first_flagged_step}"
        )
    if result is not None:
        edits = ", ".join(f"{p.key} -> {p.value}" for p in result.patches) or "none (kept the belief)"
        rival = ", ".join(f"{p.key} -> {p.value}" for p in result.runner_up) or "no change"
        margin = "inf" if math.isinf(result.runner_up_margin) else f"{result.runner_up_margin:.2f}"
        gain = "n/a" if math.isnan(result.log_likelihood_gain) else f"{result.log_likelihood_gain:.2f}"
        lines.append(f"repair: {edits}; log-likelihood gain {gain}; runner-up: {rival} (score margin {margin}"
                     f"{', ambiguous' if result.ambiguous else ''})")
    if risk is not None:
        old, new, expected = risk
        lines.append(f"fire risk ({rollouts} simulated fires, believed rates): old route {old:.3f} -> new route "
                     f"{new:.3f}; old route under the old belief {expected:.3f}")
    return tuple(lines)


def explain_adaptation(
    blueprint: Blueprint,
    obs: Observation,
    before: Rules,
    result,
    old_plan: Decision,
    new_plan: Decision,
    planner: HazardAwarePlanner,
    findings=(),
    evidence: dict[str, tuple[int, int]] | None = None,
    rollouts: int = 200,
) -> Explanation:
    """Explain one repair: what changed, how the plan changes, what it costs, and how sure I am.

    Everything comes from the agent's side: the blueprint, its observation, its belief before (`before`) and after
    (`result.rules`) the repair, the `RepairResult`, the plans made with each belief (`plan_with`), the detector's
    `findings` and its own event tallies (`evidence`: key -> (happened, tried)). It is not given the world."""
    after = result.rules
    old_route = plan_route(planner, before, obs, old_plan.target)
    new_route = plan_route(planner, after, obs, new_plan.target)
    # The old route is timed as I had planned it (doors I then thought open take no kicks); fire uses the new rates.
    old_times, new_times = arrival_ticks(old_route, before, obs), arrival_ticks(new_route, after, obs)
    risk_old, risk_new = fire_reach_chance(blueprint, after, obs, [old_times, new_times], rollouts)
    spread_changed = before.spread_prob != after.spread_prob
    if spread_changed:
        (risk_expected,) = fire_reach_chance(blueprint, before, obs, [arrival_ticks(old_route, before, obs)], rollouts)
    else:
        risk_expected = risk_old
    cost, added = describe_cost(old_route, new_route, old_times, new_times, risk_old, risk_new, risk_expected,
                                spread_changed)
    doubt = describe_doubt(result.runner_up_margin, result.runner_up, before, blueprint) if result.ambiguous else ""
    return Explanation(
        step=obs.step,
        changed=describe_changes(result.patches, before, blueprint, evidence, obs),
        plan=describe_plan_change(blueprint, obs, before, after, old_plan, new_plan, old_route, new_route),
        cost=cost,
        doubt=doubt,
        appendix=technical_appendix(findings, result, evidence, (risk_old, risk_new, risk_expected), rollouts),
        added_steps=added,
        risk_old=risk_old,
        risk_new=risk_new,
        risk_expected=risk_expected,
    )


@dataclass
class DecisionLog:
    """An audit trail: every decision with its stated reason."""

    entries: list[tuple[int, Decision]] = field(default_factory=list)

    def add(self, step: int, decision: Decision) -> None:
        self.entries.append((step, decision))

    def narrate(self, collapse_repeats: bool = True) -> list[str]:
        lines: list[str] = []
        last = None
        for step, d in self.entries:
            if collapse_repeats and d.reason == last:
                continue
            lines.append(f"t={step:>3}  {d.reason}")
            last = d.reason
        return lines

    def explanations(self) -> list[tuple[int, str]]:
        """Only the ticks where the agent explained a change of mind, as (step, plain text)."""
        return [(step, d.explanation) for step, d in self.entries if getattr(d, "explanation", "")]
