"""Plan monitoring: detect that an observation contradicts the expected epistemic state,
revise the hypothesis about the event models, and replan from the current state."""
from __future__ import annotations

import itertools
import random
from typing import Optional

from .domains import H, coin_actions, coin_env, coin_model, step
from .learning import HypothesisSpace
from .logic import K, Kw, conj
from .planner import bfs_plan


def make_space(agents=("a", "b", "c")) -> HypothesisSpace:
    agents = list(agents)
    return HypothesisSpace(agents, coin_model(agents),
                           lambda om: coin_actions(agents, om, flip=False, announce=True))


def goal_formula(agents):
    """Agent a knows whether h, and everybody else knows that a knows it."""
    a, others = agents[0], agents[1:]
    return conj(Kw(a, H), *[K(y, Kw(a, H)) for y in others])


def _execute(env, space, true_om, plan):
    for act in plan:
        env = step(env, space.actions(true_om)[act])
        if env is None:
            return None
    return env


def episode(space: HypothesisSpace, true_om, heads: bool, strategy: str, max_steps: int = 6):
    agents = space.agents
    goal = goal_formula(agents)
    obs_q = [K(y, Kw(agents[0], H)) for y in agents[1:]]       # reactions of the other agents
    env = coin_env(agents, heads)

    def plan_under(om, executed):
        m = space.model(om, tuple(executed))
        return None if m is None else bfs_plan(m, space.actions(om), goal, max_depth=3).plan

    if strategy == "blind":          # plan once under the most optimistic hypothesis
        plan = plan_under(space.omegas[0], [])
        env = _execute(env, space, true_om, plan)
        return dict(success=env is not None and env.holds(goal), steps=len(plan), replans=0)

    if strategy == "robust":         # shortest plan valid under *all* hypotheses
        names = list(space.actions(space.omegas[0]))
        for L in range(1, 4):
            for seq in itertools.product(names, repeat=L):
                if all((m := space.model(om, seq)) is not None and m.holds(goal)
                       for om in space.omegas):
                    env = _execute(env, space, true_om, seq)
                    return dict(success=env is not None and env.holds(goal), steps=L, replans=0)
        return dict(success=False, steps=0, replans=0)

    assert strategy == "monitor"
    hyps, executed, replans = list(space.omegas), [], 0

    def choose():
        for cand in hyps:
            p = plan_under(cand, executed)
            if p is not None:
                return cand, list(p)
        return None, None

    cur, plan = choose()
    while cur is not None and len(executed) < max_steps:
        if env.holds(goal):
            return dict(success=True, steps=len(executed), replans=replans)
        if not plan:
            break
        act = plan.pop(0)
        env = step(env, space.actions(true_om)[act])
        if env is None:
            break
        executed.append(act)
        obs = tuple(env.holds(q) for q in obs_q)
        hyps = [om for om in hyps
                if (m := space.model(om, tuple(executed))) is not None
                and tuple(m.holds(q) for q in obs_q) == obs]
        if cur not in hyps:                      # discrepancy: revise model, replan
            replans += 1
            cur, plan = choose()
    ok = env is not None and env.holds(goal)
    return dict(success=ok, steps=len(executed), replans=replans)


def run_monitoring(n_episodes: int = 200, seed: int = 0, agents=("a", "b", "c")):
    rng = random.Random(seed)
    space = make_space(agents)
    rows = {s: [] for s in ("blind", "robust", "monitor")}
    for _ in range(n_episodes):
        om = rng.choice(space.omegas)
        heads = rng.random() < 0.5
        for s in rows:
            rows[s].append(episode(space, om, heads, s))
    return rows
