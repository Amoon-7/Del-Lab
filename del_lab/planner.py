"""Breadth-first conformant epistemic planner.

A search state is a multi-pointed Kripke model; the goal is a formula that must hold at
*every* designated world (conformant: the plan has to work whichever designated world is
actual). With `contract=True`, states are reduced by bisimulation contraction and
deduplicated through a canonical key; with `contract=False` this is plain tree search over
raw product-update models (used as a baseline in the experiments).
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .events import EventModel, applicable, product
from .kripke import KripkeModel
from .logic import Formula


@dataclass
class PlanResult:
    plan: Optional[List[str]]
    expanded: int = 0
    generated: int = 0
    max_size: int = 0          # largest model (number of worlds) built during search
    seconds: float = 0.0
    truncated: bool = False


def bfs_plan(init: KripkeModel, actions: Dict[str, EventModel], goal: Formula,
             max_depth: int = 6, contract: bool = True, max_nodes: int = 200_000) -> PlanResult:
    t0 = time.perf_counter()
    start = init.contract() if contract else init
    res = PlanResult(None, 0, 1, start.n)
    if start.holds(goal):
        res.plan = []
        res.seconds = time.perf_counter() - t0
        return res
    visited = {start.canonical_key()} if contract else None
    queue = deque([(start, ())])
    while queue:
        model, path = queue.popleft()
        if len(path) >= max_depth:
            continue
        res.expanded += 1
        for name, event in actions.items():
            if not applicable(model, event):
                continue
            nxt = product(model, event)
            if contract:
                nxt = nxt.contract()
            res.generated += 1
            res.max_size = max(res.max_size, nxt.n)
            if res.generated > max_nodes:
                res.truncated = True
                res.seconds = time.perf_counter() - t0
                return res
            if contract:
                key = nxt.canonical_key()
                if key in visited:
                    continue
                visited.add(key)
            if nxt.holds(goal):
                res.plan = list(path) + [name]
                res.seconds = time.perf_counter() - t0
                return res
            queue.append((nxt, path + (name,)))
    res.seconds = time.perf_counter() - t0
    return res
