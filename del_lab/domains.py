"""Benchmark domains: Muddy Children and a multi-agent "coin" domain with observability."""
from __future__ import annotations

from itertools import combinations
from typing import Dict, FrozenSet, Iterable, List, Tuple

from .events import EventModel, announcement, applicable, product, public_ontic, sensing
from .kripke import KripkeModel
from .logic import Atom, Formula, K, Not, conj, disj, Kw

H = Atom("h")                       # "the coin shows heads"
Omega = FrozenSet[Tuple[str, str]]  # (x, y) in omega  <=>  y notices when x peeks


# ============================================================ Muddy Children
def muddy_model(m: int, muddy: Iterable[int]) -> KripkeModel:
    """2^m worlds (bit i = child i is muddy); child i sees every forehead but its own."""
    muddy_mask = sum(1 << i for i in muddy)
    val = [{f"m{i}" for i in range(m) if w >> i & 1} for w in range(m and 2 ** m)]
    rel = {f"c{i}": [{w, w ^ (1 << i)} for w in range(2 ** m)] for i in range(m)}
    return KripkeModel(val, rel, {muddy_mask})


def muddy_rounds(m: int, k: int) -> int:
    """Round at which the muddy children first know they are muddy (expected: k)."""
    agents = [f"c{i}" for i in range(m)]
    model = muddy_model(m, range(k))
    some = disj(*[Atom(f"m{i}") for i in range(m)])
    model = product(model, announcement("father", agents, some)).contract()
    nobody_knows = conj(*[~K(f"c{i}", Atom(f"m{i}")) for i in range(m)])
    rnd = 1
    while not all(model.holds(K(f"c{i}", Atom(f"m{i}"))) for i in range(k)):
        model = product(model, announcement("nobody-knows", agents, nobody_knows)).contract()
        rnd += 1
        if rnd > m + 1:
            raise RuntimeError("did not converge")
    return rnd


# ================================================================ Coin domain
def coin_model(agents: Iterable[str]) -> KripkeModel:
    """The coin is heads or tails; nobody knows which. Both worlds are designated."""
    agents = list(agents)
    return KripkeModel([{"h"}, set()], {a: [{0, 1}, {0, 1}] for a in agents}, {0, 1})


def coin_env(agents: Iterable[str], heads: bool) -> KripkeModel:
    """Ground-truth environment: same model, but with the actual world designated."""
    agents = list(agents)
    return KripkeModel([{"h"}, set()], {a: [{0, 1}, {0, 1}] for a in agents},
                       {0} if heads else {1})


def all_omegas(agents: Iterable[str]) -> List[Omega]:
    """Every observability configuration; ordered from most to least observed ('optimistic')."""
    agents = list(agents)
    pairs = [(x, y) for x in agents for y in agents if x != y]
    out = [frozenset(p for i, p in enumerate(pairs) if mask >> i & 1)
           for mask in range(2 ** len(pairs))]
    out.sort(key=lambda om: (-len(om), sorted(om)))
    return out


def coin_actions(agents: Iterable[str], omega: Omega, flip: bool = True,
                 set_heads: bool = False, announce: bool = False) -> Dict[str, EventModel]:
    """Action set induced by the observability configuration `omega`.

    peek_x     x looks at the coin; y notices it iff (x, y) in omega.
    flip       public ontic action h := ~h.
    set_h      public ontic action h := True (conformant effect).
    announce_x public announcement of "x knows whether h" (precondition Kw_x h).
    """
    agents = list(agents)
    acts: Dict[str, EventModel] = {}
    for x in agents:
        obs = [y for y in agents if (x, y) in omega]
        acts[f"peek_{x}"] = sensing(f"peek_{x}", agents, x, obs, H)
    if flip:
        acts["flip"] = public_ontic("flip", agents, {"h": ~H})
    if set_heads:
        acts["set_h"] = public_ontic("set_h", agents, {"h": H | ~H})
    if announce:
        for x in agents:
            acts[f"announce_{x}"] = announcement(f"announce_{x}", agents, Kw(x, H))
    return acts


def step(model: KripkeModel, event: EventModel):
    """Apply an event if executable (contracted result), else None."""
    if not applicable(model, event):
        return None
    return product(model, event).contract()
