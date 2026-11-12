"""Event models and the DEL product update (with ontic postconditions)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Iterable, List

from .kripke import KripkeModel
from .logic import Formula, Not, Top


@dataclass
class EventModel:
    name: str
    events: List[str]
    pre: Dict[str, Formula]
    rel: Dict[str, Dict[str, FrozenSet[str]]]          # agent -> event -> accessible events
    designated: FrozenSet[str]
    post: Dict[str, Dict[str, Formula]] = field(default_factory=dict)  # event -> atom -> formula


def applicable(M: KripkeModel, E: EventModel) -> bool:
    """E is executable iff every designated world enables some designated event."""
    for w in M.designated:
        if not any(e in E.designated and w in M.ext(E.pre[e]) for e in E.events):
            return False
    return True


def product(M: KripkeModel, E: EventModel) -> KripkeModel:
    """Product update M (x) E. Postconditions are evaluated in the *pre*-update model."""
    idx: dict = {}
    val: list = []
    for w in range(M.n):
        for e in E.events:
            if w in M.ext(E.pre[e]):
                idx[(w, e)] = len(val)
                v = set(M.val[w])
                for atom, f in E.post.get(e, {}).items():
                    (v.add if w in M.ext(f) else v.discard)(atom)
                val.append(v)
    rel = {}
    for a in M.agents:
        succ = [None] * len(val)
        for (w, e), i in idx.items():
            succ[i] = {idx[(w2, e2)] for w2 in M.rel[a][w] for e2 in E.rel[a][e]
                       if (w2, e2) in idx}
        rel[a] = succ
    designated = {i for (w, e), i in idx.items() if w in M.designated and e in E.designated}
    return KripkeModel(val, rel, designated)


# ----------------------------------------------------------------------- builders
def announcement(name: str, agents: Iterable[str], pre: Formula) -> EventModel:
    """Truthful public announcement of `pre`."""
    return EventModel(name, ["e"], {"e": pre},
                      {a: {"e": frozenset({"e"})} for a in agents}, frozenset({"e"}))


def public_ontic(name: str, agents: Iterable[str], post: Dict[str, Formula]) -> EventModel:
    """Public, deterministic ontic action with simultaneous assignments atom := formula."""
    return EventModel(name, ["e"], {"e": Top()},
                      {a: {"e": frozenset({"e"})} for a in agents},
                      frozenset({"e"}), {"e": post})


def sensing(name: str, agents: Iterable[str], sensor: str, observers: Iterable[str],
            phi: Formula) -> EventModel:
    """`sensor` learns whether phi.

    * observers know that the sensor looked, but not what it saw (semi-private);
    * every other agent is *unaware* and believes nothing happened (event 's').
    With observers = all other agents this is semi-public sensing; with none, fully private.
    """
    agents = list(agents)
    observers = set(observers)
    rel: dict = {}
    for a in agents:
        if a == sensor:
            r = {"t": frozenset({"t"}), "f": frozenset({"f"}), "s": frozenset({"s"})}
        elif a in observers:
            both = frozenset({"t", "f"})
            r = {"t": both, "f": both, "s": frozenset({"s"})}
        else:
            r = {"t": frozenset({"s"}), "f": frozenset({"s"}), "s": frozenset({"s"})}
        rel[a] = r
    return EventModel(name, ["t", "f", "s"], {"t": phi, "f": Not(phi), "s": Top()},
                      rel, frozenset({"t", "f"}))
