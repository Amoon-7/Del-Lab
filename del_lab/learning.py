"""Learning the observability part of the event models from noisy epistemic reports."""
from __future__ import annotations

import math
import random
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .domains import H, Omega, all_omegas, coin_actions, coin_model
from .events import EventModel
from .kripke import KripkeModel
from .domains import step
from .logic import Formula, K, Kw


class HypothesisSpace:
    """A finite set of candidate event-model families (one per observability config),
    with memoised simulation of action sequences from a fixed initial model."""

    def __init__(self, agents: Sequence[str], init: KripkeModel,
                 actions_for: Callable[[Omega], Dict[str, EventModel]]):
        self.agents = list(agents)
        self.init = init
        self.actions_for = actions_for
        self.omegas: List[Omega] = all_omegas(self.agents)      # optimistic order
        self._actions: Dict[Omega, Dict[str, EventModel]] = {}
        self._models: Dict[Tuple[Omega, tuple], Optional[KripkeModel]] = {}

    def actions(self, om: Omega) -> Dict[str, EventModel]:
        if om not in self._actions:
            self._actions[om] = self.actions_for(om)
        return self._actions[om]

    def model(self, om: Omega, seq: tuple) -> Optional[KripkeModel]:
        if not seq:
            return self.init
        key = (om, seq)
        if key not in self._models:
            prev = self.model(om, seq[:-1])
            self._models[key] = None if prev is None else step(prev, self.actions(om)[seq[-1]])
        return self._models[key]

    def predict(self, om: Omega, seq: tuple, queries: Sequence[Formula]) -> Optional[tuple]:
        m = self.model(om, seq)
        return None if m is None else tuple(m.holds(q) for q in queries)


def second_order_queries(agents: Sequence[str]):
    """'y knows that x knows whether h' for every ordered pair (y, x)."""
    pairs = [(y, x) for y in agents for x in agents if x != y]
    return pairs, [K(y, Kw(x, H)) for (y, x) in pairs]


def run_learning(eps: float, n_traces: int, seed: int, agents=("a", "b", "c"),
                 length: int = 3, space: Optional[HypothesisSpace] = None):
    """One learning run. A trace = fresh episode, `length` random actions, then the learner
    receives a noisy (flip probability eps) answer to every query K_y Kw_x h.
    Returns per-trace posterior mass on the truth, exact-MAP success and per-bit accuracy."""
    rng = random.Random(seed)
    agents = list(agents)
    if space is None:
        space = HypothesisSpace(agents, coin_model(agents),
                                lambda om: coin_actions(agents, om, flip=True))
    pairs = [(x, y) for x in agents for y in agents if x != y]   # bit (x,y): y notices peek_x
    _, queries = second_order_queries(agents)
    names = [f"peek_{x}" for x in agents] + ["flip"]
    true_om = rng.choice(space.omegas)
    e = min(max(eps, 1e-9), 0.5)
    logp = {om: 0.0 for om in space.omegas}
    hist = []
    for _ in range(n_traces):
        seq = tuple(rng.choice(names) for _ in range(length))
        truth = space.predict(true_om, seq, queries)
        answers = tuple(b ^ (rng.random() < eps) for b in truth)
        for om in space.omegas:
            pred = space.predict(om, seq, queries)
            bad = sum(p != a for p, a in zip(pred, answers))
            logp[om] += bad * math.log(e) + (len(answers) - bad) * math.log(1 - e)
        mx = max(logp.values())
        z = sum(math.exp(v - mx) for v in logp.values())
        post = {om: math.exp(v - mx) / z for om, v in logp.items()}
        marg = {p: sum(w for om, w in post.items() if p in om) for p in pairs}
        decoded = {p for p in pairs if marg[p] > 0.5}
        bit_acc = sum((p in decoded) == (p in true_om) for p in pairs) / len(pairs)
        hist.append({"p_true": post[true_om], "exact": decoded == set(true_om), "bit_acc": bit_acc})
    return hist
