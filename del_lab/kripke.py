"""Kripke models, bisimulation contraction and a canonical key for deduplication."""
from __future__ import annotations

import hashlib
from typing import Iterable, Sequence

from .logic import And, Atom, Bot, Formula, K, Not, Or, Top


def _h(obj) -> str:
    return hashlib.blake2b(repr(obj).encode(), digest_size=8).hexdigest()


class KripkeModel:
    """Multi-agent, multi-pointed Kripke model.

    worlds are 0..n-1; ``val[w]`` is the set of true atoms; ``rel[a][w]`` is the set of
    worlds agent ``a`` considers possible at ``w``. ``designated`` is the set of worlds
    the planner considers possible as the actual one (a conformant planner must succeed
    in all of them).
    """

    def __init__(self, val: Sequence[Iterable[str]], rel: dict, designated: Iterable[int]):
        self.val = [frozenset(v) for v in val]
        self.rel = {a: [frozenset(s) for s in succ] for a, succ in rel.items()}
        self.designated = frozenset(designated)
        self.agents = sorted(self.rel)
        self._ext: dict = {}

    @property
    def n(self) -> int:
        return len(self.val)

    # ------------------------------------------------------------------ semantics
    def ext(self, f: Formula) -> frozenset:
        """Set of worlds satisfying f (memoised)."""
        cached = self._ext.get(f)
        if cached is not None:
            return cached
        n = self.n
        if isinstance(f, Top):
            r = frozenset(range(n))
        elif isinstance(f, Bot):
            r = frozenset()
        elif isinstance(f, Atom):
            r = frozenset(w for w in range(n) if f.name in self.val[w])
        elif isinstance(f, Not):
            r = frozenset(range(n)) - self.ext(f.f)
        elif isinstance(f, And):
            r = self.ext(f.l) & self.ext(f.r)
        elif isinstance(f, Or):
            r = self.ext(f.l) | self.ext(f.r)
        elif isinstance(f, K):
            sub = self.ext(f.f)
            succ = self.rel[f.agent]
            r = frozenset(w for w in range(n) if succ[w] <= sub)
        else:
            raise TypeError(f"unknown formula {f!r}")
        self._ext[f] = r
        return r

    def holds_at(self, f: Formula, w: int) -> bool:
        return w in self.ext(f)

    def holds(self, f: Formula) -> bool:
        """f is true at every designated world."""
        return self.designated <= self.ext(f)

    # ------------------------------------------------------------ bisimulation
    def prune(self) -> "KripkeModel":
        """Keep only worlds reachable from the designated worlds."""
        seen = set(self.designated)
        stack = list(seen)
        while stack:
            w = stack.pop()
            for a in self.agents:
                for v in self.rel[a][w]:
                    if v not in seen:
                        seen.add(v)
                        stack.append(v)
        order = sorted(seen)
        idx = {w: i for i, w in enumerate(order)}
        val = [self.val[w] for w in order]
        rel = {a: [{idx[v] for v in self.rel[a][w]} for w in order] for a in self.agents}
        return KripkeModel(val, rel, {idx[w] for w in self.designated})

    def _colours(self) -> list:
        """Colour refinement. At the fixpoint, two worlds share a colour iff bisimilar.
        Colours are content hashes, so they are canonical across models."""
        col = [_h(tuple(sorted(v))) for v in self.val]
        count = len(set(col))
        while True:
            new = [
                _h((col[w], tuple((a, tuple(sorted({col[v] for v in self.rel[a][w]})))
                                  for a in self.agents)))
                for w in range(self.n)
            ]
            m = len(set(new))
            col = new
            if m == count:
                return col
            count = m

    def contract(self) -> "KripkeModel":
        """Bisimulation contraction restricted to the part reachable from designated worlds."""
        p = self.prune()
        col = p._colours()
        classes = sorted(set(col))
        cidx = {c: i for i, c in enumerate(classes)}
        rep: dict = {}
        for w, c in enumerate(col):
            rep.setdefault(c, w)
        val = [p.val[rep[c]] for c in classes]
        rel = {
            a: [{cidx[col[v]] for v in p.rel[a][rep[c]]} for c in classes]
            for a in p.agents
        }
        return KripkeModel(val, rel, {cidx[col[w]] for w in p.designated})

    def canonical_key(self) -> tuple:
        """Hashable key, equal for bisimilar (pointed) models; used for state deduplication."""
        p = self.prune()
        col = p._colours()
        return (tuple(sorted(set(col))), tuple(sorted({col[w] for w in p.designated})))

    def __repr__(self) -> str:
        return f"KripkeModel(n={self.n}, designated={sorted(self.designated)})"
