"""Epistemic formulas (multi-agent modal logic K, evaluated on Kripke models).

K(a, f) is the box modality of agent a: true when f holds in every world a
considers possible. With equivalence relations it is knowledge; with the
non-reflexive relations produced by private events it is (possibly false) belief.
"""
from __future__ import annotations

from dataclasses import dataclass


class Formula:
    def __invert__(self) -> "Formula":
        return Not(self)

    def __and__(self, other: "Formula") -> "Formula":
        return And(self, other)

    def __or__(self, other: "Formula") -> "Formula":
        return Or(self, other)

    def implies(self, other: "Formula") -> "Formula":
        return Or(Not(self), other)


@dataclass(frozen=True, repr=False)
class Top(Formula):
    def __repr__(self) -> str:
        return "T"


@dataclass(frozen=True, repr=False)
class Bot(Formula):
    def __repr__(self) -> str:
        return "F"


@dataclass(frozen=True, repr=False)
class Atom(Formula):
    name: str

    def __repr__(self) -> str:
        return self.name


@dataclass(frozen=True, repr=False)
class Not(Formula):
    f: Formula

    def __repr__(self) -> str:
        return f"~{self.f}"


@dataclass(frozen=True, repr=False)
class And(Formula):
    l: Formula
    r: Formula

    def __repr__(self) -> str:
        return f"({self.l} & {self.r})"


@dataclass(frozen=True, repr=False)
class Or(Formula):
    l: Formula
    r: Formula

    def __repr__(self) -> str:
        return f"({self.l} | {self.r})"


@dataclass(frozen=True, repr=False)
class K(Formula):
    agent: str
    f: Formula

    def __repr__(self) -> str:
        return f"K_{self.agent} {self.f}"


def Kw(agent: str, f: Formula) -> Formula:
    """'agent knows whether f'."""
    return K(agent, f) | K(agent, Not(f))


def conj(*fs: Formula) -> Formula:
    out: Formula = Top()
    for f in fs:
        out = f if isinstance(out, Top) else And(out, f)
    return out


def disj(*fs: Formula) -> Formula:
    out: Formula = Bot()
    for f in fs:
        out = f if isinstance(out, Bot) else Or(out, f)
    return out
