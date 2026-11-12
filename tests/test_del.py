import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from del_lab import (And, Atom, K, KripkeModel, Not, Or, Top, bfs_plan, Kw)
from del_lab.domains import (H, coin_actions, coin_env, coin_model, muddy_rounds)
from del_lab.learning import run_learning
from del_lab.monitoring import run_monitoring


def random_model(rng, n=6, agents=("a", "b"), atoms=("p", "q")):
    val = [{x for x in atoms if rng.random() < .5} for _ in range(n)]
    rel = {a: [{w for w in range(n) if rng.random() < .35} for _ in range(n)] for a in agents}
    return KripkeModel(val, rel, {0})


def random_formula(rng, depth, agents=("a", "b"), atoms=("p", "q")):
    if depth == 0:
        return Atom(rng.choice(atoms))
    c = rng.choice(["not", "and", "or", "k", "k"])
    if c == "not":
        return Not(random_formula(rng, depth - 1))
    if c == "k":
        return K(rng.choice(agents), random_formula(rng, depth - 1))
    mk = And if c == "and" else Or
    return mk(random_formula(rng, depth - 1), random_formula(rng, depth - 1))


def test_muddy_children_rounds():
    for m in (2, 3, 4, 5):
        assert [muddy_rounds(m, k) for k in range(1, m + 1)] == list(range(1, m + 1))


def test_contraction_preserves_truth():
    rng = random.Random(1)
    for _ in range(200):
        M = random_model(rng)
        C = M.contract()
        assert C.n <= M.n
        for _ in range(10):
            f = random_formula(rng, 3)
            assert M.holds(f) == C.holds(f)


def test_canonical_key_invariant_under_renaming_and_duplication():
    rng = random.Random(2)
    for _ in range(100):
        M = random_model(rng).prune()
        perm = list(range(M.n)); rng.shuffle(perm)
        inv = {w: i for i, w in enumerate(perm)}
        val = [None] * M.n
        for w in range(M.n): val[inv[w]] = M.val[w]
        rel = {a: [None] * M.n for a in M.agents}
        for a in M.agents:
            for w in range(M.n): rel[a][inv[w]] = {inv[v] for v in M.rel[a][w]}
        P = KripkeModel(val, rel, {inv[w] for w in M.designated})
        assert M.canonical_key() == P.canonical_key() == M.contract().canonical_key()


def test_conformant_plan_set_heads():
    ag = ["a", "b"]
    r = bfs_plan(coin_model(ag), coin_actions(ag, frozenset(), set_heads=True), K("a", H))
    assert r.plan == ["set_h"]
    # flipping alone can never make a know h
    r = bfs_plan(coin_model(ag), {"flip": coin_actions(ag, frozenset())["flip"]}, K("a", H))
    assert r.plan is None


def test_private_vs_observed_peek():
    ag = ["a", "b"]
    goal = Kw("a", H) & ~K("b", Kw("a", H))     # a learns h, b does not know a knows
    assert bfs_plan(coin_model(ag), coin_actions(ag, frozenset()), goal).plan == ["peek_a"]
    assert bfs_plan(coin_model(ag), coin_actions(ag, frozenset({("a", "b")})), goal,
                    max_depth=4).plan is None


def test_private_peek_gives_false_belief():
    ag = ["a", "b"]
    acts = coin_actions(ag, frozenset())
    from del_lab.events import product
    M = product(coin_model(ag), acts["peek_a"]).contract()
    assert M.holds(Kw("a", H)) and M.holds(K("b", ~Kw("a", H)))


def test_learning_converges_without_noise():
    ok = [run_learning(0.0, 40, seed)[-1]["exact"] for seed in range(5)]
    assert all(ok)


def test_monitoring_always_succeeds():
    rows = run_monitoring(60, seed=3)
    assert all(r["success"] for r in rows["monitor"])
    assert all(r["success"] for r in rows["robust"])
    assert sum(r["success"] for r in rows["blind"]) < 60
