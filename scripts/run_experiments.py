"""Run the three experiments and write CSV tables + figures to results/."""
import csv
import statistics as st
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

from del_lab import Kw, K, bfs_plan, product
from del_lab.domains import H, coin_actions, coin_model, muddy_rounds
from del_lab.learning import HypothesisSpace, run_learning
from del_lab.monitoring import run_monitoring

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def write_csv(name, rows):
    with open(OUT / name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


# ------------------------------------------------------------------ Validation
print("== Muddy Children (rounds until muddy children know; expected = k)")
for m in (3, 4, 5, 6):
    print(f"  m={m}:", [muddy_rounds(m, k) for k in range(1, m + 1)])

# ----------------------------------------------- E1: bisimulation contraction
print("\n== E1: model size under repeated private peeks (3 agents)")
agents = ["a", "b", "c"]
acts = coin_actions(agents, frozenset(), flip=False)       # all peeks fully private
seq = ["peek_a", "peek_b", "peek_c"]
raw = contracted = coin_model(agents)
rows = []
for k in range(1, 11):
    ev = acts[seq[(k - 1) % 3]]
    t = time.perf_counter(); raw = product(raw, ev); t_raw = time.perf_counter() - t
    t = time.perf_counter(); contracted = product(contracted, ev).contract(); t_con = time.perf_counter() - t
    rows.append(dict(steps=k, worlds_raw=raw.n, worlds_contracted=contracted.n,
                     ms_raw=round(1000 * t_raw, 2), ms_contracted=round(1000 * t_con, 2)))
    print("  ", rows[-1])
write_csv("e1_model_size.csv", rows)

print("\n== E1b: planning search effort (unreachable goal, exhaustive search)")
goal = Kw("a", H) & K("b", Kw("a", H))       # unreachable: peeks are private
init = coin_model(agents)
rows_b = []
for depth in (2, 3, 4, 5):
    for mode in (True, False):
        r = bfs_plan(init, acts, goal, max_depth=depth, contract=mode)
        rows_b.append(dict(depth=depth, contraction=mode, generated=r.generated,
                           max_worlds=r.max_size, seconds=round(r.seconds, 3)))
        print("  ", rows_b[-1])
write_csv("e1b_planning.csv", rows_b)

fig, ax = plt.subplots(figsize=(5, 3.4))
ks = [r["steps"] for r in rows]
ax.semilogy(ks, [r["worlds_raw"] for r in rows], "o-", label="raw product update")
ax.semilogy(ks, [r["worlds_contracted"] for r in rows], "s-", label="with bisimulation contraction")
ax.set_xlabel("number of private peeks"); ax.set_ylabel("worlds in the model")
ax.legend(); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(OUT / "e1_model_size.png", dpi=160)

# --------------------------------------------------------- E2: learning
print("\n== E2: learning observability from noisy second-order reports")
SEEDS, N = 30, 40
agents = ["a", "b", "c"]
curves = {}
summary = []
for eps in (0.0, 0.05, 0.1, 0.2):
    space = HypothesisSpace(agents, coin_model(agents), lambda om: coin_actions(agents, om))
    runs = [run_learning(eps, N, seed, space=space) for seed in range(SEEDS)]
    curves[eps] = runs
    first = []
    for h in runs:
        t = next((i + 1 for i, r in enumerate(h) if r["p_true"] > .95), None)
        first.append(t)
    reached = [t for t in first if t is not None]
    summary.append(dict(eps=eps,
                        runs_reaching_95pct=f"{len(reached)}/{SEEDS}",
                        median_traces_to_95pct=st.median(reached) if reached else "-",
                        exact_at_10=round(st.mean(r[9]["exact"] for r in runs), 2),
                        exact_at_40=round(st.mean(r[-1]["exact"] for r in runs), 2),
                        bit_acc_at_10=round(st.mean(r[9]["bit_acc"] for r in runs), 3),
                        bit_acc_at_40=round(st.mean(r[-1]["bit_acc"] for r in runs), 3)))
    print("  ", summary[-1])
write_csv("e2_learning.csv", summary)

fig, axs = plt.subplots(1, 2, figsize=(9, 3.4))
for eps, runs in curves.items():
    axs[0].plot(range(1, N + 1), [st.mean(r[i]["p_true"] for r in runs) for i in range(N)],
                label=f"noise {eps}")
    axs[1].plot(range(1, N + 1), [st.mean(r[i]["bit_acc"] for r in runs) for i in range(N)],
                label=f"noise {eps}")
axs[0].set_ylabel("posterior mass on true model"); axs[1].set_ylabel("per-bit accuracy of MAP")
for a in axs:
    a.set_xlabel("number of traces"); a.grid(alpha=.3); a.legend()
fig.tight_layout(); fig.savefig(OUT / "e2_learning.png", dpi=160)

# --------------------------------------------------------- E3: monitoring
print("\n== E3: plan monitoring (goal: a knows h and b, c know that a knows)")
rows = run_monitoring(n_episodes=400, seed=0)
rows_c = []
for s, rs in rows.items():
    rows_c.append(dict(strategy=s, success_rate=round(st.mean(r["success"] for r in rs), 3),
                       mean_steps=round(st.mean(r["steps"] for r in rs), 3),
                       mean_replans=round(st.mean(r["replans"] for r in rs), 3)))
    print("  ", rows_c[-1])
write_csv("e3_monitoring.csv", rows_c)
print("\nresults written to", OUT)
