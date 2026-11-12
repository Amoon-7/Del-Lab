# DEL-lab — conformant and multi-agent epistemic planning, with learning and monitoring

A small, dependency-free (stdlib only) Python toolkit built around **Dynamic Epistemic Logic (DEL)**:

| Module | Content |
|---|---|
| `del_lab/logic.py` | multi-agent modal formulas (`K`, `Kw`, boolean connectives) |
| `del_lab/kripke.py` | multi-pointed Kripke models, formula evaluation, **bisimulation contraction** (colour refinement) and a **canonical key** for state deduplication |
| `del_lab/events.py` | event models and the **product update** with ontic postconditions; builders for public announcements, public ontic actions and (semi-)private sensing |
| `del_lab/planner.py` | **conformant epistemic planner**: BFS over epistemic states, goal = a formula true at *every* designated world |
| `del_lab/domains.py` | Muddy Children, and a multi-agent *coin* domain with configurable observability |
| `del_lab/learning.py` | Bayesian **learning of observability** (who notices which action) from noisy second-order reports |
| `del_lab/monitoring.py` | **plan monitoring**: discrepancy detection, hypothesis revision, replanning |

## Why this project
Standard epistemic planning assumes the action models (preconditions, effects, *observability*) are given
and correct. Here the observability part of the event models is unknown, observations are noisy, and the
agent must (1) plan conformantly under epistemic uncertainty, (2) learn the event models from traces, and
(3) monitor execution and revise when an outcome contradicts its expected epistemic state.

## Quick start
```bash
pip install -r requirements.txt
python -m pytest -q tests            # 8 tests
python scripts/run_experiments.py    # writes results/*.csv and results/*.png
```

```python
from del_lab import K, Kw, bfs_plan
from del_lab.domains import H, coin_model, coin_actions

agents = ["a", "b"]
goal = Kw("a", H) & ~K("b", Kw("a", H))   # a learns h, b does not know that a knows
# b does not notice a's peek  -> plan exists
print(bfs_plan(coin_model(agents), coin_actions(agents, frozenset()), goal).plan)               # ['peek_a']
# b notices a's peek          -> no plan
print(bfs_plan(coin_model(agents), coin_actions(agents, frozenset({("a", "b")})), goal, max_depth=4).plan)  # None
```

## Modelling choices
* A planning state is a **multi-pointed** Kripke model; the designated set is what the planner considers
  possible as the actual world (conformant planning = the plan must work in all of them).
* `sensing(x, observers)`: `x` learns whether `h`; *observers* learn that `x` looked but not what it saw;
  every other agent is **unaware** and believes nothing happened. Unaware agents get non-reflexive
  accessibility relations, i.e. possibly **false beliefs** (`K_b` is then a belief box).
* An action is executable iff every designated world enables a designated event.
* `bfs_plan(..., contract=True)` contracts each successor by bisimulation and deduplicates by canonical key;
  `contract=False` is the raw tree-search baseline.

## Experiments (`results/`)
**Validation.** Muddy Children: the muddy children know after exactly `k` rounds, for `m ≤ 6`.

**E1 — bisimulation contraction.** Ten successive private peeks by 3 agents: the raw product update doubles
at every step (2 048 worlds after 10 steps), the contracted model stays at 10 worlds. In exhaustive search for an
unreachable goal at depth 5: 364 vs 25 generated nodes, 64 vs 10 worlds in the largest model.

**E2 — learning observability.** Hypothesis space: 2⁶ = 64 observability configurations (who notices whose
peek). After each episode the learner gets a noisy answer to `K_y Kw_x h` for every pair. Median number of
traces for 95 % posterior mass on the truth (30 runs): 2 (no noise), 6 (5 %), 8 (10 %), 18.5 (20 %); the exact
configuration is recovered in 30/30 runs after 40 traces at every noise level.

**E3 — monitoring.** Goal: `a` knows `h`, and `b` and `c` know that `a` knows. Optimistic plan = `[peek_a]`.
Over 400 random worlds:

| strategy | success | mean steps | mean replans |
|---|---|---|---|
| blind (optimistic plan, no monitoring) | 27.8 % | 1.00 | 0 |
| robust (plan valid under all 64 hypotheses) | 100 % | 2.00 | 0 |
| **monitor (optimistic + discrepancy-driven replanning)** | **100 %** | **1.72** | 0.72 |

## Limitations and next steps
* Single-agent-style search without heuristics; the epistemic-state space is not abstracted beyond bisimulation.
* Only deterministic events with a **fixed structure**; learning concerns the observability bits, not
  preconditions/effects. Learning full DEL event models (in the spirit of identification in the limit) is the
  natural continuation.
* Plausibility models / belief revision are not used: revision here = Bayesian elimination of hypotheses.
* Decidable fragments and complexity bounds are not proved here; this is an experimental companion to the
  theoretical questions (conformant fragments, bounded nesting depth, learning of event models).
