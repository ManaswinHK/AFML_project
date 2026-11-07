"""Sensitivity check B, part 2: re-test H0 and H1b under the closed-loop removal truth.

`scripts/sensitivity_loop.py` re-scored H1a, H2, H2' and H-fid against a closed-loop truth but left H0
and H1b at their open-loop values, because both need quantities it did not compute:

- **H0** compares two removal nulls: an absent agent is taken off the grid (`remove`) or stays on the
  grid doing nothing (`noop`). The closed-loop analogue of `noop` is computed here: the scripted team is
  re-run and treats the absent agents exactly as under `remove` (they are absent as far as its plans go),
  but those agents stay physically on the grid, inert. So the two closed-loop nulls differ only in
  physical presence, as the two open-loop ones do. The second H0 criterion (realised = planted) uses the
  closed-loop truth already stored by `sensitivity_loop.py`.
- **H1b** scores a perfect per-step critic's credit against leave-one-out removal credit. The per-step
  counterfactuals do not depend on the removal protocol, so they are regenerated and scored against the
  closed-loop leave-one-out target.

Validation, per cell, before anything is written:
- the regenerated per-step counterfactuals reproduce the stored open-loop `semantic_gap` exactly;
- the regenerated closed-loop `remove` truth reproduces the closed-loop truth stored in
  `results/mode_p/main_closed_loop/`.

The closed-loop cell JSONs are then updated in place (`h0`, `data.test_matches_plan`, `semantic_gap`;
the open-loop values are kept under `*_open_loop` keys), and `scripts/analyse_hypotheses.py` re-tests
every hypothesis on that folder. Needs `scripts/sensitivity_loop.py` to have been run first.

    .venv/Scripts/python scripts/sensitivity_loop_h0_h1b.py --workers 10
    .venv/Scripts/python scripts/sensitivity_loop_h0_h1b.py --only d0_b0_es12_s0 --workers 1   # one cell
    .venv/Scripts/python scripts/sensitivity_loop_h0_h1b.py --report-only

Writes results/mode_p/sensitivity/loop_h0_h1b.md.
"""

import argparse
import glob
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from types import SimpleNamespace

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from run_mode_p import cells  # noqa: E402
from whoearned.envs.gridworld import NOOP, PlantedCoalitionGridworld  # noqa: E402
from whoearned.experiments import _h0, semantic_gap  # noqa: E402
from whoearned.replay import (  # noqa: E402
    _collect,
    _run,
    ground_truth_closed_loop,
    ground_truth_open_loop,
    planted_dividends,
    step_counterfactuals,
)

MAIN = os.path.join(ROOT, "results", "mode_p", "main")
CLOSED = os.path.join(ROOT, "results", "mode_p", "main_closed_loop")
REPORT = os.path.join(ROOT, "results", "mode_p", "sensitivity", "loop_h0_h1b.md")
DONE_KEY = "h0_h1b_closed_loop"


class _PresenceView:
    """The live environment as the team sees it: everything is the real env except ``present``."""

    def __init__(self, env, present):
        self._env, self.present = env, present

    def __getattr__(self, name):
        return getattr(self._env, name)


class _InertAbsent:
    """Wraps a team for the closed-loop no-op null: agents outside ``present`` stay on the grid but
    only ever take NOOP, and the team plans as if they were absent."""

    def __init__(self, team, present_mask, n):
        self.team = team
        self.present = [bool(present_mask >> i & 1) for i in range(n)]

    def reset(self, env):
        self.team.reset(_PresenceView(env, self.present))

    def act(self, env):
        actions = list(self.team.act(_PresenceView(env, self.present)))
        return [a if self.present[i] else NOOP for i, a in enumerate(actions)]


def rollout_inert(cfg, team, present_mask):
    env = PlantedCoalitionGridworld(cfg)
    env.reset()  # everyone on the grid
    wrapped = _InertAbsent(team, present_mask, cfg.n_agents)
    wrapped.reset(env)
    return _run(env, lambda t: wrapped.act(env), present_mask)


def ground_truth_closed_loop_noop(cfg, make_team):
    return _collect(cfg, lambda s: rollout_inert(cfg, make_team(), s))


def _close(a, b, atol=1e-9):
    if a is None or b is None:
        return a is None and b is None
    return bool(np.allclose(np.asarray(a, float), np.asarray(b, float), atol=atol))


def run_cell(cell) -> str:
    path = os.path.join(CLOSED, f"{cell.key}.json")
    rec_open = json.load(open(os.path.join(MAIN, f"{cell.key}.json")))
    rec = json.load(open(path))
    cfg = cell.config()
    planted = planted_dividends(cfg)
    test_eps = cell.episodes("test")

    open_truths, closed, closed_noop, steps = [], [], [], {"noop": [], "baseline": []}
    for i, ep in enumerate(test_eps):
        make = lambda i=i: cell.team("test", i)  # noqa: E731
        step = step_counterfactuals(cfg, make, gamma=cell.gamma)
        if not np.array_equal(step.actions, ep.actions):
            raise RuntimeError(f"{cell.key} ep {i}: step truth did not reproduce the test episode")
        steps["noop"].append(step.noop_effect())
        steps["baseline"].append(step.baseline_effect())
        open_truths.append(ground_truth_open_loop(cfg, ep.actions))
        gt = ground_truth_closed_loop(cfg, make)
        if not (_close(gt.shapley, rec["truth"]["shapley"][i]) and _close(gt.leave_one_out, rec["truth"]["loo"][i])):
            raise RuntimeError(f"{cell.key} ep {i}: closed-loop truth differs from the stored one")
        closed.append(gt)
        closed_noop.append(ground_truth_closed_loop_noop(cfg, make))

    # validation: regenerated step truth reproduces the stored open-loop semantic gap
    sg_open = semantic_gap(SimpleNamespace(truths=open_truths, step=steps), cfg)
    for kind, stored in rec_open["semantic_gap"].items():
        for k in ("n_valid", "l1", "leakage", "missed"):
            if not _close(sg_open[kind][k], stored[k]):
                raise RuntimeError(f"{cell.key} semantic_gap {kind}.{k}: {sg_open[kind][k]} != stored {stored[k]}")

    sg_closed = semantic_gap(SimpleNamespace(truths=closed, step=steps), cfg)
    h0_closed = _h0(SimpleNamespace(shapley=[g.shapley.tolist() for g in closed],
                                    noop_shapley=[g.shapley.tolist() for g in closed_noop]))
    rec.setdefault("h0_open_loop", rec_open["h0"])
    rec.setdefault("semantic_gap_open_loop", rec_open["semantic_gap"])
    rec["data"].setdefault("test_matches_plan_open_loop", rec_open["data"]["test_matches_plan"])
    rec["h0"] = h0_closed
    rec["semantic_gap"] = sg_closed
    rec["data"]["test_matches_plan"] = float(np.mean([g.matches(planted) for g in closed]))
    rec[DONE_KEY] = True
    with open(path + ".tmp", "w") as f:
        json.dump(rec, f)
    os.replace(path + ".tmp", path)
    return path


def _verdicts(path):
    return [ln.strip() for ln in open(path, encoding="utf-8") if re.search(r"Verdict|→ \*\*", ln)]


def report():
    recs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(CLOSED, "d*.json")))]
    done = [r for r in recs if r.get(DONE_KEY)]
    if len(done) != len(recs):
        raise SystemExit(f"only {len(done)} of {len(recs)} closed-loop cells have H0/H1b re-computed")
    import analyse_hypotheses  # noqa: E402

    analyse_hypotheses.main(CLOSED)
    out = ["# Sensitivity B, part 2: H0 and H1b under the closed-loop truth", "",
           f"Generated by `scripts/sensitivity_loop_h0_h1b.py` ({len(recs)} cells). Every cell passed both "
           "validations (regenerated per-step truth reproduces the stored open-loop semantic gap; regenerated "
           "closed-loop truth reproduces the stored one).", ""]

    # H0 by data configuration (bystanders, enabler, delay), max over seeds as the rule is per cell
    g = {}
    for r in recs:
        c = r["cell"]
        key = (c["n_bystanders"], "critical" if c["enabler_start"] else "slack", c["delay"])
        g.setdefault(key, []).append((r["h0_open_loop"]["remove_vs_noop_l1_mean"] or 0.0,
                                      r["h0"]["remove_vs_noop_l1_mean"] or 0.0,
                                      r["data"]["test_matches_plan_open_loop"], r["data"]["test_matches_plan"]))
    out += ["## H0 per data configuration", "",
            "Rule (per cell): mean remove-vs-noop Shapley-share L1 ≤ 0.15 and realised = planted ≥ 0.6. Worst cell "
            "over the 5 seeds.", "",
            "| bystanders | enabler | delay | L1 open | L1 closed | planted open | planted closed |",
            "|---|---|---|---|---|---|---|"]
    for key in sorted(g):
        v = np.array(g[key])
        out.append(f"| {key[0]} | {key[1]} | {key[2]} | {v[:, 0].max():.3f} | {v[:, 1].max():.3f} | "
                   f"{v[:, 2].min():.2f} | {v[:, 3].min():.2f} |")

    # H1b
    out += ["", "## H1b: level-2 missed enabler credit (mean over seeds)", "",
            "| notion | enabler | open loop range | closed loop range |", "|---|---|---|---|"]
    for kind in ("noop", "baseline"):
        for es_name, crit in (("slack", False), ("critical", True)):
            cond = {}
            for r in recs:
                c = r["cell"]
                if bool(c["enabler_start"]) != crit:
                    continue
                k = (c["delay"], c["n_bystanders"])
                cond.setdefault(k, []).append((r["semantic_gap_open_loop"][kind]["missed"],
                                               r["semantic_gap"][kind]["missed"]))
            o = [np.mean([x[0] for x in v if x[0] is not None]) for v in cond.values()]
            cl = [np.mean([x[1] for x in v if x[1] is not None]) for v in cond.values()]
            out.append(f"| {kind} | {es_name} | {min(o):.3f}–{max(o):.3f} | {min(cl):.3f}–{max(cl):.3f} |")

    opn = _verdicts(os.path.join(MAIN, "hypotheses.md"))
    cls = _verdicts(os.path.join(CLOSED, "hypotheses.md"))
    out += ["", "## Verdicts (all Mode P hypotheses, now including H0 and H1b)", ""]
    if len(opn) != len(cls):
        out += ["(the verdict lines do not line up; compare the two `hypotheses.md` files by hand)"]
    else:
        out += ["| open loop | closed loop |", "|---|---|"]
        out += [f"| {a} | {b} |" for a, b in zip(opn, cls)]
    text = "\n".join(out) + "\n"
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write(text)
    sys.stdout.buffer.write(text.encode("utf-8"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    p.add_argument("--only", nargs="*", help="cell keys to run (default: every cell not yet done)")
    p.add_argument("--report-only", action="store_true")
    a = p.parse_args()
    if not a.report_only:
        todo = [c for c in cells("main") if (not a.only or c.key in a.only)
                and not json.load(open(os.path.join(CLOSED, f"{c.key}.json"))).get(DONE_KEY)]
        print(f"{len(todo)} cells with {a.workers} workers", flush=True)
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            futs = {pool.submit(run_cell, c): c for c in todo}
            for i, fut in enumerate(as_completed(futs), 1):
                try:
                    fut.result()
                    print(f"[{i}/{len(todo)}] {futs[fut].key} done", flush=True)
                except Exception as e:
                    print(f"[{i}/{len(todo)}] {futs[fut].key} FAILED: {e!r}", flush=True)
        if a.only:
            return
    report()


if __name__ == "__main__":
    main()
