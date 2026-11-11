"""Aggregate v3 cell JSONs into per-condition tables (mean over seeds, with the seed count).

    .venv/bin/python scripts/v3/aggregate.py pilot_fixed [--metrics fid_l1,step_r,top_k] [--csv out.csv]

Metrics per (cell, method):
  fid_l1    L1 vs the exact persistent-ablation truth under the method's own null (closed loop, CRN) [ATD only]
  step_r    per-step Pearson r vs its level-1 target (exact kappa for ATD; one-step truth for EPyMARL methods)
  loo_c     L1 vs closed-loop leave-one-out (removal at t = 0)     loo_o  same, open-loop truth
  shap_c    L1 vs closed-loop Shapley                               shap_o same, open loop
  leak_c    leakage vs closed-loop LOO;  miss_c missed enabler credit vs closed-loop LOO
  top_k     P(top-|coalition| credited agents == coalition), ties broken at random
  div_c     dividend error vs closed-loop dividends                 div_o  same, open loop
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def cell_condition(cell: dict) -> tuple:
    return (("perm_" if cell.get("shuffle_roles") else "") + cell["layout"], cell["delay"], cell["n_bystanders"],
            "crit" if cell["enabler_start"] else "slack")


def metrics(rec: dict) -> dict:
    f = rec.get("summary_fidelity") or {}
    c = rec.get("summary_coalition") or {}
    cc = rec.get("summary_coalition_closed") or {}
    lc = rec.get("summary_loo_closed") or {}
    return {
        "fid_l1": f.get("l1"), "step_r": f.get("step_pearson"),
        "loo_c": lc.get("l1"), "loo_o": rec["summary_loo"]["l1"],
        "shap_c": (rec.get("summary_shapley_closed") or {}).get("l1"), "shap_o": rec["summary_shapley"]["l1"],
        "leak_c": lc.get("leakage"), "miss_c": lc.get("missed"),
        "top_k": rec.get("top_k"), "div_c": cc.get("dividend_error"), "div_o": c.get("dividend_error"),
    }


def load(grid: str):
    rows = []
    for p in sorted(glob.glob(os.path.join(ROOT, "results", "v3", grid, "*.json"))):
        r = json.load(open(p))
        cond = cell_condition(r["cell"])
        for alg, rec in r["algs"].items():
            rows.append((cond, r["cell"]["seed"], alg, metrics(rec)))
    return rows


def table(rows, keys, algs=None):
    groups = defaultdict(lambda: defaultdict(list))
    for cond, seed, alg, m in rows:
        if algs and alg not in algs:
            continue
        for k in keys:
            if m.get(k) is not None:
                groups[(cond, alg)][k].append(m[k])
    conds = sorted({c for c, _ in groups})
    alg_order = list(dict.fromkeys(a for _, a in groups))
    if algs:
        alg_order = [a for a in algs if a in alg_order]
    out = []
    head = "| condition | method | n | " + " | ".join(keys) + " |"
    out += [head, "|" + "---|" * (3 + len(keys))]
    for cond in conds:
        for alg in alg_order:
            g = groups.get((cond, alg))
            if not g:
                continue
            n = max(len(v) for v in g.values())
            vals = [f"{np.mean(g[k]):.3f}" if g.get(k) else "" for k in keys]
            out.append(f"| {' '.join(map(str, cond))} | {alg} | {n} | " + " | ".join(vals) + " |")
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("grid")
    p.add_argument("--metrics", default="fid_l1,step_r,loo_c,loo_o,top_k,div_c,div_o")
    p.add_argument("--algs", default=None)
    a = p.parse_args()
    rows = load(a.grid)
    print(table(rows, a.metrics.split(","), a.algs.split(",") if a.algs else None))


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(ROOT, "src"))
    main()
