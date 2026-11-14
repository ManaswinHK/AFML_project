"""v5 figures: Part E learning curves (mappo_ns vs ippo_ns), Part F structural credit vs dropout.

    .venv/bin/python scripts/v5/figures.py
"""
from __future__ import annotations

import glob
import json
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "scripts", "v3"))
from run_phase3 import GRIDS, OUT, curve, run_name  # noqa: E402

FIG = os.path.join(ROOT, "docs", "v5", "figures")
COL = {"team": "tab:blue", "expected": "tab:orange", "difference": "tab:red"}


def part_e():
    panels = [("online_credit", "ippo_ns", 8, "local critics (ippo_ns, v4 Part C)"),
              ("credit_mappo_ns", "mappo_ns", 8, "state critic (mappo_ns, v5 Part E)"),
              ("credit_mappo", "mappo", 8, "shared MAPPO (descriptive)")]
    panels = [p for p in panels if os.path.isdir(os.path.join(OUT, p[0]))]
    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 3), squeeze=False)
    for c, (grid, alg, K, title) in enumerate(panels):
        ax = axes[0][c]
        for arm in ("team", "expected", "difference"):
            cs = [curve(os.path.join(OUT, grid, run_name(alg, K, arm, s))) for s in GRIDS[grid]["seeds"]]
            cs = [cv for cv in cs if cv is not None]
            if not cs:
                continue
            for t, y in cs:
                ax.plot(t / 1000, y, color=COL[arm], lw=0.5, alpha=0.35)
            g = np.unique(np.concatenate([t for t, _ in cs]))
            ax.plot(g / 1000, np.mean([np.interp(g, t, y, right=y[-1]) for t, y in cs], 0), color=COL[arm], lw=2, label=arm)
        ax.axhline(0.9, color="k", lw=0.5, ls=":")
        ax.set_xscale("log")
        ax.set_ylim(0, 1.02)
        ax.set_title(f"K = {K}: {title}", fontsize=9)
        ax.set_xlabel("env steps (k, log scale)")
        if c == 0:
            ax.set_ylabel("greedy test success")
            ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "credit_mappo_curves.png"), dpi=130)


def part_f():
    files = glob.glob(os.path.join(ROOT, "results", "v5", "phase4_v5", "*.json"))
    if not files:
        return
    R = defaultdict(list)
    for f in files:
        r = json.load(open(f))
        R[(r["K"], r["dropout"])].append(r)
    fig, ax = plt.subplots(figsize=(4.5, 3))
    for K, col in ((2, "tab:red"), (4, "tab:purple")):
        ps = sorted(p for k, p in R if k == K)
        for p in ps:
            ys = [r["posthoc_struct_l1"] for r in R[(K, p)]]
            ax.scatter([p + (0.008 if K == 4 else -0.008)] * len(ys), ys, color=col, s=10, alpha=0.6)
        ax.plot(ps, [np.mean([r["posthoc_struct_l1"] for r in R[(K, p)]]) for p in ps], color=col, lw=2,
                label=f"K = {K} lanes + 2 bystanders")
        ax.axhline(np.mean([r["uniform_l1"] for p in ps for r in R[(K, p)]]), color=col, ls=":", lw=1)
    ax.set_xlabel("training dropout p")
    ax.set_ylabel("structural credit L1 vs exact")
    ax.set_title("dotted: equal split", fontsize=8)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "phase4_dropout.png"), dpi=130)


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    part_e()
    part_f()
    print("saved to", FIG)
