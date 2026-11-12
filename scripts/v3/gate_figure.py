"""Figure: Phase 3 learning curves (greedy test success vs environment steps), team vs difference rewards.

One panel per K; one row per grid (gate = shared parameters; gate_ns = no parameter sharing, post-hoc).
Thin lines: seeds; thick: mean over seeds (curves are held at their last value after an early stop).

    .venv/bin/python scripts/v3/gate_figure.py
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
from run_phase3 import GRIDS, OUT, curve, run_name  # noqa: E402

COLORS = {"team": "tab:blue", "team_vector": "tab:green", "difference": "tab:red"}


def main():
    grids = [g for g in ("gate", "gate_ns") if os.path.isdir(os.path.join(OUT, g))]
    Ks = GRIDS["gate"]["Ks"]
    fig, axes = plt.subplots(len(grids), len(Ks), figsize=(3.2 * len(Ks), 2.8 * len(grids)), squeeze=False)
    for r, grid in enumerate(grids):
        g = GRIDS[grid]
        for c, K in enumerate(Ks):
            ax = axes[r][c]
            for arm in ("team", "difference"):
                curves = []
                for s in g["seeds"]:
                    cv = curve(os.path.join(OUT, grid, run_name(g["algs"][0], K, arm, s)))
                    if cv is not None:
                        ax.plot(cv[0] / 1000, cv[1], color=COLORS[arm], lw=0.5, alpha=0.4)
                        curves.append(cv)
                if curves:
                    grid_x = np.unique(np.concatenate([cv[0] for cv in curves]))
                    ys = [np.interp(grid_x, cv[0], cv[1], right=cv[1][-1]) for cv in curves]
                    ax.plot(grid_x / 1000, np.mean(ys, 0), color=COLORS[arm], lw=2, label=arm)
            ax.axhline(0.9, color="k", lw=0.5, ls=":")
            ax.set_title(f"{'shared' if grid == 'gate' else 'non-shared (post-hoc)'}, K = {K}", fontsize=9)
            ax.set_xlabel("env steps (k)")
            if c == 0:
                ax.set_ylabel("test success")
            ax.set_ylim(0, 1.02)
            if r == 0 and c == 0:
                ax.legend(fontsize=8)
    fig.tight_layout()
    out = os.path.join(ROOT, "docs", "v3", "figures", "gate_curves.png")
    fig.savefig(out, dpi=130)
    print("saved", out)


if __name__ == "__main__":
    main()
