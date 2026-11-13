"""v4 figures: Part C learning curves, Part A mixed-size bars, Part B LBF credit bars.

    .venv/bin/python scripts/v4/figures.py
"""
from __future__ import annotations

import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
FIG = os.path.join(ROOT, "docs", "v4", "figures")
COL = {"team": "tab:blue", "expected": "tab:orange", "difference": "tab:red",
       "mixed": "tab:red", "single": "tab:blue", "in_size": "tab:gray", "uniform": "lightgray"}


def part_c():
    curves = json.load(open(os.path.join(ROOT, "results", "v4", "online_credit_curves.json")))
    fig, axes = plt.subplots(1, 2, figsize=(8, 3), squeeze=False)
    for c, K in enumerate((4, 8)):
        ax = axes[0][c]
        for arm in ("team", "expected", "difference"):
            cs = [v for k, v in curves.items() if v and f"_K{K}_{arm}_" in k]
            xs = []
            for cv in cs:
                t, y = np.array(cv["t"]), np.array(cv["success"])
                ax.plot(t / 1000, y, color=COL[arm], lw=0.5, alpha=0.35)
                xs.append((t, y))
            g = np.unique(np.concatenate([t for t, _ in xs]))
            ax.plot(g / 1000, np.mean([np.interp(g, t, y, right=y[-1]) for t, y in xs], 0), color=COL[arm], lw=2,
                    label=arm)
        ax.axhline(0.9, color="k", lw=0.5, ls=":")
        ax.set_xscale("log")
        ax.set_title(f"non-shared IPPO, K = {K} lanes", fontsize=9)
        ax.set_xlabel("env steps (k, log scale)")
        ax.set_ylim(0, 1.02)
        if c == 0:
            ax.set_ylabel("greedy test success")
            ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "online_credit_curves.png"), dpi=130)


def part_a():
    fig, axes = plt.subplots(1, 2, figsize=(8, 3), squeeze=False)
    for c, design in enumerate(("interp", "extrap")):
        res = json.load(open(os.path.join(ROOT, "results", "v4", f"size_mixed_confirm_{design}.json")))
        sizes = sorted(res[0]["results"])
        methods = ("in_size", "mixed", "single", "uniform")
        ax = axes[0][c]
        w = 0.2
        for j, m in enumerate(methods):
            vals = np.array([[r["results"][b][m]["l1"] for r in res] for b in sizes])
            ax.bar(np.arange(len(sizes)) + (j - 1.5) * w, vals.mean(1), w, yerr=vals.std(1, ddof=1), color=COL[m],
                   label=m.replace("_", "-"), capsize=2)
        ax.set_xticks(range(len(sizes)))
        ax.set_xticklabels([f"{int(b[1:]) + 3} agents (held out)" for b in sizes], fontsize=8)
        ax.set_title(f"{'interpolation' if design == 'interp' else 'extrapolation'}", fontsize=9)
        if c == 0:
            ax.set_ylabel("L1 vs exact LOO credit")
            ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "size_mixed.png"), dpi=130)


def part_b():
    files = sorted(glob.glob(os.path.join(ROOT, "results", "v4", "lbf_confirm", "*.json")))
    if not files:
        return
    recs = [json.load(open(f)) for f in files]
    variants = sorted({r["variant"] for r in recs})
    methods = ("set", "food", "mlp", "mlp_hi", "set_noint", "uniform", "floor_expected")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3), squeeze=False)
    for c, key in enumerate(("l1", "shap")):
        ax = axes[0][c]
        w = 0.8 / len(methods)
        for j, m in enumerate(methods):
            vals = [[r["methods"][m].get(key) for r in recs if r["variant"] == v] for v in variants]
            if any(x is None for vs in vals for x in vs):
                continue
            vals = np.array(vals, float)
            ax.bar(np.arange(len(variants)) + (j - len(methods) / 2 + 0.5) * w, vals.mean(1), w,
                   yerr=vals.std(1, ddof=1), label=m, capsize=2)
        ax.set_xticks(range(len(variants)))
        ax.set_xticklabels(variants, fontsize=8)
        ax.set_title("leave-one-out L1" if key == "l1" else "Shapley L1 (t = 0 game)", fontsize=9)
        if c == 0:
            ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "lbf_credit.png"), dpi=130)


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    part_c()
    part_a()
    part_b()
    print("saved to", FIG)
