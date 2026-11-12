"""Figure: credit error vs number of stops (interventions), MLP vs set critic, fixed and shuffled roles.

Reads results/v3/confirm_main (atd_n*, atd, atd_noint) and results/v3/confirm_perm (atd, atd_n800, atd_set_n*,
atd_set, atd_n800_set, atd_noint). Values: mean over seeds of L1 vs the exact no-op truth; bars = min..max.

    .venv/bin/python scripts/v3/scaling_figure.py
"""
from __future__ import annotations

import glob
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MLP = {"atd_noint": 0, "atd_n25": 25, "atd_n50": 50, "atd_n100": 100, "atd": 200, "atd_n400": 400, "atd_n800": 800}
SET = {"atd_set_n25": 25, "atd_set_n50": 50, "atd_set_n100": 100, "atd_set": 200, "atd_n800_set": 800}


def collect(grid, mapping, cond_ok=lambda c: True):
    vals = defaultdict(lambda: defaultdict(list))
    for p in glob.glob(os.path.join(ROOT, "results", "v3", grid, "*.json")):
        r = json.load(open(p))
        c = r["cell"]
        if not cond_ok(c):
            continue
        for alg, n in mapping.items():
            rec = r["algs"].get(alg)
            if rec and (rec.get("summary_loo_closed_noop") or {}).get("l1") is not None:
                vals[c["n_bystanders"]][n].append(rec["summary_loo_closed_noop"]["l1"])
    return vals


def plot(ax, vals, label, style):
    for b, by_n in sorted(vals.items()):
        ns = sorted(by_n)
        x = [max(n, 10) for n in ns]   # put "no stops" at 10 on the log axis
        m = [np.mean(by_n[n]) for n in ns]
        lo = [m_ - np.min(by_n[n]) for n, m_ in zip(ns, m)]
        hi = [np.max(by_n[n]) - m_ for n, m_ in zip(ns, m)]
        ax.errorbar(x, m, yerr=[lo, hi], marker=style, capsize=3, label=f"{label}, b = {b}")


def main():
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    plot(ax[0], collect("confirm_main", MLP, lambda c: c["delay"] == 25 and c["enabler_start"] == 0), "MLP", "o")
    ax[0].set_title("Fixed roles (coalition layout, d = 25, slack)")
    plot(ax[1], collect("confirm_perm", MLP), "MLP", "o")
    plot(ax[1], collect("confirm_perm", SET), "set critic", "s")
    ax[1].set_title("Shuffled roles (d = 25)")
    for a in ax:
        a.set_xscale("log")
        a.set_xlabel("number of stops in the data (10 = none)")
        a.axhline(0.0, color="k", lw=0.5)
        a.legend(fontsize=8)
    ax[0].set_ylabel("L1 vs exact removal credit (0 = exact)")
    fig.suptitle("Credit by intervention: error vs data (seeds 5-9, mean and range)", fontsize=10)
    fig.tight_layout()
    out = os.path.join(ROOT, "docs", "v3", "figures", "scaling.png")
    fig.savefig(out, dpi=130)
    print("saved", out)


if __name__ == "__main__":
    main()
