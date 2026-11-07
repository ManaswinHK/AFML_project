"""Figures for the report (docs/report.md). Reads the stored Mode P results; nothing is retrained.

    .venv/Scripts/python scripts/make_figures.py

Writes docs/figures/fig1_l1_vs_uniform.png, fig2_interaction_mass.png, fig3_dividend_error.png and, when
the Mode T evaluations exist (results/mode_t/*.json), fig4_return_vs_credit.png.
"""

import collections
import glob
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from analyse_h3 import uniform_l1  # noqa: E402

MODE_P = os.path.join(ROOT, "results", "mode_p")
OUT = os.path.join(ROOT, "docs", "figures")

# Reference categorical palette, slots 1-5 in fixed order (light mode, validated), plus neutral inks.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "regular", "legend.frameon": False,
})


def cells(grid):
    for f in sorted(glob.glob(os.path.join(MODE_P, grid, "*.json"))):
        with open(f) as fh:
            rec = json.load(fh)
        if "truth" in rec:
            yield rec


def ms(v):
    v = np.array(v, float)
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0


def fig1():
    """Individual L1 against leave-one-out vs delay, main grid, slack setting, with the uniform baseline."""
    algs = [("vdn", "VDN"), ("qmix", "QMIX"), ("coma", "COMA")]
    seed = collections.defaultdict(list)  # (series, b, d) -> per-seed means
    for rec in cells("main"):
        c = rec["cell"]
        if c["enabler_start"]:
            continue
        b, d = c["n_bystanders"], c["delay"]
        for a, _ in algs:
            seed[(a, b, d)].append(rec["algs"][a]["summary"]["l1"])
        valid = [e["episode"] for e in rec["algs"]["vdn"]["episodes"] if e["matched"]["valid"]]
        seed[("uniform", b, d)].append(float(np.mean([uniform_l1(rec["truth"]["loo"][i]) for i in valid])))
    delays = (0, 10, 25, 50)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6), sharey=True)
    for ax, b in zip(axes, (0, 2, 4)):
        for (a, label), col in zip(algs, SERIES):
            m = np.array([ms(seed[(a, b, d)]) for d in delays])
            ax.plot(delays, m[:, 0], color=col, lw=2, marker="o", ms=6, mfc=col, mec=SURFACE, mew=2, label=label)
            ax.fill_between(delays, m[:, 0] - m[:, 1], m[:, 0] + m[:, 1], color=col, alpha=0.12, lw=0)
        u = np.array([ms(seed[("uniform", b, d)]) for d in delays])
        ax.plot(delays, u[:, 0], color=INK2, lw=2, ls=(0, (4, 3)), label="uniform credit (1/n)")
        ax.set_title(f"{b} bystanders", loc="left")
        ax.set_xlabel("payout delay d")
        ax.set_xticks(delays)
        ax.set_ylim(0, 1.9)
    axes[0].set_ylabel("individual credit error L1 (lower is better)")
    axes[0].legend(loc="upper left", fontsize=8.5, ncol=2, columnspacing=1)
    fig.suptitle("Learned per-step credit rarely beats an equal split (main grid, slack enabler, mean ± sd over 5 seeds)",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(OUT, "fig1_l1_vs_uniform.png"), dpi=170)


def collect_coalition(layout_dirs, algs_wanted, metric, layouts, delay=0):
    """(alg, layout) -> per-seed values of a summary_coalition metric at one delay."""
    out = collections.defaultdict(dict)
    for g in layout_dirs:
        for rec in cells(g):
            c = rec["cell"]
            lay = c.get("layout", "coalition")
            if lay not in layouts or c["delay"] != delay:
                continue
            for a in algs_wanted:
                r = rec["algs"].get(a)
                if r and "summary_coalition" in r and r["summary_coalition"].get(metric) is not None:
                    out[(a, lay)][c["seed"]] = r["summary_coalition"][metric]
    return out


def grouped_bars(ax, groups, series, data, labels, fmt="{:.2f}"):
    w = 0.8 / len(series)
    for i, (a, label) in enumerate(series):
        xs, ms_ = [], []
        for j, g in enumerate(groups):
            v = list(data.get((a, g), {}).values())
            if v:
                xs.append(j - 0.4 + w * (i + 0.5))
                ms_.append(ms(v))
        if not xs:
            continue
        m = np.array(ms_)
        bars = ax.bar(xs, m[:, 0], width=w - 0.04, color=SERIES[i], label=label, zorder=3)
        ax.errorbar(xs, m[:, 0], yerr=m[:, 1], fmt="none", ecolor=INK2, elinewidth=1, capsize=2, zorder=4)
        for x, (mv, sd) in zip(xs, m):
            ax.text(x, mv + sd + 0.02 * ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else mv, fmt.format(mv),
                    ha="center", va="bottom", fontsize=7.5, color=INK2)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(labels)


def fig2():
    series = [("vdn", "VDN"), ("qmix", "QMIX"), ("na2q", "NA²Q"), ("ece2_mix", "ECE-2 (replay data)"),
              ("ece3_mix", "ECE-3 (replay data)")]
    layouts = ("overlap", "partial")
    data = collect_coalition(("h3", "h5"), [a for a, _ in series], "interaction_mass", layouts)
    fig, ax = plt.subplots(figsize=(8.5, 3.9))
    ax.set_ylim(0, 1.18)
    grouped_bars(ax, layouts, series, data, ["overlap layout", "partial-synergy layout"])
    ax.axhline(1.0, color=INK, lw=1.2, ls=(0, (4, 3)), zorder=2)
    ax.text(-0.45, 1.012, "true value 1.0", ha="left", va="bottom", fontsize=8.5, color=INK)
    ax.set_ylabel("interaction mass\n(share of dividend on 2+ agents)")
    ax.set_title("Per-step mixer games find no coalitions; the episodic estimator does (d = 0, mean ± sd over 5 seeds)",
                 loc="left", fontsize=10.5)
    ax.legend(loc="upper center", ncol=5, fontsize=8, bbox_to_anchor=(0.5, -0.12), columnspacing=1)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2_interaction_mass.png"), dpi=170)


def fig3():
    series = [("na2q", "NA²Q"), ("na2q_mix", "NA²Q + replay data"), ("ece2_mix", "ECE-2 + replay data"),
              ("ece3_mix", "ECE-3 + replay data")]
    layouts = ("coalition", "overlap", "partial")
    data = collect_coalition(("h3", "h5", "h5_ref"), [a for a, _ in series], "dividend_error", layouts)
    fig, ax = plt.subplots(figsize=(8.5, 3.9))
    ax.set_ylim(0, 2.45)
    grouped_bars(ax, layouts, series, data, ["coalition (order 3)", "overlap (order 3)", "partial (order 2)"])
    ax.axhline(2.0, color=INK, lw=1.2, ls=(0, (4, 3)), zorder=2)
    ax.text(2.46, 2.02, "maximum error 2.0", ha="right", va="bottom", fontsize=8.5, color=INK)
    ax.set_ylabel("dividend error (0 = exact)")
    ax.set_title("Dividend error: only the estimator trained with replay data gets below 2 (d = 0, mean ± sd over 5 seeds)",
                 loc="left", fontsize=10.5)
    ax.legend(loc="upper center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.12), columnspacing=1)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3_dividend_error.png"), dpi=170)


def fig4():
    """H4: return vs 1 - L1 per checkpoint, one panel per method (VDN, QMIX with a seeded buffer)."""
    runs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(ROOT, "results", "mode_t", "*_d*_s*.json")))]
    if not runs:
        return False
    methods = [("qmix", "QMIX", SERIES[0]), ("vdn", "VDN", SERIES[1])]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0), sharex=True, sharey=True)
    for ax, (alg, label, col) in zip(axes, methods):
        used = dropped = 0
        xs_all, ys_all = [], []
        for r in (r for r in runs if r["alg"] == alg):
            for c in r["checkpoints"]:
                sm = c["summary"]
                if sm["l1"] is None:
                    dropped += 1
                    continue
                used += 1
                x, y = c["return_mean"], 1.0 - sm["l1"]
                xs_all.append(x)
                ys_all.append(y)
                solid = sm["n_valid"] >= 10
                ax.scatter(x, y, s=34, marker="o" if r["delay"] == 0 else "s", zorder=3,
                           facecolor=col if solid else SURFACE, edgecolor=col, linewidth=1.4)
        if len(xs_all) > 3:
            from scipy.stats import spearmanr
            rho = spearmanr(xs_all, ys_all).statistic
            ax.text(0.97, 0.04, f"Spearman ρ = {rho:+.2f}  ({used} checkpoints)", transform=ax.transAxes,
                    fontsize=9, color=INK2, ha="right")
        ax.axhline(0, color=GRID, lw=1.2, zorder=1)
        ax.set_title(f"{label}  ({dropped} checkpoints with no defined credit not shown)", loc="left", fontsize=9.5)
        ax.set_xlabel("mean return of the test episodes")
    axes[0].set_ylabel("credit accuracy 1 − L1 (higher is better)")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", ls="", color=INK2, label="d = 0"),
               Line2D([], [], marker="s", ls="", color=INK2, label="d = 25"),
               Line2D([], [], marker="o", ls="", mfc=SURFACE, color=INK2, label="< 10 of 30 episodes with defined credit")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8.5, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Return barely predicts credit accuracy (Mode T, seeded VDN and QMIX; each point is a checkpoint)",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    fig.savefig(os.path.join(OUT, "fig4_return_vs_credit.png"), dpi=170)
    return True


def main():
    os.makedirs(OUT, exist_ok=True)
    fig1()
    fig2()
    fig3()
    fig4()
    print("wrote", sorted(os.listdir(OUT)))


if __name__ == "__main__":
    main()
