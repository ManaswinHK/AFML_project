"""Plain-language figures for the slides (for an audience new to multi-agent RL).

    ../../.venv/Scripts/python make_slide_figs.py
Writes fig_background.png (what RL and credit assignment are) and fig_plain_overview.png
(the running example and our method, without jargon).
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

GOLD, TEAL, INK, INK2, GREY, CARD, BLUE = "#E0A030", "#2E8C7E", "#1E2235", "#4A5068", "#C9CDD8", "#EEF1F6", "#2a6fdb"
HERE = os.path.dirname(__file__)


def agent(ax, x, y, L, c, tc="white", r=0.26, fs=12):
    ax.add_patch(Circle((x, y), r, color=c, zorder=3))
    ax.text(x, y, L, ha="center", va="center", color=tc, fontsize=fs, weight="bold", zorder=4)


def box(ax, x, y, w, h, text, color=CARD, tc=INK, fs=10):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.12", color=color, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc, zorder=3)


def arrow(ax, a, b, color=INK2, rad=0.0, lw=1.6):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="->", mutation_scale=14, color=color, lw=lw,
                                 connectionstyle=f"arc3,rad={rad}", zorder=1))


# ---------------------------------------------------------------- background
fig, ax = plt.subplots(figsize=(10, 4.2), dpi=200)
ax.set_xlim(0, 10); ax.set_ylim(0, 4.2); ax.axis("off")

ax.text(0.1, 4.0, "1. Reinforcement learning (trial and error)", fontsize=11, weight="bold", color=INK)
agent(ax, 1.0, 2.3, "agent", BLUE, r=0.5, fs=10)
box(ax, 3.0, 1.85, 1.6, 0.9, "world\n(a game)", fs=10)
arrow(ax, (1.5, 2.55), (3.0, 2.55), rad=-0.25)
ax.text(2.25, 3.05, "takes an action", ha="center", fontsize=9, color=INK2)
arrow(ax, (3.0, 2.05), (1.5, 2.05), rad=-0.25)
ax.text(2.25, 1.35, "gets points (reward)", ha="center", fontsize=9, color=INK2)
ax.text(2.3, 0.55, "Repeat thousands of times;\ndo more of what earned points.", ha="center", fontsize=10, color=INK)

ax.plot([5.05, 5.05], [0.3, 3.8], color=GREY, lw=1)

ax.text(5.45, 4.0, "2. Many agents, one reward", fontsize=11, weight="bold", color=INK)
for i, (L, c) in enumerate([("A", GOLD), ("B", GOLD), ("C", GOLD), ("D", GREY), ("E", GREY)]):
    agent(ax, 5.8 + i * 0.62, 2.75, L, c, INK2 if c == GREY else "white")
box(ax, 7.9, 1.45, 1.2, 0.55, "+10", color=TEAL, tc="white", fs=13)
for i in range(5):
    arrow(ax, (5.8 + i * 0.62, 2.45), (8.3, 2.02), color=GREY, lw=1)
ax.text(7.4, 0.85, "Who earned the +10?", ha="center", fontsize=11, weight="bold", color=INK)
ax.text(7.4, 0.35, "Splitting the reward fairly = credit assignment", ha="center", fontsize=10, color=INK2)
fig.savefig(os.path.join(HERE, "fig_background.png"), bbox_inches="tight", facecolor="white")
plt.close(fig)

# ---------------------------------------------------------------- plain overview
fig, ax = plt.subplots(figsize=(10, 5.2), dpi=200)
ax.set_xlim(0, 10); ax.set_ylim(0, 5.2); ax.axis("off")
ax.text(0.1, 5.0, "(a) Our example: we plant the right answer", fontsize=12, weight="bold", color=INK)
ax.annotate("", xy=(7.4, 4.1), xytext=(0.3, 4.1), arrowprops=dict(arrowstyle="->", color=INK2, lw=1.4))
ax.text(7.45, 4.1, "time", fontsize=9, color=INK2, va="center")
agent(ax, 0.9, 4.1, "A", GOLD); ax.text(0.9, 3.55, "A pulls a lever\n(opens a gate)", ha="center", va="top", fontsize=9, color=INK)
agent(ax, 3.0, 4.1, "B", GOLD); agent(ax, 3.6, 4.1, "C", GOLD)
ax.text(3.3, 3.55, "later, B and C push\na crate together", ha="center", va="top", fontsize=9, color=INK)
ax.add_patch(FancyBboxPatch((5.55, 3.85), 1.1, 0.5, boxstyle="round,pad=0.04", color=TEAL, zorder=3))
ax.text(6.1, 4.1, "+10", ha="center", va="center", color="white", fontsize=13, weight="bold", zorder=4)
ax.text(6.1, 3.55, "points arrive\neven later", ha="center", va="top", fontsize=9, color=INK)
agent(ax, 8.4, 4.35, "D", GREY, INK2); agent(ax, 9.05, 4.35, "E", GREY, INK2)
ax.text(8.72, 3.85, "D, E: wander,\nhelp with nothing", ha="center", va="top", fontsize=9, color=INK2)
ax.text(5.0, 2.8, "Right answer: A, B and C each earned a third.  D and E earned nothing.",
        ha="center", fontsize=10.5, weight="bold", color=INK)

ax.text(0.1, 2.45, "(b) How we check a method", fontsize=12, weight="bold", color=INK)
steps = ["Design the game:\nwe decide who\nmust do what", "A team plays;\nwe record\nevery game", "Re-play each game\nwith agents taken\nout (every combo)",
         "True credit =\nhow much reward\nis lost without you", "Compare: does the\nmethod's credit\nmatch the truth?"]
bw, gap = 1.72, 0.27
for i, t in enumerate(steps):
    x = 0.15 + i * (bw + gap)
    dark = i == 4
    ax.add_patch(FancyBboxPatch((x, 0.25), bw, 1.75, boxstyle="round,pad=0.03,rounding_size=0.12", color=INK if dark else CARD))
    ax.add_patch(Circle((x + 0.3, 1.7), 0.17, color=GOLD))
    ax.text(x + 0.3, 1.7, str(i + 1), ha="center", va="center", color="white", fontsize=10, weight="bold")
    ax.text(x + bw / 2, 0.95, t, ha="center", va="center", fontsize=9, color="white" if dark else INK)
    if i < 4:
        arrow(ax, (x + bw + 0.02, 1.12), (x + bw + gap - 0.02, 1.12))
fig.savefig(os.path.join(HERE, "fig_plain_overview.png"), bbox_inches="tight", facecolor="white")
print("wrote fig_background.png, fig_plain_overview.png")


# ---------------------------------------------------------------- plain copies of the report's results figures
import collections  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(HERE, "..", "..", "scripts"))
import make_figures as mf  # noqa: E402  (same data loaders and style as the report)


def fig_results_plain():
    algs = [("vdn", "VDN"), ("qmix", "QMIX"), ("coma", "COMA")]
    seed = collections.defaultdict(list)
    for rec in mf.cells("main"):
        c = rec["cell"]
        if c["enabler_start"]:
            continue
        b, d = c["n_bystanders"], c["delay"]
        for a, _ in algs:
            seed[(a, b, d)].append(rec["algs"][a]["summary"]["l1"])
        valid = [e["episode"] for e in rec["algs"]["vdn"]["episodes"] if e["matched"]["valid"]]
        seed[("uniform", b, d)].append(float(np.mean([mf.uniform_l1(rec["truth"]["loo"][i]) for i in valid])))
    delays = (0, 10, 25, 50)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6), sharey=True)
    for ax, b in zip(axes, (0, 2, 4)):
        for (a, label), col in zip(algs, mf.SERIES):
            m = np.array([mf.ms(seed[(a, b, d)]) for d in delays])
            ax.plot(delays, m[:, 0], color=col, lw=2, marker="o", ms=6, mfc=col, mec=mf.SURFACE, mew=2, label=label)
            ax.fill_between(delays, m[:, 0] - m[:, 1], m[:, 0] + m[:, 1], color=col, alpha=0.12, lw=0)
        u = np.array([mf.ms(seed[("uniform", b, d)]) for d in delays])
        ax.plot(delays, u[:, 0], color=mf.INK2, lw=2, ls=(0, (4, 3)), label="equal split (no learning)")
        ax.set_title(f"{b} idle agents" if b else "no idle agents", loc="left")
        ax.set_xlabel("how late the reward arrives (steps)")
        ax.set_xticks(delays)
        ax.set_ylim(-0.06, 1.9)
        if b == 0:
            ax.annotate("equal split is exactly right here\n(A, B, C each earned a third): error 0",
                        xy=(32, 0.0), xytext=(12, 0.2), fontsize=8, color=mf.INK,
                        arrowprops=dict(arrowstyle="->", color=mf.INK2, lw=1))
    axes[0].set_ylabel("credit error (lower is better)\n0 = matches the truth")
    axes[0].legend(loc="upper left", fontsize=8.5, ncol=2, columnspacing=1)
    fig.suptitle("Existing methods are rarely better than splitting the reward equally "
                 "(early agent with slack; mean over 5 runs)",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(HERE, "fig_results_plain.png"), dpi=170)
    plt.close(fig)


def fig_teamwork_plain():
    series = [("vdn", "VDN"), ("qmix", "QMIX"), ("na2q", "NA²Q"), ("ece2_mix", "our estimator (pairs)"),
              ("ece3_mix", "our estimator (up to 3)")]
    layouts = ("overlap", "partial")
    data = mf.collect_coalition(("h3", "h5"), [a for a, _ in series], "interaction_mass", layouts)
    fig, ax = plt.subplots(figsize=(8.5, 3.9))
    ax.set_ylim(0, 1.18)
    mf.grouped_bars(ax, layouts, series, data, ["two overlapping teams", "partial teamwork"])
    ax.axhline(1.0, color=mf.INK, lw=1.2, ls=(0, (4, 3)), zorder=2)
    ax.text(-0.45, 1.012, "truth: all the reward comes from teamwork", ha="left", va="bottom", fontsize=8.5, color=mf.INK)
    ax.set_ylabel("share of reward credited\nto teamwork")
    ax.set_title("Existing methods credit almost nothing to teamwork (mean over 5 runs)", loc="left", fontsize=10.5)
    ax.legend(loc="upper center", ncol=5, fontsize=8, bbox_to_anchor=(0.5, -0.12), columnspacing=1)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig_teamwork_plain.png"), dpi=170)
    plt.close(fig)


fig_results_plain()
fig_teamwork_plain()
print("wrote fig_results_plain.png, fig_teamwork_plain.png")


def fig_lbf_plain():
    """Level-Based Foraging, leave-one-out credit error, mean over seeds 10-14 (docs/v4/results.md, Part B)."""
    groups = ["4 agents", "6 agents"]
    series = [  # label, values (4 agents, 6 agents), colour
        ("equal split (no learning)", (0.420, 1.650), "#b5b5b5"),
        ("ours without sitting out", (1.020, 1.242), "#7a7a7a"),
        ("ours", (0.255, 0.873), "#7fa9ec"),
        ("ours, 4× more practice games", (0.155, 0.812), BLUE),
    ]
    fig, ax = plt.subplots(figsize=(8.0, 3.0), dpi=200)
    w = 0.8 / len(series)
    for i, (label, vals, col) in enumerate(series):
        xs = [j - 0.4 + w * (i + 0.5) for j in range(len(groups))]
        ax.bar(xs, vals, width=w - 0.04, color=col, label=label, zorder=3)
        for x, v in zip(xs, vals):
            ax.text(x, v + 0.03, f"{v:.2f}", ha="center", va="bottom", fontsize=8, color=INK2)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, fontsize=10)
    ax.set_ylim(0, 1.95)
    ax.set_ylabel("credit error\n(0 = matches the truth)", fontsize=9, color=INK2)
    ax.set_title("Standard foraging game, new random map every game (mean over 5 runs)", loc="left", fontsize=10, color=INK)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#e6e6e6", lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=8, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig_lbf_plain.png"), facecolor="white")
    plt.close(fig)


fig_lbf_plain()
print("wrote fig_lbf_plain.png")
