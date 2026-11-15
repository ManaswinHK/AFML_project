"""Figures that explain our method against the existing ones (slides 8 and 9).

    ../../.venv/Scripts/python make_method_figs.py
Writes fig_two_questions.png (schematic) and, when results/viz holds a cell with our estimators,
fig_credit_over_time.png (real per-step credit from the interactive viewer's data).
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

GOLD, TEAL, INK, INK2, GREY, CARD = "#E0A030", "#2E8C7E", "#1E2235", "#4A5068", "#C9CDD8", "#EEF1F6"
RED, GREEN, BLUE, ORANGE = "#c0392b", "#1e8449", "#2a6fdb", "#eb6834"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))


def agent(ax, x, y, L, c=GOLD, tc="white", r=0.22, alpha=1.0, ls="-"):
    ax.add_patch(Circle((x, y), r, facecolor=c, edgecolor=INK2 if ls != "-" else c, lw=1.2, ls=ls, alpha=alpha, zorder=3))
    ax.text(x, y, L, ha="center", va="center", color=tc, fontsize=11, weight="bold", zorder=4, alpha=alpha)


def reward(ax, x, y, text, color=TEAL):
    ax.add_patch(FancyBboxPatch((x - 0.42, y - 0.2), 0.84, 0.4, boxstyle="round,pad=0.03", color=color, zorder=3))
    ax.text(x, y, text, ha="center", va="center", color="white", fontsize=11, weight="bold", zorder=4)


def timeline(ax, y, x0=0.3, x1=5.6):
    ax.annotate("", xy=(x1, y), xytext=(x0, y), arrowprops=dict(arrowstyle="->", color=INK2, lw=1.2))


def two_questions():
    fig, ax = plt.subplots(figsize=(12, 5.0), dpi=200)
    ax.set_xlim(0, 12); ax.set_ylim(0, 5.0); ax.axis("off")
    for x0, title, sub, col in [(0.0, "Existing methods ask:", "“What if A paused for one step?”", RED),
                                (6.2, "The right question:", "“What if A weren’t there at all?”", GREEN)]:
        ax.add_patch(FancyBboxPatch((x0 + 0.1, 0.15), 5.6, 4.7, boxstyle="round,pad=0.02,rounding_size=0.15",
                                    facecolor="white", edgecolor=col, lw=2))
        ax.text(x0 + 0.35, 4.45, title, fontsize=12, color=INK2)
        ax.text(x0 + 0.35, 4.0, sub, fontsize=14, weight="bold", color=col)

    # left: real game, then the one-step pause
    ax.text(0.35, 3.35, "Real game", fontsize=10, color=INK2)
    timeline(ax, 3.0)
    agent(ax, 1.0, 3.0, "A"); ax.text(1.0, 2.6, "pulls lever", ha="center", fontsize=8.5, color=INK)
    agent(ax, 2.9, 3.0, "B"); agent(ax, 3.4, 3.0, "C"); ax.text(3.15, 2.6, "push crate", ha="center", fontsize=8.5, color=INK)
    reward(ax, 4.9, 3.0, "+10")
    ax.text(0.35, 2.05, "A pauses one step", fontsize=10, color=INK2)
    timeline(ax, 1.7)
    agent(ax, 1.0, 1.7, "A", c="white", tc=INK2, ls="--", alpha=0.9)
    agent(ax, 1.55, 1.7, "A"); ax.text(1.55, 1.3, "pulls one step later", ha="center", fontsize=8.5, color=INK)
    agent(ax, 2.9, 1.7, "B"); agent(ax, 3.4, 1.7, "C")
    reward(ax, 4.9, 1.7, "+10")
    ax.text(2.95, 0.55, "Nothing is lost  →  A gets ≈ 0 credit  ✗", ha="center", fontsize=12, weight="bold", color=RED)

    # right: real game, then A removed
    ax.text(6.55, 3.35, "Real game", fontsize=10, color=INK2)
    timeline(ax, 3.0, 6.5, 11.8)
    agent(ax, 7.2, 3.0, "A"); ax.text(7.2, 2.6, "pulls lever", ha="center", fontsize=8.5, color=INK)
    agent(ax, 9.1, 3.0, "B"); agent(ax, 9.6, 3.0, "C"); ax.text(9.35, 2.6, "push crate", ha="center", fontsize=8.5, color=INK)
    reward(ax, 11.1, 3.0, "+10")
    ax.text(6.55, 2.05, "A is not there", fontsize=10, color=INK2)
    timeline(ax, 1.7, 6.5, 11.8)
    agent(ax, 7.2, 1.7, "A", c="white", tc=INK2, ls="--", alpha=0.9)
    ax.text(7.2, 1.3, "gate stays shut", ha="center", fontsize=8.5, color=INK)
    agent(ax, 9.1, 1.7, "B", alpha=0.45); agent(ax, 9.6, 1.7, "C", alpha=0.45)
    ax.text(9.35, 1.3, "can't reach crate", ha="center", fontsize=8.5, color=INK)
    reward(ax, 11.1, 1.7, "0", color="#9a9a9a")
    ax.text(9.15, 0.55, "All 10 points lost  →  A earned a third  ✓", ha="center", fontsize=12, weight="bold", color=GREEN)
    fig.savefig(os.path.join(HERE, "fig_two_questions.png"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def how_it_works():
    fig, ax = plt.subplots(figsize=(12, 3.3), dpi=200)
    ax.set_xlim(0, 12); ax.set_ylim(0, 3.3); ax.axis("off")
    heads = [(0.05, "1. Practice: someone sits out"), (5.0, "2. Train a critic"), (8.25, "3. Credit = with you − without you")]
    for x, h in heads:
        ax.text(x, 3.1, h, fontsize=11.5, weight="bold", color=INK)
    rows = [("normal game", "", "+10", TEAL), ("A sits out from the start", "A", "0", "#9a9a9a"),
            ("D sits out", "D", "+10", TEAL), ("B sits out after the push", "B", "+10", TEAL)]
    for k, (lab, who, r, col) in enumerate(rows):
        y = 2.5 - k * 0.62
        ax.text(0.1, y, lab, fontsize=9.5, color=INK, va="center")
        for j, L in enumerate("ABCDE"):
            c = GOLD if L in "ABC" else GREY
            if L == who:
                agent(ax, 2.55 + j * 0.38, y, L, c="white", tc=INK2, r=0.15, ls="--")
            else:
                agent(ax, 2.55 + j * 0.38, y, L, c=c, tc="white" if L in "ABC" else INK2, r=0.15)
        ax.add_patch(FancyBboxPatch((4.2, y - 0.16), 0.55, 0.32, boxstyle="round,pad=0.02", color=col))
        ax.text(4.475, y, r, ha="center", va="center", color="white", fontsize=9.5, weight="bold")
    ax.add_patch(FancyArrowPatch((4.85, 1.6), (5.15, 1.6), arrowstyle="->", mutation_scale=16, color=INK2, lw=1.6))
    ax.add_patch(FancyBboxPatch((5.2, 0.75), 2.7, 1.7, boxstyle="round,pad=0.03,rounding_size=0.12", color=CARD))
    ax.text(6.55, 1.6, "neural network that predicts\nthe points still to come,\ngiven who is still playing",
            ha="center", va="center", fontsize=10, color=INK)
    ax.add_patch(FancyArrowPatch((7.95, 1.6), (8.25, 1.6), arrowstyle="->", mutation_scale=16, color=INK2, lw=1.6))
    ax.add_patch(FancyBboxPatch((8.3, 0.45), 3.6, 2.3, boxstyle="round,pad=0.03,rounding_size=0.12", color=INK))
    lines = [("A:", "10 − 0 = 10", "→ a third", GOLD), ("B, C:", "10 − 0 = 10", "→ a third", GOLD),
             ("D, E:", "10 − 10 = 0", "→ nothing", GREY)]
    for k, (who, calc, res, c) in enumerate(lines):
        y = 2.25 - k * 0.55
        ax.text(8.5, y, who, fontsize=11, weight="bold", color=c, va="center")
        ax.text(9.35, y, calc, fontsize=11, color="white", va="center")
        ax.text(10.75, y, res, fontsize=10, color="white", va="center")
    ax.text(10.1, 0.68, "exactly “what if you weren’t there?”", ha="center", fontsize=9.5, color="#dddddd", style="italic")
    fig.savefig(os.path.join(HERE, "fig_how_it_works.png"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def credit_over_time(key="d25_b2_s0"):
    """Agent A's cumulative credit share over one test episode: our sitting-out critic vs QMIX vs the truth."""
    path = os.path.join(ROOT, "results", "viz", f"{key}.json")
    if not os.path.exists(path):
        print("skipped fig_credit_over_time.png: no", path)
        return
    data = json.load(open(path))
    if "estimators" not in data:
        print("skipped fig_credit_over_time.png: cell has no estimators yet")
        return
    # pick a slack-enabler test episode where the realised truth matches the plan (A, B, C a third each)
    eps = [e for e in data["episodes"] if e["matches_plan"] and not e["enabler_critical"]
           and e["methods"]["atd_set_remove"]["shares"] is not None and e["methods"]["qmix"]["shares"] is not None]
    e = eps[0]
    T = len(e["rewards"])
    pull = min(int(t) for ev in e["events"] for t in ev["pull_times"].values())
    done = e["events"][0]["completion_time"]
    pay = e["events"][0]["payout_time"]

    def cum_share(per_step, agent=0):
        ps = np.array(per_step)                    # (T, n)
        total = np.abs(ps.sum(0)).sum()
        return np.cumsum(ps[:, agent]) / total if total else np.zeros(len(ps))

    ours = cum_share(e["methods"]["atd_set_remove"]["per_step"])
    qmix = cum_share(e["methods"]["qmix"]["per_step"])
    truth_share = e["truth"]["loo_shares"][0]
    t = np.arange(len(ours))
    fig, ax = plt.subplots(figsize=(6.2, 3.3), dpi=220)
    ax.axhline(truth_share, color="#888888", lw=2, ls=(0, (4, 3)), label=f"truth: A earned {truth_share:.2f} of the credit")
    ax.step(t, ours, where="post", color=BLUE, lw=2.4, label="ours (trained with sitting out)")
    ax.step(t, qmix, where="post", color=ORANGE, lw=2.0, label="QMIX")
    for x, lab in [(pull, "A pulls\nthe lever"), (done, "B, C push\nthe crate"), (pay, "+10\narrives")]:
        if x is not None and x < T:
            ax.axvline(x, color="#cccccc", lw=1, zorder=0)
            ax.text(x + 0.6, 0.62, lab, fontsize=8, color=INK2, va="top")
    ax.set_xlim(0, min(T, (pay or T) + 6))
    ax.set_ylim(-0.1, 0.66)
    ax.set_xlabel("time step in the game", fontsize=9, color=INK2)
    ax.set_ylabel("agent A's credit so far\n(share of the total)", fontsize=9, color=INK2)
    ax.set_title(f"One real test game (reward delay {data['cell']['delay']} steps): when does A get its credit?",
                 loc="left", fontsize=10, color=INK)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(color="#eeeeee", lw=0.8)
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.42), fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig_credit_over_time.png"), facecolor="white")
    plt.close(fig)
    print(f"fig_credit_over_time.png from {key}, test episode #{e['index'] + 1}")


two_questions()
how_it_works()
credit_over_time()
print("done")
