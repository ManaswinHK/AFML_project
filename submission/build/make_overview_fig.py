"""Figure 0 for the report: the running example (top) and the evaluation pipeline (bottom).
    ../../.venv/Scripts/python make_overview_fig.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, FancyArrowPatch

GOLD, TEAL, INK, INK2, GREY, CARD = "#E0A030", "#2E8C7E", "#1E2235", "#4A5068", "#C9CDD8", "#EEF1F6"
OUT = os.path.join(os.path.dirname(__file__), "fig0_overview.png")

fig, ax = plt.subplots(figsize=(10, 5.2), dpi=200)
ax.set_xlim(0, 10); ax.set_ylim(0, 5.2); ax.axis("off")


def agent(x, y, L, c, tc="white", r=0.26):
    ax.add_patch(Circle((x, y), r, color=c, zorder=3))
    ax.text(x, y, L, ha="center", va="center", color=tc, fontsize=12, weight="bold", zorder=4)


ax.text(0.1, 5.0, "(a) Running example", fontsize=12, weight="bold", color=INK)
ax.annotate("", xy=(7.4, 4.1), xytext=(0.3, 4.1), arrowprops=dict(arrowstyle="->", color=INK2, lw=1.4))
agent(0.9, 4.1, "A", GOLD); ax.text(0.9, 3.55, "step 3: A pulls a lever,\nopening a gate", ha="center", va="top", fontsize=9, color=INK)
agent(3.0, 4.1, "B", GOLD); agent(3.6, 4.1, "C", GOLD)
ax.text(3.3, 3.55, "step 9: B and C push\na crate together", ha="center", va="top", fontsize=9, color=INK)
ax.add_patch(FancyBboxPatch((5.55, 3.85), 1.1, 0.5, boxstyle="round,pad=0.04", color=TEAL, zorder=3))
ax.text(6.1, 4.1, "+10", ha="center", va="center", color="white", fontsize=13, weight="bold", zorder=4)
ax.text(6.1, 3.55, "step 9 + d: team\nreward arrives", ha="center", va="top", fontsize=9, color=INK)
agent(8.0, 4.35, "D", GREY, INK2); agent(8.65, 4.35, "E", GREY, INK2)
ax.text(8.32, 3.85, "bystanders:\nmove, never act", ha="center", va="top", fontsize=9, color=INK2)

ax.text(0.1, 2.55, "(b) How credit is graded", fontsize=12, weight="bold", color=INK)
steps = ["Plant events:\nwho must act,\nand when", "Scripted team\nplays; episodes\nrecorded", "Replay with every\nsubset of agents\nremoved ($2^n$)", "Exact credit:\ndividends, Shapley,\nleave-one-out", "Compare with each\nmethod's credit\n(L1, leakage, ...)"]
bw, gap = 1.72, 0.27
for i, t in enumerate(steps):
    x = 0.15 + i * (bw + gap)
    dark = i == 4
    ax.add_patch(FancyBboxPatch((x, 0.35), bw, 1.75, boxstyle="round,pad=0.03,rounding_size=0.12", color=INK if dark else CARD))
    ax.add_patch(Circle((x + 0.3, 1.8), 0.17, color=GOLD))
    ax.text(x + 0.3, 1.8, str(i + 1), ha="center", va="center", color="white", fontsize=10, weight="bold")
    ax.text(x + bw / 2, 1.05, t, ha="center", va="center", fontsize=9, color="white" if dark else INK)
    if i < 4:
        ax.add_patch(FancyArrowPatch((x + bw + 0.02, 1.22), (x + bw + gap - 0.02, 1.22), arrowstyle="->", mutation_scale=12, color=INK2))
fig.savefig(OUT, bbox_inches="tight", facecolor="white")
print("wrote", OUT)
