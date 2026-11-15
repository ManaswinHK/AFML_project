"""Figure for the 'our fix' slide: credit error of every method on the 12 main v3 conditions.

Numbers are the ranges over conditions (means over seeds 5-9) reported in docs/v3/report.md §4.2
and docs/v3/confirmatory_results.md. L1 = share distance to exact credit (0 = exact, 2 = disjoint).
Writes submission/build/fig_fix.png.
"""
from pathlib import Path

import matplotlib.pyplot as plt

OUT = Path(__file__).with_name("fig_fix.png")
GRAY, OURS, INK, MUTED = "#9a9a9a", "#2a6fdb", "#222222", "#666666"

rows = [  # label, lo, hi, ours?
    ("COMA", 1.45, 1.62, False),
    ("VDN", 0.71, 1.72, False),
    ("QMIX", 0.50, 1.42, False),
    ("MAPPO", 0.83, 2.00, False),
    ("Equal split (no learning)", 0.80, 1.14, False),
    ("Our method without sitting out", 0.85, 1.50, False),
    ("Ours: practice with sitting out", 0.012, 0.063, True),
    ("Ours + treats agents alike", 0.002, 0.026, True),
]

def fmt(v):
    return f"{v:.3f}" if v < 0.1 else f"{v:.2f}"


fig, ax = plt.subplots(figsize=(8.6, 3.9), dpi=200)
for y, (label, lo, hi, ours) in enumerate(reversed(rows)):
    c = OURS if ours else GRAY
    ax.plot([lo, hi], [y, y], color=c, lw=9, solid_capstyle="butt")
    ax.text(hi + 0.05, y, f"{fmt(lo)}–{fmt(hi)}", va="center", fontsize=10,
            color=INK if ours else MUTED, fontweight="bold" if ours else "normal")
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([r[0] for r in reversed(rows)], fontsize=11)
for t, r in zip(ax.get_yticklabels(), reversed(rows)):
    if r[3]:
        t.set_color(OURS)
        t.set_fontweight("bold")
ax.set_xlim(0, 2.35)
ax.set_xticks([0, 0.5, 1.0, 1.5, 2.0])
ax.set_xlabel("credit error  (0 = matches the truth, 2 = completely wrong)", fontsize=10, color=MUTED)
ax.grid(axis="x", color="#e6e6e6", lw=0.8)
ax.set_axisbelow(True)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color("#bbbbbb")
ax.tick_params(axis="y", length=0)
ax.tick_params(axis="x", colors=MUTED)
ax.set_title("Each bar: lowest to highest error over 12 test settings",
             fontsize=10, color=MUTED, loc="left")
fig.text(0.01, 0.01, "MAPPO's 2.00 is a known glitch: its standard credit is an equal split with the wrong sign. "
         "A corrected version scores 0.38–0.85.", fontsize=8, color=MUTED)
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig(OUT, facecolor="white")
print("wrote", OUT)
