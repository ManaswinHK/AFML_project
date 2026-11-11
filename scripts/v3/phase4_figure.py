"""Figure: Phase 4, credit error of MAPPO's own critic over training, with and without dropout (seeds 5-7).

    .venv/bin/python scripts/v3/phase4_figure.py
"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for p in sorted(glob.glob(os.path.join(ROOT, "results", "v3", "phase4", "*.json"))):
        r = json.load(open(p))
        drop = "drop" in r["run"]
        t = [c["t_env"] / 1000 for c in r["checkpoints"]]
        kw = dict(color="tab:red" if drop else "tab:blue", alpha=0.8)
        ax[0].plot(t, [c["posthoc_struct_l1"] for c in r["checkpoints"]], **kw,
                   label=("dropout 0.25" if drop else "no dropout") if r["run"].endswith(("s5", "s5_drop0.25")) else None)
        ax[1].plot(t, [c["masked_l1"] for c in r["checkpoints"]], **kw)
    for a, title in zip(ax, ("structural read-out V - W(-i) (post-hoc)", "kappa read-out (pre-registered)")):
        a.axhline(0.667, color="k", ls=":", lw=1)
        a.text(5, 0.69, "equal split", fontsize=8)
        a.set_title(title, fontsize=9)
        a.set_xlabel("training steps (k)")
    ax[0].set_ylabel("L1 vs exact credit of the learned policy")
    ax[0].legend(fontsize=8)
    fig.suptitle("Phase 4: does MAPPO's own critic learn who earned the reward? (2 lanes + 2 bystanders)", fontsize=10)
    fig.tight_layout()
    out = os.path.join(ROOT, "docs", "v3", "figures", "phase4.png")
    fig.savefig(out, dpi=130)
    print("saved", out)


if __name__ == "__main__":
    main()
