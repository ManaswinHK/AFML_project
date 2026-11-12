"""Figure: the commitment timeline of one episode, exact (oracle) vs learned (set critic, 200 stops).

Panels: (a) cumulative leave-one-out structural credit per agent; (b) per-step Shapley credit of the pending
game (efficient: sums to the value committed at each step); (c) Harsanyi dividend flows of {A,B,C} and {B,C}.
The payout arrives d steps after the joint finish; credit does not move there.

    .venv/bin/python scripts/v3/timeline_figure.py --seed 7 --delay 25
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))
from whoearned import ablation as ab  # noqa: E402
from whoearned import ground_truth as gt  # noqa: E402
from whoearned.critics import fit_set_critic  # noqa: E402
from whoearned.experiments import DataCell  # noqa: E402

NAMES = "ABCDEFG"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--delay", type=int, default=25)
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--out", default=os.path.join(ROOT, "docs", "v3", "figures", "commitment_timeline.png"))
    a = p.parse_args()
    cell = DataCell(delay=a.delay, n_bystanders=2, seed=a.seed, horizon=60)
    cfg = cell.config()
    n = cfg.n_agents
    facts = [ab.factual_rollout(cfg, lambda i=i: cell.team("train", i)) for i in range(cell.n_train)]
    data = ab.intervention_dataset(cfg, lambda i: (lambda: cell.team("train", i)), facts, 200, seed=a.seed)
    critic = fit_set_critic(data.X, data.y, n, 1, seed=a.seed)
    make = lambda: cell.team("test", a.episode)
    f = ab.factual_rollout(cfg, make)
    W_true = ab.all_masks_values(cfg, make, f, null="noop")
    W_hat = ab.learned_all_masks(critic, f, null="noop")
    W_hat[-1] = ab.learned_value(critic, f)
    sv_true, sv_hat = ab.step_semivalue_credit(W_true), ab.step_semivalue_credit(W_hat)
    rec = f.event_log[0]
    T = cfg.horizon
    fig, ax = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    colors = plt.cm.tab10(np.arange(n))
    for i in range(n):
        ax[0].plot(np.cumsum(sv_true["loo"][i]), color=colors[i], lw=2, label=f"{NAMES[i]} exact")
        ax[0].plot(np.cumsum(sv_hat["loo"][i]), color=colors[i], lw=1, ls="--")
        ax[1].bar(np.arange(T) + (i - n / 2) * 0.12, sv_true["shapley"][i], width=0.12, color=colors[i])
    ax[1].plot(sv_hat["committed"], color="k", lw=1, ls="--", label="committed value (learned)")
    for name, coal in (("{A,B,C}", (0, 1, 2)), ("{B,C}", (1, 2))):
        m = gt.mask_of(coal)
        ax[2].plot(np.cumsum(sv_true["dividends"][m]), lw=2, label=f"{name} exact")
        ax[2].plot(np.cumsum(sv_hat["dividends"][m]), lw=1, ls="--", label=f"{name} learned")
    for axis in ax:
        for t, lab in ((rec.pull_times.get(0), "pull"), (rec.completion_time, "finish"), (rec.payout_time, "payout")):
            if t is not None:
                axis.axvline(t, color="grey", lw=0.8, ls=":")
                axis.text(t, axis.get_ylim()[1] * 0.9, lab, fontsize=8, rotation=90, va="top")
    ax[0].set_ylabel("cumulative LOO credit")
    ax[0].legend(ncol=5, fontsize=7)
    ax[1].set_ylabel("per-step Shapley credit")
    ax[1].legend(fontsize=7)
    ax[2].set_ylabel("cumulative dividend released")
    ax[2].set_xlabel("step t")
    ax[2].legend(fontsize=7)
    fig.suptitle(f"Commitment timeline, coalition layout d = {a.delay}, b = 2 (seed {a.seed}, test episode {a.episode}). "
                 "Solid: exact; dashed: learned set critic (200 stops)", fontsize=9)
    fig.tight_layout()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=130)
    print("saved", a.out, "final LOO exact", np.round(sv_true["loo"].sum(1), 2), "learned", np.round(sv_hat["loo"].sum(1), 2))


if __name__ == "__main__":
    main()
