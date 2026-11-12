"""Zero-shot size transfer of the permutation-invariant ablation critic (exploratory pilot / P12).

Train the set critic on shuffled-role teams with b_train bystanders (200 stops, stratified, no-op null);
read out ablation-TD credit for shuffled-role teams with b_test bystanders it has never seen, and score it
against the exact closed-loop truth (no-op null). References: the same critic trained in-size, and uniform.

    .venv/bin/python scripts/v3/size_transfer.py --seeds 0,1,2 --out results/v3/size_transfer_pilot.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))
from whoearned import ablation as ab  # noqa: E402
from whoearned.critics import fit_set_critic, transfer_set_critic  # noqa: E402
from whoearned.experiments import DataCell, _roles  # noqa: E402
from whoearned.metrics import evaluate_against, pearson, top_k_selection  # noqa: E402


def train_critic(cell: DataCell, n_int=200):
    facts = [ab.factual_rollout(cell.config_for("train", i), lambda i=i: cell.team("train", i))
             for i in range(cell.n_train)]
    data = ab.intervention_dataset(cell.config(), lambda i: (lambda: cell.team("train", i)), facts, n_int,
                                   seed=cell.seed, null="noop", sampler="stratified")
    return fit_set_critic(data.X, data.y, cell.config().n_agents, 1, seed=cell.seed)


def score(predict, cell: DataCell, tests):
    rows, kh, kt = [], [], []
    for s, (f, W) in enumerate(tests):
        cfg = cell.config_for("test", s)
        enablers, bystanders = _roles(cfg)
        Wh = ab.learned_ablation_values(predict, f, null="noop")
        k = ab.ablation_td_credit(f.rewards, Wh)
        kt_ = ab.ablation_td_credit(f.rewards, W)
        r = evaluate_against(k.sum(1), kt_.sum(1), enablers, bystanders).as_dict()
        r["top_k"] = top_k_selection(k.sum(1), cfg.events[0].coalition)
        rows.append(r)
        kh.append(k)
        kt.append(kt_)
    v = [r for r in rows if r["valid"]]
    out = {m: float(np.mean([r[m] for r in v if r[m] is not None])) for m in ("l1", "leakage", "missed", "top_k")}
    out["step_r"] = pearson(np.stack(kh), np.stack(kt))
    out["n_valid"] = len(v)
    return out


def run(args):
    seed, b_train, b_test = args
    tr = DataCell(delay=25, n_bystanders=b_train, seed=seed, horizon=60, shuffle_roles=True)
    te = DataCell(delay=25, n_bystanders=b_test, seed=seed, horizon=60, shuffle_roles=True)
    tests = []
    for s in range(te.n_test):
        cfg = te.config_for("test", s)
        make = lambda s=s: te.team("test", s)
        f = ab.factual_rollout(cfg, make)
        tests.append((f, ab.oracle_ablation_values(cfg, make, f, null="noop")))
    src = train_critic(tr)
    n_test = te.config().n_agents
    out = {"seed": seed, "b_train": b_train, "b_test": b_test,
           "transfer": score(transfer_set_critic(src.net, n_test, 1), te, tests),
           "in_size": score(train_critic(te), te, tests),
           "uniform": score(lambda S: np.zeros(len(S)), te, tests) if False else None}
    # uniform: equal split scored against the same truth
    rows = []
    for s, (f, W) in enumerate(tests):
        cfg = te.config_for("test", s)
        enablers, bystanders = _roles(cfg)
        rows.append(evaluate_against(np.ones(n_test), f.total - W[:, 0], enablers, bystanders).as_dict())
    v = [r for r in rows if r["valid"]]
    out["uniform"] = {"l1": float(np.mean([r["l1"] for r in v])), "n_valid": len(v)}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--pairs", default="2:4,4:2,2:0")
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    seeds = [int(x) for x in a.seeds.split(",")]
    pairs = [tuple(int(y) for y in x.split(":")) for x in a.pairs.split(",")]
    jobs = [(s, bt, be) for (bt, be) in pairs for s in seeds]
    with Pool(a.workers) as pool:
        res = pool.map(run, jobs)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    for r in res:
        print(r["seed"], f"b{r['b_train']}->b{r['b_test']}", "transfer", round(r["transfer"]["l1"], 3),
              "in-size", round(r["in_size"]["l1"], 3), "uniform", round(r["uniform"]["l1"], 3),
              "top_k", round(r["transfer"]["top_k"], 2))


if __name__ == "__main__":
    main()
