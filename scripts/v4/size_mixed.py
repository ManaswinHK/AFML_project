"""v4 Part A: does training the set critic on several team sizes fix credit leakage at unseen sizes?

For each seed: shuffled-role cells (d = 25) for team sizes b in TRAIN; 200 full-team episodes + 200 stops per size
(stratified, no-op null). Trained critics:
  mixed   - one set critic on all TRAIN sizes (fit_set_critic_multi)
  single  - one set critic on a single size (SINGLE_SIZE) with the SAME total number of stops (data-matched)
Each critic is read out (ablation-TD credit) on the 30 test episodes of every TEST size, scored against the exact
closed-loop no-op truth. Also: the in-size critic (trained on the test size itself) and the equal split.

    .venv/bin/python scripts/v4/size_mixed.py --seeds 0,1,2 --design interp --out results/v4/size_mixed_pilot_interp.json
designs: interp (train b = 1, 3, 5; test 2, 4), extrap (train 1, 2, 3; test 4, 5)
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
sys.path.insert(0, os.path.join(ROOT, "scripts", "v3"))
from whoearned import ablation as ab  # noqa: E402
from whoearned.critics import fit_set_critic, fit_set_critic_multi, transfer_set_critic  # noqa: E402
from whoearned.experiments import DataCell  # noqa: E402
from size_transfer import score  # noqa: E402

DESIGNS = {"interp": dict(train=(1, 3, 5), test=(2, 4), single=3),
           "extrap": dict(train=(1, 2, 3), test=(4, 5), single=2)}


def cell(b, seed):
    return DataCell(delay=25, n_bystanders=b, seed=seed, horizon=60, shuffle_roles=True)


def data_for(c: DataCell, n_int: int, data_seed: int):
    facts = [ab.factual_rollout(c.config_for("train", i), lambda i=i: c.team("train", i)) for i in range(c.n_train)]
    d = ab.intervention_dataset(c.config(), lambda i: (lambda: c.team("train", i)), facts, n_int, seed=data_seed,
                                null="noop", sampler="stratified")
    return d.X, d.y


def tests_for(c: DataCell):
    out = []
    for s in range(c.n_test):
        cfg = c.config_for("test", s)
        make = lambda s=s: c.team("test", s)
        f = ab.factual_rollout(cfg, make)
        out.append((f, ab.oracle_ablation_values(cfg, make, f, null="noop")))
    return out


def run(args):
    seed, design = args
    D = DESIGNS[design]
    datasets = {}
    for b in D["train"]:
        X, y = data_for(cell(b, seed), 200, seed)
        datasets[3 + b] = (X, y)
    mixed = fit_set_critic_multi(datasets, 1, seed=seed)
    # data-matched single-size control: same total number of stops, one size
    sb = D["single"]
    Xs, ys = data_for(cell(sb, seed), 200 * len(D["train"]), seed + 1000)
    single = fit_set_critic(Xs, ys, 3 + sb, 1, seed=seed)
    out = {"seed": seed, "design": design, "results": {}}
    for b in D["test"]:
        c = cell(b, seed)
        tests = tests_for(c)
        n = 3 + b
        Xi, yi = data_for(c, 200, seed)
        in_size = fit_set_critic(Xi, yi, n, 1, seed=seed)
        rows = {}
        rows["mixed"] = score(mixed(n), c, tests)
        rows["single"] = score(transfer_set_critic(single.net, n, 1), c, tests)
        rows["in_size"] = score(in_size, c, tests)
        uni = []
        from whoearned.experiments import _roles
        from whoearned.metrics import evaluate_against
        for s, (f, W) in enumerate(tests):
            en, by = _roles(c.config_for("test", s))
            uni.append(evaluate_against(np.ones(n), f.total - W[:, 0], en, by).as_dict()["l1"])
        rows["uniform"] = {"l1": float(np.mean(uni))}
        out["results"][f"b{b}"] = rows
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--design", default="interp", choices=sorted(DESIGNS))
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    jobs = [(int(s), a.design) for s in a.seeds.split(",")]
    with Pool(a.workers) as pool:
        res = pool.map(run, jobs)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    for r in res:
        for b, rows in r["results"].items():
            print(r["seed"], r["design"], b, {k: round(v["l1"], 3) for k, v in rows.items()},
                  "leak mixed", round(rows["mixed"].get("leakage", float("nan")), 3), "top_k mixed", rows["mixed"].get("top_k"))


if __name__ == "__main__":
    main()
