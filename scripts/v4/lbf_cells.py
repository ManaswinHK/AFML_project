"""v4 Part B: credit by intervention on Level-Based Foraging (deterministic LBF, exact truth).

One cell = (variant, seed). 200 training episodes and 30 test episodes, each with its own random layout (positions,
levels, food). Scripted coalition-forming team, epsilon = 0.2. Truth per test episode: exact persistent-ablation
values under the no-op null (closed loop, common random numbers) for every agent at every step, and the closed-loop
no-op game v(S) at t = 0 for every subset S.

Methods: uniform; MLP critic without stops / with 200 / 800 stops; two-set critic without stops / with 200 stops
(kappa and structural read-outs). Metrics per method: L1 vs exact LOO, per-step r vs exact kappa, Shapley L1,
dividend error, spurious credit (share given to agents with zero true LOO), top-k (keeping exactly the agents that
matter, ties at random).

    .venv/bin/python scripts/v4/lbf_cells.py --variants base --seeds 0,1,2 --out results/v4/lbf_pilot --workers 3
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))
from whoearned import ablation as ab  # noqa: E402
from whoearned import ground_truth as gt  # noqa: E402
from whoearned.critics import fit_mlp, fit_two_set_critic  # noqa: E402
from whoearned.envs.lbf import LBFConfig, LBFEnv, ScriptedLBFTeam  # noqa: E402
from whoearned.metrics import pearson, top_k_selection  # noqa: E402

VARIANTS = {"base": dict(n_agents=4, n_food=3, size=8, horizon=50),
            "large": dict(n_agents=6, n_food=4, size=10, horizon=60),
            # time pressure: an absent agent costs food within the horizon (removal null is then informative)
            "h20": dict(n_agents=4, n_food=3, size=8, horizon=20),
            "large_h25": dict(n_agents=6, n_food=4, size=10, horizon=25),
            # heavy food (cap = sum of all levels): which agents are pivotal for which food is the credit question
            "heavy": dict(n_agents=4, n_food=3, size=8, horizon=50, food_cap="all"),
            "heavy_large": dict(n_agents=6, n_food=4, size=10, horizon=60, food_cap="all")}
N_TRAIN, N_TEST, EPS = 200, 30, 0.2
CFG = {"n_train": N_TRAIN, "n_int": (0, 200, 800), "steps": 3000, "null": "noop", "eps": EPS, "floor_draws": 16}


def episode(variant, seed, split, idx):
    sid = 0 if split == "train" else 1
    vid = sorted(VARIANTS).index(variant)
    cfg = LBFConfig(spawn_seed=(seed, sid, idx, vid), **VARIANTS[variant])
    team_seed = seed * 100_000 + sid * 50_000 + idx
    return cfg, (lambda: ScriptedLBFTeam(cfg, seed=team_seed, epsilon=CFG["eps"]))


def l1(credit, target):
    c, g = gt.normalise_shares(np.asarray(credit, float)), gt.normalise_shares(np.asarray(target, float))
    if c is None or g is None:
        return None
    return float(np.abs(c - g).sum())


def spurious(credit, target):
    c = gt.normalise_shares(np.asarray(credit, float))
    if c is None or np.abs(target).sum() < 1e-9:
        return None
    return float(sum(abs(c[i]) for i in range(len(c)) if abs(target[i]) < 1e-9))


def run(args):
    variant, seed, out_dir, cfg_over = args
    CFG.update(cfg_over)          # spawned workers re-import this module, so the config travels with the job
    out = os.path.join(out_dir, f"{variant}_s{seed}.json")
    if os.path.exists(out):
        return out
    t0 = time.time()
    n = VARIANTS[variant]["n_agents"]
    F = VARIANTS[variant]["n_food"]
    n_train, steps = CFG["n_train"], CFG["steps"]
    lo, mid, hi = CFG["n_int"]
    train = [episode(variant, seed, "train", i) for i in range(n_train)]
    facts = [ab.factual_rollout(cfg, mk, make_env=LBFEnv) for cfg, mk in train]
    tests = []
    for i in range(N_TEST):
        cfg, mk = episode(variant, seed, "test", i)
        f = ab.factual_rollout(cfg, mk, make_env=LBFEnv)
        W = ab.oracle_ablation_values(cfg, mk, f, null=CFG["null"])
        v0 = np.array([f.total if m == (1 << n) - 1 else
                       ab.continue_stopped(cfg, mk, f.snaps[0], tuple(j for j in range(n) if not m >> j & 1), CFG["null"])
                       for m in range(1 << n)])
        # luck floor: the expected LOO credit of this layout (mean over extra noise draws) as a prediction
        draws = []
        for d in range(CFG["floor_draws"]):
            mk_d = (lambda d=d: ScriptedLBFTeam(cfg, seed=7_000_000 + 1000 * i + d, epsilon=CFG["eps"]))
            fd = ab.factual_rollout(cfg, mk_d, make_env=LBFEnv)
            draws.append([fd.total - ab.continue_stopped(cfg, mk_d, fd.snaps[0], (j,), CFG["null"]) for j in range(n)])
        tests.append((cfg, f, W, v0, np.mean(draws, 0)))
    truth_s = time.time() - t0
    make_for = lambda i: train[i][1]
    cfg0 = train[0][0]
    data = {k: ab.intervention_dataset(cfg0, make_for, facts, k, seed=seed, null=CFG["null"], sampler="stratified")
            for k in (lo, mid, hi)}
    critics = {
        "mlp_noint": fit_mlp(data[lo].X, data[lo].y, seed=seed, steps=steps),
        "mlp": fit_mlp(data[mid].X, data[mid].y, seed=seed, steps=steps),
        "mlp_n800": fit_mlp(data[hi].X, data[hi].y, seed=seed, steps=steps),
        "set_noint": fit_two_set_critic(data[lo].X, data[lo].y, n, F, seed=seed, steps=steps),
        "set": fit_two_set_critic(data[mid].X, data[mid].y, n, F, seed=seed, steps=steps),
    }
    all_stopped = [tuple(j for j in range(n) if not m >> j & 1) for m in range(1 << n)]
    rec = {"variant": variant, "seed": seed, "truth_seconds": truth_s, "cfg": dict(CFG), "methods": {}, "truth": {
        "loo": [(f.total - W[:, 0]).tolist() for _, f, W, _, _ in tests], "v0": [v.tolist() for _, _, _, v, _ in tests],
        "returns": [f.total for _, f, _, _, _ in tests]}}
    readouts = [(name, "kappa") for name in critics] + [("set", "struct"), ("mlp", "struct")]
    for name, ro in readouts:
        pred = critics[name]
        rows = {"l1": [], "shap": [], "div": [], "spur": [], "topk": []}
        kh, kt = [], []
        for cfg, f, W, v0, _ in tests:
            Wh = ab.learned_ablation_values(pred, f, null=CFG["null"])
            if ro == "kappa":
                k = ab.ablation_td_credit(f.rewards, Wh)
            else:
                _, k = ab.commitment_split(f.rewards, ab.learned_value(pred, f), Wh)
            credit = k.sum(1)
            loo = f.total - W[:, 0]
            rows["l1"].append(l1(credit, loo))
            rows["spur"].append(spurious(credit, loo))
            matter = [i for i in range(n) if loo[i] > 1e-9]
            rows["topk"].append(top_k_selection(credit, matter) if matter else None)
            vh = pred(np.stack([ab.stopped_state(f.states[0], n, st, CFG["null"]) for st in all_stopped]))
            vh = vh - vh[0]
            rows["shap"].append(l1(gt.shapley_values(vh), gt.shapley_values(v0)))
            dt, de = gt.harsanyi_dividends(v0), gt.harsanyi_dividends(vh)
            st, se = np.abs(dt[1:]).sum(), np.abs(de[1:]).sum()
            rows["div"].append(float(np.abs(de[1:] / se - dt[1:] / st).sum()) if st > 1e-9 and se > 1e-9 else None)
            kh.append(k)
            kt.append(ab.ablation_td_credit(f.rewards, W))
        m = {key: (float(np.mean([v for v in vals if v is not None])) if any(v is not None for v in vals) else None)
             for key, vals in rows.items()}
        m["n_valid"] = sum(v is not None for v in rows["l1"])
        m["step_r"] = pearson(np.stack(kh), np.stack(kt))
        m["final_loss"] = pred.final_loss
        rec["methods"][f"{name}{'' if ro == 'kappa' else '_struct'}"] = m
    floor = []
    for cfg, f, W, v0, exp_loo in tests:
        floor.append(l1(exp_loo, f.total - W[:, 0]))
    rec["methods"]["floor_expected"] = {"l1": float(np.mean([x for x in floor if x is not None]))}
    uni = {"l1": [], "spur": [], "topk": [], "shap": []}
    for cfg, f, W, v0, _ in tests:
        loo = f.total - W[:, 0]
        uni["l1"].append(l1(np.ones(n), loo))
        uni["spur"].append(spurious(np.ones(n), loo))
        matter = [i for i in range(n) if loo[i] > 1e-9]
        uni["topk"].append(top_k_selection(np.ones(n), matter) if matter else None)
        uni["shap"].append(l1(np.ones(n), gt.shapley_values(v0)))
    rec["methods"]["uniform"] = {k: float(np.mean([v for v in vals if v is not None])) for k, vals in uni.items()}
    rec["seconds"] = time.time() - t0
    os.makedirs(out_dir, exist_ok=True)
    json.dump(rec, open(out, "w"))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variants", default="base")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--n-train", type=int, default=N_TRAIN)
    p.add_argument("--n-int", default="0,200,800", help="stops for: no-intervention, main, large (MLP) critics")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--null", default="noop", choices=("noop", "remove"))
    p.add_argument("--eps", type=float, default=EPS)
    a = p.parse_args()
    CFG.update(n_train=a.n_train, n_int=tuple(int(x) for x in a.n_int.split(",")), steps=a.steps, null=a.null,
               eps=a.eps)
    jobs = [(v, int(s), a.out, dict(CFG)) for v, s in itertools.product(a.variants.split(","), a.seeds.split(","))]
    with Pool(a.workers) as pool:
        for o in pool.imap_unordered(run, jobs):
            r = json.load(open(o))
            print(os.path.basename(o), f"{r['seconds']:.0f}s",
                  {k: (round(v["l1"], 3) if v.get("l1") is not None else None) for k, v in r["methods"].items()}, flush=True)


if __name__ == "__main__":
    main()
