"""Per-cell pipeline: truth-level persistent credit (Test B) and the learned removal-conditioned
critic with its no-intervention ablation (Test C). Writes one JSON per cell.

Usage: python run_rcc.py <grid> <out_dir> [workers]
"""
from __future__ import annotations

import itertools
import json
import os
import sys
import time
import warnings
from multiprocessing import Pool

import numpy as np
from sklearn.neural_network import MLPRegressor

from rcc_lib import (GAMMA, Cell, closed_loop_values, make_team, persistent_credit, remove_from_state, run,
                     shares, sparse_dividends, to_go, training_set, true_persistent_values)
from whoearned import ground_truth as gt
from whoearned.metrics import (coalition_recovery, dividend_error, evaluate_against, identified_coalitions,
                               pearson)
from whoearned.replay import ground_truth_open_loop, planted_dividends

warnings.filterwarnings("ignore")
IDENTIFY_THRESHOLD = 0.1  # as in whoearned.experiments
SCALE = 10.0


def horizon_for(d):
    return 80 if d >= 50 else 60


GRIDS = {
    "main": [Cell(delay=d, n_bystanders=b, seed=s, enabler_start=es, horizon=horizon_for(d))
             for d, b, es, s in itertools.product((0, 25, 50), (2, 4), (0, 12), range(5))],
    "h3": [Cell(delay=d, n_bystanders=(1 if lay == "overlap" else 2), seed=s, horizon=horizon_for(d), layout=lay)
           for lay, d, s in itertools.product(("overlap", "partial"), (0, 25, 50), range(5))],
    "main_b2": [Cell(delay=d, n_bystanders=2, seed=s, enabler_start=es, horizon=horizon_for(d))
                for d, es, s in itertools.product((0, 25, 50), (0, 12), range(5))],
    "main_b4": [Cell(delay=d, n_bystanders=4, seed=s, enabler_start=es, horizon=horizon_for(d))
                for d, es, s in itertools.product((0, 25, 50), (0, 12), range(3))],
    "smoke": [Cell(delay=25, n_bystanders=2, seed=0), Cell(delay=0, n_bystanders=1, seed=0, layout="overlap")],
}


def fit(X, y2, seed):
    """y2 columns: discounted and undiscounted return-to-go."""
    m = MLPRegressor(hidden_layer_sizes=(128, 128), learning_rate_init=1e-3, batch_size=256,
                     max_iter=100, tol=0.0, n_iter_no_change=1000, random_state=seed)
    m.fit(X, y2 / SCALE)
    return m


def predict(m, X):
    return m.predict(X) * SCALE  # (N, 2)


def mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def rep(credit, target, enablers, bystanders):
    r = evaluate_against(np.asarray(credit), np.asarray(target), enablers, bystanders)
    return r.as_dict()


def div_dict(d):
    return {frozenset(gt.agents_of(s)): float(d[s]) for s in range(1, len(d)) if abs(d[s]) > 1e-9}


def coalition_scores(d, truth_div):
    est = {frozenset(gt.agents_of(s)): float(d[s]) for s in range(1, len(d))}
    tru = div_dict(truth_div)
    rec = coalition_recovery(identified_coalitions(est, IDENTIFY_THRESHOLD), list(tru))
    tot = sum(abs(x) for x in est.values())
    return {"dividend_error": dividend_error(est, tru), "recall": rec.recall, "precision": rec.precision,
            "interaction_mass": sum(abs(x) for g, x in est.items() if len(g) >= 2) / tot if tot else None}


def run_cell(args):
    cell, out_dir = args
    path = os.path.join(out_dir, cell.key + ".json")
    if os.path.exists(path):
        return path
    t0 = time.time()
    cfg = cell.config()
    n, T = cfg.n_agents, cfg.horizon
    full = (1 << n) - 1
    enablers = sorted({e for ev in cfg.events for e in ev.enablers})
    bystanders = list(cfg.bystanders)
    planted = planted_dividends(cfg)
    planted_v = np.zeros(1 << n)
    for s in range(1 << n):
        planted_v[s] = sum(val for g, val in planted.items() if all(s >> i & 1 for i in g))
    uni = np.full(n, 1.0 / n)

    # ---------- test episodes and all ground truths ----------
    test = []
    for e in range(cell.n_test):
        Vm, states, rewards, actions = true_persistent_values(cell, e)
        ol = ground_truth_open_loop(cfg, actions)                       # repo default truth
        vd, vu = closed_loop_values(cell, e)                            # repo sensitivity protocol
        test.append(dict(Vm=Vm, states=states, rewards=rewards, ol=ol, cl_v=vu,
                         cl_loo=gt.leave_one_out(vu), cl_shap=gt.shapley_values(vu)))
    truth_sec = time.time() - t0

    rec = {"cell": cell.__dict__ | {"key": cell.key}, "n_agents": n,
           "data": {"test_return": mean([t["rewards"].sum() for t in test]),
                    "matches_plan": mean([float(t["ol"].matches(planted)) for t in test])}}

    # ---------- baselines that need no model ----------
    def score_fixed(vec_fn):
        rows = {"loo": [], "shapley": []}
        for t in test:
            rows["loo"].append(rep(vec_fn(t), t["ol"].leave_one_out, enablers, bystanders))
            rows["shapley"].append(rep(vec_fn(t), t["ol"].shapley, enablers, bystanders))
        return {k: {m: mean([r[m] for r in v if r["valid"]]) for m in ("l1", "spearman", "leakage", "missed")}
                for k, v in rows.items()}

    rec["uniform"] = score_fixed(lambda t: uni)
    rec["planted_oracle"] = {
        "loo": {m: mean([rep(gt.leave_one_out(planted_v), t["ol"].leave_one_out, enablers, bystanders)[m] for t in test])
                for m in ("l1", "leakage", "missed")},
        "shapley": {m: mean([rep(gt.shapley_values(planted_v), t["ol"].shapley, enablers, bystanders)[m] for t in test])
                    for m in ("l1", "leakage", "missed")}}

    # ---------- Test B: exact persistent-removal credit vs the repo's removal truth ----------
    rows_sum, rows_tel, rows_cl = [], [], []
    for t in test:
        c = persistent_credit(t["Vm"], t["rewards"], cell.gamma)           # (T, n)
        rows_sum.append(rep(c.sum(0), t["ol"].leave_one_out, enablers, bystanders))        # unweighted, as level 2
        w = cell.gamma ** np.arange(T)
        rows_tel.append(rep((w[:, None] * c).sum(0), t["ol"].leave_one_out, enablers, bystanders))
        rows_cl.append(rep(t["cl_loo"], t["ol"].leave_one_out, enablers, bystanders))      # closed vs open loop
    summ = lambda rows: {m: mean([r[m] for r in rows if r["valid"]]) for m in ("l1", "spearman", "leakage", "missed")} | {
        "n_valid": sum(r["valid"] for r in rows)}
    rec["persistent_truth"] = {"sum": summ(rows_sum), "telescoped": summ(rows_tel), "closed_vs_open_loo": summ(rows_cl)}
    rec["persistent_truth"]["closed_vs_open_shapley_l1"] = mean(
        [rep(t["cl_shap"], t["ol"].shapley, enablers, bystanders)["l1"] for t in test])

    # ---------- Test C: learned critic, with and without removal interventions in its data ----------
    S_test = np.stack([t["states"] for t in test])                           # (E, T+1, S)
    Vm_true = np.stack([t["Vm"] for t in test])                              # (E, T+1, n)
    c_true = np.stack([persistent_credit(t["Vm"], t["rewards"], cell.gamma) for t in test])
    rec["rcc"] = {}
    for name, p_drop in (("rcc", 0.5), ("no_intervention", 0.0)):
        ta = time.time()
        X, Y, info = training_set(cell, p_drop=p_drop)
        m = fit(X, Y, cell.seed)
        # per-step masked values on test episodes
        Vhat = np.zeros_like(Vm_true)
        for i in range(n):
            q = remove_from_state(S_test[:, :T], n, (i,)).reshape(-1, S_test.shape[-1])
            Vhat[:, :T, i] = predict(m, q)[:, 0].reshape(len(test), T)
        c_hat = np.stack([persistent_credit(Vhat[e], test[e]["rewards"], cell.gamma) for e in range(len(test))])
        # s0 coalition game (undiscounted head)
        s0 = S_test[0, 0]
        q0 = np.stack([remove_from_state(s0, n, [i for i in range(n) if not s >> i & 1]) for s in range(1 << n)])
        v_hat = predict(m, q0)[:, 1]
        d_raw = gt.harsanyi_dividends(v_hat - v_hat[0])
        d_sp = sparse_dividends(v_hat, seed=cell.seed)
        v_sp = gt.values_from_dividends(d_sp)
        loo_rows, shap_rows, coal_rows, cl_rows, coal_sp, shap_sp, loo_sp = [], [], [], [], [], [], []
        for e, t in enumerate(test):
            G = t["rewards"].sum()
            loo_hat = G - v_hat[[full ^ (1 << i) for i in range(n)]]          # pathwise: actual return - predicted without i
            loo_rows.append(rep(loo_hat, t["ol"].leave_one_out, enablers, bystanders))
            cl_rows.append(rep(loo_hat, t["cl_loo"], enablers, bystanders))
            shap_rows.append(rep(gt.shapley_values(v_hat - v_hat[0]), t["ol"].shapley, enablers, bystanders))
            coal_rows.append(coalition_scores(d_raw, t["ol"].dividends))
            coal_sp.append(coalition_scores(d_sp, t["ol"].dividends))
            shap_sp.append(rep(gt.shapley_values(v_sp), t["ol"].shapley, enablers, bystanders))
            loo_sp.append(rep(G - v_sp[[full ^ (1 << i) for i in range(n)]], t["ol"].leave_one_out, enablers, bystanders))
        rec["rcc"][name] = {
            "train": info, "seconds": time.time() - ta,
            "fidelity_value_pearson": pearson(Vhat[:, :T], Vm_true[:, :T]),
            "fidelity_step_pearson": pearson(c_hat, c_true),
            "value_mae": float(np.abs(Vhat[:, :T] - Vm_true[:, :T]).mean()),
            "loo_vs_open": summ(loo_rows), "loo_vs_closed": summ(cl_rows), "shapley_vs_open": summ(shap_rows),
            "coalition": {k: mean([r[k] for r in coal_rows]) for k in coal_rows[0]},
            "coalition_sparse": {k: mean([r[k] for r in coal_sp]) for k in coal_sp[0]},
            "shapley_sparse_vs_open": summ(shap_sp), "loo_sparse_vs_open": summ(loo_sp),
            "dividends_sparse": {"".join("ABCDEFGH"[i] for i in gt.agents_of(s)): round(float(d_sp[s]), 3)
                                 for s in range(1, len(d_sp)) if abs(d_sp[s]) > 0.05},
            "v_hat": [float(x) for x in v_hat],
        }
    rec["truth"] = {"open_v": [t["ol"].v.tolist() for t in test], "closed_v": [t["cl_v"].tolist() for t in test],
                    "returns": [float(t["rewards"].sum()) for t in test]}
    rec["seconds"] = time.time() - t0
    rec["truth_seconds"] = truth_sec
    os.makedirs(out_dir, exist_ok=True)
    with open(path + ".tmp", "w") as f:
        json.dump(rec, f, default=lambda o: o if not isinstance(o, np.generic) else o.item())
    os.replace(path + ".tmp", path)
    return path


if __name__ == "__main__":
    grid, out_dir = sys.argv[1], sys.argv[2]
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    jobs = [(c, out_dir) for c in GRIDS[grid]]
    with Pool(workers) as p:
        for k, path in enumerate(p.imap_unordered(run_cell, jobs)):
            print(f"[{k + 1}/{len(jobs)}] {os.path.basename(path)}", flush=True)
