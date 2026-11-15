"""Data for the interactive credit visualisation (`docs/credit_viz.html`).

The stored Mode P cell JSONs keep only episode-level credit shares. The viewer also needs what happens
*inside* an episode: agent positions, event timings, each method's per-step credit (T, n) and the
per-step counterfactual truth. :func:`export_cell` regenerates all of it by re-running one stored cell
exactly as :func:`whoearned.experiments.run_data_cell` did (same data, same RNG stream, same training
order), and refuses to export unless the regenerated episode shares and removal truth reproduce the
stored JSON.
"""

from __future__ import annotations

import numpy as np
import torch as th

from whoearned.envs.gridworld import PlantedCoalitionGridworld
from whoearned.epymarl_bridge import episodes_to_batch
from whoearned.experiments import (
    MATCHED_TARGET,
    STEP_TARGET,
    DataCell,
    TrainSpec,
    _enabler_critical,
    _roles,
    _score,
    _test_truth,
    summarise,
)
from whoearned.ground_truth import agents_of, normalise_shares
from whoearned.metrics import evaluate_credit, pearson
from whoearned.mode_p import base_alg, make_harness, read_credit, train_step

SHARE_TOL = 1e-6


def _r(x, sig: int = 4):
    """Round floats (recursively) to ``sig`` significant figures to keep the page small."""
    if isinstance(x, np.ndarray):
        x = x.tolist()
    if isinstance(x, (list, tuple)):
        return [_r(v, sig) for v in x]
    if isinstance(x, dict):
        return {k: _r(v, sig) for k, v in x.items()}
    if isinstance(x, (float, np.floating)):
        return 0.0 if x == 0 else float(f"{float(x):.{sig}g}")
    if isinstance(x, np.integer):
        return int(x)
    return x


def layout_dict(cfg) -> dict:
    """Static layout of a GridConfig, for drawing."""
    enablers, bystanders = _roles(cfg)
    finishers = sorted({f for ev in cfg.events for f in ev.finishers})
    roles = ["enabler" if i in enablers else "finisher" if i in finishers else "bystander"
             for i in range(cfg.n_agents)]
    return {
        "n_agents": cfg.n_agents,
        "height": cfg.height,
        "width": cfg.width,
        "horizon": cfg.horizon,
        "delay": cfg.delay,
        "walls": sorted(list(w) for w in cfg.walls),
        "start_cells": [list(c) for c in cfg.start_cells],
        "roles": roles,
        "bystanders": list(bystanders),
        "events": [{
            "enablers": list(ev.enablers),
            "finishers": list(ev.finishers),
            "reward": ev.reward,
            "partial": ev.partial,
            "levers": {str(a): list(c) for a, c in ev.lever_cells.items()},
            "goals": {str(a): list(c) for a, c in ev.goal_cells.items()},
        } for ev in cfg.events],
    }


def trajectory(cfg, actions: np.ndarray) -> dict:
    """Replay recorded full-team actions and record what the viewer draws at each step.

    ``positions[t]`` is the grid *before* step t (so it has T + 1 entries); ``pulled[t][k]`` and
    ``done[t][k]`` are likewise before step t.
    """
    env = PlantedCoalitionGridworld(cfg)
    env.reset()

    def snap():
        return ([list(p) for p in env.positions],
                [[e for e, p in pulled.items() if p] for pulled in env.pulled],
                [bool(d) for d in env.done_events])

    positions, pulled, done = [], [], []
    rewards = []
    for t in range(cfg.horizon):
        p, u, d = snap()
        positions.append(p), pulled.append(u), done.append(d)
        _, r, _, _ = env.step(actions[t])
        rewards.append(r)
    p, u, d = snap()
    positions.append(p), pulled.append(u), done.append(d)
    events = [{
        "pull_times": {str(a): t for a, t in rec.pull_times.items()},
        "gate_open_time": rec.gate_open_time,
        "completion_time": rec.completion_time,
        "participants": list(rec.participants),
        "payout_time": rec.payout_time,
        "payout": rec.payout,
        "clipped": rec.clipped,
    } for rec in env.event_log]
    return {"positions": positions, "pulled": pulled, "done": done, "rewards": rewards, "events": events}


def _dividends(arr: np.ndarray, tol: float = 1e-6) -> list[list]:
    """Non-negligible dividends as [[agents...], value], largest |value| first."""
    out = [[list(agents_of(s)), float(arr[s])] for s in range(1, len(arr)) if abs(arr[s]) > tol]
    return sorted(out, key=lambda x: -abs(x[1]))


def export_cell(cell: DataCell, stored: dict, train: TrainSpec | None = None, episodes=None,
                l1_tol: float | None = None) -> dict:
    """Re-run ``cell`` and return the viewer's data for it.

    ``stored`` is the cell's JSON from `results/mode_p/main`. The regenerated removal truth must always
    match it exactly. Training is checked in one of two modes:

    - exact (``l1_tol=None``, the default): every algorithm's episode shares must match to 1e-6. This holds
      on the machine that produced ``stored``.
    - tolerant (``l1_tol=x``): PyTorch's CPU arithmetic is not bit-identical across machines, so training
      on another machine drifts away from the stored run (QMIX's loss is already ~0.2% off by update 100).
      Then the loss at update 0, computed before any parameter changes, must still match (same data,
      initialisation and batch order) and each algorithm's mean L1 must be within ``x`` of the stored one. The measured deviations are returned under ``"reproduction"`` so the page can show them.

    Raises ``RuntimeError`` on any failed check. ``episodes`` (indices) limits which test episodes are
    exported; all are regenerated, because training does not depend on them.
    """
    train = train or TrainSpec(**{k: tuple(v) if k == "algs" else v for k, v in stored["train"].items()})
    th.set_num_threads(1)
    cfg = cell.config()
    train_eps, test_eps = cell.episodes("train"), cell.episodes("test")
    truth = _test_truth(cell, test_eps)
    if not np.allclose(np.array(truth.loo), np.array(stored["truth"]["loo"]), atol=SHARE_TOL):
        raise RuntimeError(f"{cell.key}: regenerated leave-one-out truth differs from the stored cell")

    env_args = cell.env_args()
    rng = np.random.default_rng(cell.seed)
    algs, reproduction = {}, {"mode": "exact" if l1_tol is None else "tolerant", "algs": {}}
    for alg in train.algs:
        h = make_harness(alg, env_args=env_args, seed=cell.seed)
        data, test = episodes_to_batch(h, train_eps), episodes_to_batch(h, test_eps)
        losses = []
        for k in range(train.updates):
            idx = rng.choice(len(train_eps), train.batch_size, replace=False)
            loss = train_step(h, alg, data[idx], k)
            if k % train.log_every == 0 or k == train.updates - 1:
                losses.append(loss)
        out = read_credit(h, alg, test, test_eps)
        rows = _score(alg, out, truth, cfg)
        ref_rows = stored["algs"][alg]["episodes"]
        devs = []
        for b, (row, ref) in enumerate(zip(rows, ref_rows)):
            a, r = row["shares"], ref["shares"]
            if (a is None) != (r is None):
                if l1_tol is None:
                    raise RuntimeError(f"{cell.key}: {alg} episode {b} shares differ from the stored cell")
                continue
            if a is not None:
                devs.append(float(np.abs(np.subtract(a, r)).sum()))
        if l1_tol is None:
            if any(d > SHARE_TOL * len(cfg.start_cells) for d in devs):
                b = next(i for i, d in enumerate(devs) if d > SHARE_TOL * len(cfg.start_cells))
                raise RuntimeError(f"{cell.key}: {alg} episode {b} shares differ from the stored cell")
        else:
            ref_loss = stored["algs"][alg]["loss_trace"][0][1]
            if not np.allclose(losses[0], ref_loss, rtol=1e-5, equal_nan=True):
                raise RuntimeError(f"{cell.key}: {alg} early losses: update-0 loss {losses[0]} differs from stored {ref_loss}")
        new_l1 = summarise(rows)["l1"]
        old_l1 = stored["algs"][alg]["summary"]["l1"]
        if l1_tol is not None and new_l1 is not None and old_l1 is not None and abs(new_l1 - old_l1) > l1_tol:
            raise RuntimeError(f"{cell.key}: {alg} mean L1 {new_l1:.3f} vs stored {old_l1:.3f} (tolerance {l1_tol})")
        reproduction["algs"][alg] = {
            "episode_share_l1_mean": float(np.mean(devs)) if devs else None,
            "episode_share_l1_max": float(np.max(devs)) if devs else None,
            "mean_l1": new_l1, "stored_mean_l1": old_l1,
        }
        algs[alg] = {"out": out, "rows": rows}

    enablers, bystanders = _roles(cfg)
    n = cfg.n_agents
    keep = range(len(test_eps)) if episodes is None else episodes
    eps = []
    for b in keep:
        gt = truth.truths[b]
        uniform = {t: evaluate_credit(np.ones(n), gt, cfg, target=t).as_dict() for t in ("loo", "shapley")}
        step = {kind: truth.step[kind][b] for kind in ("noop", "baseline")}
        perfect = {}
        for kind, s in step.items():
            sh = normalise_shares(s.sum(0))
            perfect[kind] = {"shares": None if sh is None else sh.tolist(),
                             **{t: evaluate_credit(s.sum(0), gt, cfg, target=t).as_dict() for t in ("loo", "shapley")}}
        methods = {}
        for alg, res in algs.items():
            row = res["rows"][b]
            m = {
                "per_step": res["out"]["per_step"][b],
                "shares": row["shares"],
                "loo": row["loo"],
                "shapley": row["shapley"],
                "matched_target": MATCHED_TARGET[base_alg(alg)],
                "step_target": STEP_TARGET[base_alg(alg)],
            }
            if "fidelity" in row:
                m["fidelity"] = row["fidelity"]
            if "dividends" in res["out"]:
                m["dividends"] = _dividends(res["out"]["dividends"][b], tol=1e-3)
            methods[alg] = m
        eps.append({
            "index": b,
            "return": truth.returns[b],
            "matches_plan": bool(truth.matches_plan[b]),
            "enabler_critical": _enabler_critical(test_eps[b]),
            **trajectory(cfg, test_eps[b].actions),
            "truth": {
                "loo": gt.leave_one_out.tolist(),
                "shapley": gt.shapley.tolist(),
                "loo_shares": _shares(gt.leave_one_out),
                "shapley_shares": _shares(gt.shapley),
                "dividends": _dividends(gt.dividends),
                "step_noop": step["noop"],
                "step_baseline": step["baseline"],
            },
            "uniform": uniform,
            "perfect": perfect,
            "methods": methods,
        })
    summary = {}
    for alg, res in algs.items():
        rows = res["rows"]
        summary[alg] = {"summary": summarise(rows)}
        if STEP_TARGET[base_alg(alg)] is not None:
            fid = summarise(rows, "fidelity")
            fid["step_pearson"] = pearson(res["out"]["per_step"],
                                          np.stack(truth.step[STEP_TARGET[base_alg(alg)]]))
            summary[alg]["summary_fidelity"] = fid
    return _r({
        "key": cell.key,
        "cell": {"delay": cell.delay, "n_bystanders": cell.n_bystanders, "seed": cell.seed,
                 "enabler_start": cell.enabler_start, "horizon": cell.horizon, "layout": cell.layout},
        "layout": layout_dict(cfg),
        "enablers": enablers,
        "data": stored["data"],
        "semantic_gap": {k: {m: v[m] for m in ("l1", "leakage", "missed")} for k, v in stored["semantic_gap"].items()},
        "summary": summary,
        "reproduction": reproduction,
        "episodes": eps,
    })


def _shares(x):
    s = normalise_shares(x)
    return None if s is None else s.tolist()



# ---------------------------------------------------------------- our estimators (add-on stage)
#: Our methods shown next to VDN, QMIX, COMA and MAPPO: the round-2 coalition estimator (ECE-3, trained on
#: replays with random agents removed) and the v3 fix (credit by intervention: a set critic trained on episodes
#: where random agents stop for good; removal null, so "stopping" means leaving the grid, as in the truth).
ESTIMATOR_ALGS = ("ece3_mix", "atd_set_remove")


def add_estimators(cell: DataCell, data: dict, algs=ESTIMATOR_ALGS, train: TrainSpec | None = None) -> dict:
    """Train our estimators on ``cell`` and merge their per-episode credit into an exported cell ``data``.

    Leaves the four EPyMARL read-outs untouched, so this needs no bit-exact reproduction: the test episodes and
    their removal truth are deterministic (scripted team) and are checked against ``data`` before anything is
    merged. Also adds the closed-loop ("team reacts") removal truth, the target the v3 critic is built for.
    """
    from whoearned.experiments import ATD_ALGS, MIX, _train_facts, atd_credit

    th.set_num_threads(1)
    train = train or TrainSpec()
    cfg = cell.config()
    train_eps, test_eps = cell.episodes("train"), cell.episodes("test")
    nulls = tuple(dict.fromkeys(ATD_ALGS[a].null for a in algs if a in ATD_ALGS))
    truth = _test_truth(cell, test_eps, ablation_nulls=nulls)
    stored_loo = np.array([data["episodes"][k]["truth"]["loo"] for k in range(len(data["episodes"]))])
    idx = [e["index"] for e in data["episodes"]]
    if not np.allclose(np.array(truth.loo)[idx], stored_loo, rtol=1e-3, atol=1e-3):
        raise RuntimeError(f"{cell.key}: regenerated leave-one-out truth differs from the exported cell")

    env_args = cell.env_args()
    mix_eps = cell.episodes("mix") if any(a.endswith(MIX) for a in algs) else None
    train_facts = _train_facts(cell, train_eps) if any(a in ATD_ALGS for a in algs) else None
    results, seconds = {}, {}
    import time
    for alg in algs:
        t0 = time.time()
        if alg in ATD_ALGS:
            out = atd_credit(cell, ATD_ALGS[alg], train_facts, truth.facts, {})
            out.pop("info")
        else:
            rng = np.random.default_rng(cell.seed)
            h = make_harness(alg, env_args=env_args, seed=cell.seed)
            alg_train = mix_eps if alg.endswith(MIX) else train_eps
            batch_all, test = episodes_to_batch(h, alg_train), episodes_to_batch(h, test_eps)
            for k in range(train.updates):
                sel = rng.choice(len(alg_train), train.batch_size, replace=False)
                train_step(h, alg, batch_all[sel], k)
            out = read_credit(h, alg, test, test_eps)
        results[alg] = {"out": out, "rows": _score(alg, out, truth, cfg)}
        seconds[alg] = round(time.time() - t0, 1)

    n = cfg.n_agents
    for e in data["episodes"]:
        b = e["index"]
        closed = truth.closed[b]
        gt = truth.truths[b]
        e["truth"]["loo_closed"] = _r(closed.leave_one_out.tolist())
        e["truth"]["loo_closed_shares"] = _r(_shares(closed.leave_one_out))
        e["uniform"]["loo_closed"] = _r(evaluate_credit(np.ones(n), closed, cfg, target="loo").as_dict())
        for null in nulls:
            kind = f"kappa_{null}"
            s = truth.step[kind][b]
            e["truth"][f"step_{kind}"] = _r(s)
            e["perfect"][kind] = _r({"shares": _shares(s.sum(0)),
                                     **{t: evaluate_credit(s.sum(0), gt, cfg, target=t).as_dict() for t in ("loo", "shapley")},
                                     "loo_closed": evaluate_credit(s.sum(0), closed, cfg, target="loo").as_dict()})
        for alg, m in e["methods"].items():   # the four read-outs, scored against the new target from their shares
            if "loo_closed" not in m:
                m["loo_closed"] = (_r(evaluate_credit(np.array(m["shares"]), closed, cfg, target="loo").as_dict())
                                   if m.get("shares") is not None else {"valid": False})
        for alg, res in results.items():
            row = res["rows"][b]
            m = {"per_step": res["out"]["per_step"][b], "shares": row["shares"], "loo": row["loo"],
                 "shapley": row["shapley"], "loo_closed": row["loo_closed"],
                 "matched_target": MATCHED_TARGET[base_alg(alg)], "step_target": STEP_TARGET[base_alg(alg)],
                 "ours": True}
            if "fidelity" in row:
                m["fidelity"] = row["fidelity"]
            if "dividends" in res["out"]:
                m["dividends"] = _dividends(res["out"]["dividends"][b], tol=1e-3)
            e["methods"][alg] = _r(m)
    for alg, res in results.items():
        rows = [res["rows"][b] for b in idx]
        s = {"summary": summarise(rows), "summary_loo": summarise(rows, "loo"),
             "summary_loo_closed": summarise(rows, "loo_closed"), "ours": True, "train_seconds": seconds[alg]}
        if STEP_TARGET[base_alg(alg)] is not None:
            fid = summarise(rows, "fidelity")
            fid["step_pearson"] = pearson(res["out"]["per_step"], np.stack(truth.step[STEP_TARGET[base_alg(alg)]]))
            s["summary_fidelity"] = fid
        data["summary"][alg] = _r(s)
    for alg in list(data["summary"]):
        if alg in results:
            continue
        ms = [e["methods"][alg]["loo_closed"] for e in data["episodes"]]
        v = [x["l1"] for x in ms if x.get("valid")]
        data["summary"][alg]["summary_loo_closed"] = {"l1": _r(float(np.mean(v))) if v else None,
                                                       "n_valid": len(v), "n_total": len(ms)}
    data["estimators"] = {"algs": list(algs), "updates": train.updates, "seconds": seconds}
    return data
