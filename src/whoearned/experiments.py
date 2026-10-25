"""Mode P experiment cells.

One *data cell* = (delay, n_bystanders, seed). It generates scripted training and
test episodes and the test episodes' ground truth once, then trains and scores
every algorithm on that same data, so algorithms are compared on identical inputs.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch as th

from whoearned.envs import coalition_layout
from whoearned.epymarl_bridge import build_harness, episodes_to_batch
from whoearned.ground_truth import normalise_shares
from whoearned.metrics import evaluate_credit
from whoearned.mode_p import ALGS, episode_credit, read_credit, train_step
from whoearned.replay import ground_truth_open_loop, planted_dividends, rollout
from whoearned.scripted import ScriptedTeam

# Read-out -> matched ground-truth target (problem statement v2, Section 2.4).
MATCHED_TARGET = {"vdn": "loo", "qmix": "loo", "coma": "loo", "mappo": "loo"}


@dataclass(frozen=True)
class DataCell:
    delay: int
    n_bystanders: int
    seed: int
    horizon: int = 60
    n_train: int = 200
    n_test: int = 30
    epsilon: float = 0.2
    finish_not_before: int = 0

    @property
    def key(self) -> str:
        return f"d{self.delay}_b{self.n_bystanders}_s{self.seed}"

    def config(self):
        return coalition_layout(self.n_bystanders, delay=self.delay, horizon=self.horizon)

    def episodes(self, split: str):
        cfg = self.config()
        base = self.seed * 100_000 + (0 if split == "train" else 50_000)
        count = self.n_train if split == "train" else self.n_test
        return [
            rollout(cfg, ScriptedTeam(cfg, seed=base + s, epsilon=self.epsilon,
                                      finish_not_before=self.finish_not_before))
            for s in range(count)
        ]


@dataclass(frozen=True)
class TrainSpec:
    updates: int = 1500
    batch_size: int = 32
    log_every: int = 100
    algs: tuple[str, ...] = ALGS


@dataclass
class TestTruth:
    """Ground truth for each test episode, under the default and the no-op protocol."""

    shapley: list
    loo: list
    noop_shapley: list
    matches_plan: list
    clipped: list
    returns: list
    truths: list = field(default_factory=list, repr=False)


def _test_truth(cell: DataCell, test_eps) -> TestTruth:
    cfg = cell.config()
    planted = planted_dividends(cfg)
    out = TestTruth([], [], [], [], [], [])
    for ep in test_eps:
        truth = ground_truth_open_loop(cfg, ep.actions)
        noop = ground_truth_open_loop(cfg, ep.actions, null="noop")
        out.truths.append(truth)
        out.shapley.append(truth.shapley.tolist())
        out.loo.append(truth.leave_one_out.tolist())
        out.noop_shapley.append(noop.shapley.tolist())
        out.matches_plan.append(truth.matches(planted))
        out.clipped.append(truth.clipped)
        out.returns.append(ep.total)
    return out


def _score(alg: str, credit: np.ndarray, truth: TestTruth, cfg) -> list[dict]:
    rows = []
    for b, gt in enumerate(truth.truths):
        row = {"episode": b}
        for target in ("loo", "shapley"):
            rep = evaluate_credit(credit[b], gt, cfg, target=target)
            row[target] = rep.as_dict()
        shares = normalise_shares(credit[b])
        row["shares"] = None if shares is None else shares.tolist()
        row["matched"] = row[MATCHED_TARGET[alg]]
        rows.append(row)
    return rows


def _mean(values) -> float | None:
    vals = [v for v in values if v is not None]
    return float(np.mean(vals)) if vals else None


def summarise(rows: list[dict], key: str = "matched") -> dict:
    valid = [r[key] for r in rows if r[key]["valid"]]
    return {
        "n_valid": len(valid),
        "n_total": len(rows),
        **{m: _mean(r[m] for r in valid) for m in ("l1", "spearman", "leakage", "missed")},
    }


def run_data_cell(cell: DataCell, train: TrainSpec, out_dir: str) -> str:
    """Run every algorithm in ``train.algs`` on one data cell; write and return a JSON path."""
    th.set_num_threads(1)
    t0 = time.time()
    cfg = cell.config()
    train_eps, test_eps = cell.episodes("train"), cell.episodes("test")
    truth = _test_truth(cell, test_eps)
    record = {
        "cell": asdict(cell),
        "train": asdict(train),
        "data": {
            "train_return_mean": float(np.mean([e.total for e in train_eps])),
            "test_return_mean": float(np.mean(truth.returns)),
            "test_matches_plan": float(np.mean(truth.matches_plan)),
            "test_clipped": float(np.mean(truth.clipped)),
            "truth_seconds": time.time() - t0,
        },
        "h0": _h0(truth),
        "truth": {"shapley": truth.shapley, "loo": truth.loo},
        "algs": {},
    }
    env_args = {"map_name": "coalition_layout",
                "layout_kwargs": {"n_bystanders": cell.n_bystanders, "delay": cell.delay, "horizon": cell.horizon}}
    rng = np.random.default_rng(cell.seed)
    for alg in train.algs:
        ta = time.time()
        h = build_harness(alg, env_args=env_args, seed=cell.seed)
        data, test = episodes_to_batch(h, train_eps), episodes_to_batch(h, test_eps)
        losses = []
        for k in range(train.updates):
            idx = rng.choice(len(train_eps), train.batch_size, replace=False)
            loss = train_step(h, alg, data[idx], k)
            if k % train.log_every == 0 or k == train.updates - 1:
                losses.append((k, loss))
        out = read_credit(h, alg, test, test_eps)
        credit = episode_credit(out["per_step"])
        rows = _score(alg, credit, truth, cfg)
        rec = {
            "seconds": time.time() - ta,
            "loss_trace": losses,
            "summary": summarise(rows),
            "summary_loo": summarise(rows, "loo"),
            "summary_shapley": summarise(rows, "shapley"),
            "episodes": rows,
        }
        if "snr" in out:
            rec["snr"] = {k: v.tolist() for k, v in out["snr"].items()}
        record["algs"][alg] = rec
    record["seconds"] = time.time() - t0
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{cell.key}.json")
    with open(path + ".tmp", "w") as f:
        json.dump(record, f)
    os.replace(path + ".tmp", path)
    return path


def _h0(truth: TestTruth) -> dict:
    """Protocol sensitivity of the ground truth: L1 between Shapley shares under removal vs no-op."""
    diffs = []
    for a, b in zip(truth.shapley, truth.noop_shapley):
        sa, sb = normalise_shares(np.array(a)), normalise_shares(np.array(b))
        if sa is not None and sb is not None:
            diffs.append(float(np.abs(sa - sb).sum()))
    return {"remove_vs_noop_l1_mean": _mean(diffs), "remove_vs_noop_l1_max": max(diffs) if diffs else None}
