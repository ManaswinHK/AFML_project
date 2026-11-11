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

from whoearned import ablation as ab
from whoearned import ground_truth as gtm
from whoearned.critics import fit_mlp, fit_set_critic
from whoearned.envs import NOOP, coalition_layout, overlap_example
from whoearned.epymarl_bridge import episodes_to_batch
from whoearned.ground_truth import agents_of, mask_of, normalise_shares
from whoearned.metrics import (
    coalition_recovery,
    dividend_error,
    evaluate_against,
    evaluate_credit,
    identified_coalitions,
    pearson,
    top_k_selection,
    true_dividends,
)
from whoearned.mode_p import ALGS, MIX, base_alg, episode_credit, make_harness, read_credit, train_step
from whoearned.replay import (
    ground_truth_closed_loop,
    ground_truth_open_loop,
    planted_dividends,
    replay,
    rollout,
    step_counterfactuals,
)
from whoearned.scripted import ScriptedTeam

# Enabler start that puts A on the critical path in about half of the noisy (eps = 0.2)
# episodes, versus 3% with start 0. Measured before the main runs (docs/pilot_findings.md).
CRITICAL_ENABLER_START = 12

# Read-out -> matched removal target (problem statement v2, Section 2.4). STAS splits with
# Shapley values; NA²Q's individual credit is its mixer game's Shapley value.
MATCHED_TARGET = {"vdn": "loo", "qmix": "loo", "coma": "loo", "mappo": "loo", "na2q": "shapley", "stas": "shapley",
                  "ece2": "shapley", "ece3": "shapley"}
# Read-out -> the per-step counterfactual it estimates (two-level grading, docs/pilot_findings.md).
# None: the read-out estimates no single-agent per-step counterfactual, so it has no level-1 score.
STEP_TARGET = {"vdn": "noop", "qmix": "noop", "coma": "baseline", "mappo": "noop", "na2q": None, "stas": None,
               "ece2": None, "ece3": None}
# --- v3: methods computed directly from scripted data (no EPyMARL), scored like every other method ---
# Presence-only table (sparse Moebius regression) on the same ``mix`` data ECE-k_mix sees (Phase 0.4).
TABLE_ALGS = ("table2_mix", "table3_mix")
TABLE_ALPHA = 0.025  # ECE's 0.05 sparsity weight in scikit-learn's Lasso convention; not tuned


@dataclass(frozen=True)
class AtdSpec:
    """A persistent-ablation critic (``whoearned.ablation``): data protocol + critic + read-out."""
    n_int: int = 200                # stopped continuations added to the 200 full-team episodes
    null: str = "noop"
    sampler: str = "stratified"
    critic: str = "mlp"             # "mlp" or "set" (permutation-invariant)
    steps: int = 3000


ATD_ALGS = {
    "atd": AtdSpec(),                                   # primary (pre-registration v3)
    "atd_noint": AtdSpec(n_int=0),                      # control: same critic, no interventions
    "atd_remove": AtdSpec(null="remove"),
    "atd_bd": AtdSpec(sampler="before_done"),           # round-2 sampler (reproduction check, Phase 1.4)
    "atd_set": AtdSpec(critic="set"),
    **{f"atd_n{k}": AtdSpec(n_int=k) for k in (25, 50, 100, 400, 800)},   # data scaling (Phase 1.7)
    "atd_remove_n800": AtdSpec(n_int=800, null="remove"),
}
DIRECT_ALGS = TABLE_ALGS + tuple(ATD_ALGS)
for _a, _spec in ATD_ALGS.items():
    MATCHED_TARGET[_a] = "loo"
    STEP_TARGET[_a] = f"kappa_{_spec.null}"   # level 1: the exact ablation-TD credit under its own null
for _a in TABLE_ALGS:
    MATCHED_TARGET[base_alg(_a)] = "shapley"   # registered under the base name, like ece2/ece3
    STEP_TARGET[base_alg(_a)] = None

# A coalition counts as identified if it holds more than this share of |dividends| (H3).
IDENTIFY_THRESHOLD = 0.1
# Replay-varied training data (H5): in each extra replay, each agent is kept with this probability.
MIX_KEEP = 0.75


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
    # 0: the enabler has slack (pulls before finishers are ready). CRITICAL_ENABLER_START:
    # the enabler starts late, so the finishers wait at the gate (A on the critical path).
    enabler_start: int = 0
    gamma: float = 0.99  # EPyMARL's default; the per-step truth must use the critics' discount
    # "coalition": coalition_layout(n_bystanders). "partial": the same with partial synergy (the
    # event pays if any finisher interacts). "overlap": overlap_example (B in two events; n_bystanders ignored).
    layout: str = "coalition"
    # v3 Phase 2.1: a fresh random role assignment per episode (coalition layout only).
    shuffle_roles: bool = False

    @property
    def key(self) -> str:
        crit = f"_es{self.enabler_start}" if self.enabler_start else ""
        prefix = "" if self.layout == "coalition" else f"{self.layout}_"
        perm = "perm_" if self.shuffle_roles else ""
        return f"{perm}{prefix}d{self.delay}_b{self.n_bystanders}{crit}_s{self.seed}"

    def env_args(self) -> dict:
        """EPyMARL env_args that build the same configuration as :meth:`config`."""
        if self.layout == "overlap":
            return {"map_name": "overlap_example", "layout_kwargs": {"delay": self.delay, "horizon": self.horizon}}
        return {"map_name": "coalition_layout",
                "layout_kwargs": {"n_bystanders": self.n_bystanders, "delay": self.delay, "horizon": self.horizon,
                                  "partial": self.layout == "partial"}}

    def role_perm(self, split: str, index: int):
        """The role permutation of one episode (None for fixed roles). Seeded by (seed, split, index)."""
        if not self.shuffle_roles:
            return None
        rng = np.random.default_rng([self.seed, 0 if split == "train" else 1, index, 4242])
        return rng.permutation(3 + self.n_bystanders)

    def config_for(self, split: str, index: int):
        """Configuration of one episode: :meth:`config` unless roles are shuffled."""
        if not self.shuffle_roles:
            return self.config()
        if self.layout != "coalition":
            raise ValueError("shuffle_roles is defined for the coalition layout only")
        return coalition_layout(self.n_bystanders, delay=self.delay, horizon=self.horizon,
                                role_perm=self.role_perm(split, index))

    def config(self):
        if self.layout == "overlap":
            return overlap_example(delay=self.delay, horizon=self.horizon)
        if self.layout not in ("coalition", "partial"):
            raise ValueError(f"unknown layout {self.layout!r}")
        return coalition_layout(self.n_bystanders, delay=self.delay, horizon=self.horizon,
                                partial=self.layout == "partial")

    def team(self, split: str, index: int) -> ScriptedTeam:
        base = self.seed * 100_000 + (0 if split == "train" else 50_000)
        return ScriptedTeam(self.config_for(split, index), seed=base + index, epsilon=self.epsilon,
                            finish_not_before=self.finish_not_before,
                            enabler_start=self.enabler_start)

    def episodes(self, split: str):
        """``train`` / ``test``: full team. ``mix``: the training episodes plus one open-loop replay
        of each with a random subset removed (each agent kept with probability MIX_KEEP; removed
        agents' actions set to noop). Actions stay as recorded, as in the ground-truth replay."""
        cfg = self.config()
        if split == "mix":
            rng = np.random.default_rng(10_000 + self.seed)
            out = self.episodes("train")
            for ep in list(out):
                kept = mask_of(np.flatnonzero(rng.random(cfg.n_agents) < MIX_KEEP))
                rep = replay(ep.config, ep.actions, present=kept)
                rep.actions[:, [i for i in range(cfg.n_agents) if not kept >> i & 1]] = NOOP
                out.append(rep)
            return out
        count = self.n_train if split == "train" else self.n_test
        return [rollout(self.config_for(split, s), self.team(split, s)) for s in range(count)]


@dataclass(frozen=True)
class TrainSpec:
    updates: int = 1500
    batch_size: int = 32
    log_every: int = 100
    algs: tuple[str, ...] = ALGS


@dataclass
class TestTruth:
    """Ground truth for each test episode.

    Removal level: Shapley / leave-one-out under the default protocol, and Shapley
    under no-op (for H0). Step level: per-step counterfactual effects (T, n) of
    replacing one action with noop ("noop") or with the behaviour-policy average
    ("baseline").
    """

    shapley: list
    loo: list
    noop_shapley: list
    matches_plan: list
    clipped: list
    returns: list
    truths: list = field(default_factory=list, repr=False)
    step: dict = field(default_factory=lambda: {"noop": [], "baseline": []}, repr=False)
    # v3: closed-loop (re-run) ground truth, the recommended default (Phase 0.6, 0.8)
    closed: list = field(default_factory=list, repr=False)
    # v3: factual test episodes with snapshots, and exact ablation values W[null] (n, T+1) per episode
    facts: list = field(default_factory=list, repr=False)
    ablation_W: dict = field(default_factory=dict, repr=False)
    configs: list = field(default_factory=list, repr=False)   # per test episode (differs if roles are shuffled)


def _test_truth(cell: DataCell, test_eps, ablation_nulls=()) -> TestTruth:
    out = TestTruth([], [], [], [], [], [])
    for null in ablation_nulls:
        out.ablation_W[null] = []
        out.step[f"kappa_{null}"] = []
    for s, ep in enumerate(test_eps):
        cfg = cell.config_for("test", s)
        planted = planted_dividends(cfg)
        out.configs.append(cfg)
        make = lambda s=s: cell.team("test", s)
        out.closed.append(ground_truth_closed_loop(cfg, make))
        if ablation_nulls:
            f = ab.factual_rollout(cfg, make)
            if not np.array_equal(f.actions, ep.actions):
                raise RuntimeError(f"factual rollout of test episode {s} did not reproduce its actions")
            out.facts.append(f)
            for null in ablation_nulls:
                W = ab.oracle_ablation_values(cfg, make, f, null=null)
                out.ablation_W[null].append(W)
                out.step[f"kappa_{null}"].append(ab.ablation_td_credit(f.rewards, W).T)
        truth = ground_truth_open_loop(cfg, ep.actions)
        noop = ground_truth_open_loop(cfg, ep.actions, null="noop")
        step = step_counterfactuals(cfg, lambda: cell.team("test", s), gamma=cell.gamma)
        if not np.array_equal(step.actions, ep.actions):
            raise RuntimeError(f"step truth for test episode {s} did not reproduce its actions")
        out.truths.append(truth)
        out.shapley.append(truth.shapley.tolist())
        out.loo.append(truth.leave_one_out.tolist())
        out.noop_shapley.append(noop.shapley.tolist())
        out.matches_plan.append(truth.matches(planted))
        out.clipped.append(truth.clipped)
        out.returns.append(ep.total)
        out.step["noop"].append(step.noop_effect())
        out.step["baseline"].append(step.baseline_effect())
    return out


def _roles(cfg):
    enablers = sorted({e for ev in cfg.events for e in ev.enablers})
    return enablers, list(cfg.bystanders)


def coalition_scores(dividends: np.ndarray, gt) -> dict:
    """Coalition level (H3): one episode's estimated dividends (2^n,) vs the true dividends."""
    est = {frozenset(agents_of(s)): float(dividends[s]) for s in range(1, len(dividends))}
    tru = true_dividends(gt)
    rec = coalition_recovery(identified_coalitions(est, IDENTIFY_THRESHOLD), list(tru))
    total = sum(abs(v) for v in est.values())
    return {
        "dividend_error": dividend_error(est, tru),
        "dividend_error_order2": dividend_error(est, tru, max_order=2),
        "recall": rec.recall,
        "precision": rec.precision,
        # Fraction of |dividend| mass on coalitions of 2+ agents. It discriminates even when the
        # dividend error saturates at 2 (e.g. a near-additive mixer against purely coalitional truth).
        "interaction_mass": sum(abs(v) for g, v in est.items() if len(g) >= 2) / total if total else None,
        "true_interaction_mass": sum(abs(v) for g, v in tru.items() if len(g) >= 2) / sum(abs(v) for v in tru.values())
        if tru else None,
        "dividends": [float(x) for x in dividends],
    }


def _score(alg: str, out: dict, truth: TestTruth, cfg) -> list[dict]:
    """Per test episode: end-to-end (read-out vs removal), level 1 (read-out vs per-step truth)
    where the read-out has a per-step target, and coalition level where it has dividends."""
    per_step = out["per_step"]
    credit = episode_credit(per_step)
    rows = []
    for b, gt in enumerate(truth.truths):
        cfg = truth.configs[b]
        enablers, bystanders = _roles(cfg)
        coalition = sorted({a for ev in cfg.events for a in ev.coalition})
        row = {"episode": b}
        for target in ("loo", "shapley"):
            row[target] = evaluate_credit(credit[b], gt, cfg, target=target).as_dict()
            if truth.closed:
                row[f"{target}_closed"] = evaluate_credit(credit[b], truth.closed[b], cfg, target=target).as_dict()
        row["top_k"] = top_k_selection(credit[b], coalition)
        row["matched"] = row[MATCHED_TARGET[base_alg(alg)]]
        if STEP_TARGET[base_alg(alg)] is not None:
            step_truth = truth.step[STEP_TARGET[base_alg(alg)]][b]
            row["fidelity"] = evaluate_against(credit[b], step_truth.sum(0), enablers, bystanders).as_dict()
            row["fidelity"]["step_pearson"] = pearson(per_step[b], step_truth)
        if "dividends" in out:
            row["coalition"] = coalition_scores(out["dividends"][b], gt)
            if truth.closed:
                row["coalition_closed"] = coalition_scores(out["dividends"][b], truth.closed[b])
        shares = normalise_shares(credit[b])
        row["shares"] = None if shares is None else shares.tolist()
        rows.append(row)
    return rows


def semantic_gap(truth: TestTruth, cfg) -> dict:
    """Level 2 (method-independent): a *perfect* per-step critic's credit vs removal credit."""
    out = {}
    for kind in ("noop", "baseline"):
        rows = []
        for b, gt in enumerate(truth.truths):
            enablers, bystanders = _roles(truth.configs[b])
            rep = evaluate_against(truth.step[kind][b].sum(0), gt.leave_one_out, enablers, bystanders)
            rows.append({"matched": rep.as_dict()})
        out[kind] = summarise(rows)
        out[kind]["mean_shares"] = _mean_shares(truth.step[kind])
    return out


def _mean_shares(step_list) -> list | None:
    shares = [normalise_shares(s.sum(0)) for s in step_list]
    shares = [s for s in shares if s is not None]
    return np.mean(shares, axis=0).tolist() if shares else None


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
    nulls = tuple(dict.fromkeys(ATD_ALGS[a].null for a in train.algs if a in ATD_ALGS))
    truth = _test_truth(cell, test_eps, ablation_nulls=nulls)
    record = {
        "cell": asdict(cell),
        "train": asdict(train),
        "data": {
            "train_return_mean": float(np.mean([e.total for e in train_eps])),
            "test_return_mean": float(np.mean(truth.returns)),
            "test_matches_plan": float(np.mean(truth.matches_plan)),
            "test_clipped": float(np.mean(truth.clipped)),
            "test_enabler_critical": float(np.mean([_enabler_critical(e) for e in test_eps])),
            "truth_seconds": time.time() - t0,
        },
        "h0": _h0(truth),
        "semantic_gap": semantic_gap(truth, cfg),
        "truth": {"shapley": truth.shapley, "loo": truth.loo},
        "algs": {},
    }
    env_args = cell.env_args()
    rng = np.random.default_rng(cell.seed)
    mix_eps = cell.episodes("mix") if any(a.endswith(MIX) for a in train.algs) else None  # tables are *_mix too
    if mix_eps is not None:
        record["data"]["mix_return_mean"] = float(np.mean([e.total for e in mix_eps]))
        record["data"]["mix_return_zero"] = float(np.mean([e.total == 0 for e in mix_eps]))
    train_facts = None
    for alg in train.algs:
        ta = time.time()
        h, losses, extra = None, [], {}
        if alg in TABLE_ALGS:
            out = table_credit(cfg, mix_eps, k=int(alg[5]), n_test=len(test_eps))
        elif alg in ATD_ALGS:
            if train_facts is None:
                train_facts = _train_facts(cell, train_eps)
            out = atd_credit(cell, ATD_ALGS[alg], train_facts, truth.facts)
            extra = out.pop("info")
        else:
            h = make_harness(alg, env_args=env_args, seed=cell.seed)
            alg_train = mix_eps if alg.endswith(MIX) else train_eps
            data, test = episodes_to_batch(h, alg_train), episodes_to_batch(h, test_eps)
            for k in range(train.updates):
                idx = rng.choice(len(alg_train), train.batch_size, replace=False)
                loss = train_step(h, alg, data[idx], k)
                if k % train.log_every == 0 or k == train.updates - 1:
                    losses.append((k, loss))
            out = read_credit(h, alg, test, test_eps)
        rows = _score(alg, out, truth, cfg)
        rec = {
            "seconds": time.time() - ta,
            "loss_trace": losses,
            "summary": summarise(rows),
            "summary_loo": summarise(rows, "loo"),
            "summary_shapley": summarise(rows, "shapley"),
            "summary_loo_closed": summarise(rows, "loo_closed"),
            "summary_shapley_closed": summarise(rows, "shapley_closed"),
            "top_k": _mean(r["top_k"] for r in rows),
            "episodes": rows,
            **extra,
        }
        if STEP_TARGET[base_alg(alg)] is not None:
            fidelity = summarise(rows, "fidelity")
            # Pooled over every (episode, step, agent): is the critic right about the per-step effect?
            fidelity["step_pearson"] = pearson(out["per_step"], np.stack(truth.step[STEP_TARGET[base_alg(alg)]]))
            rec["summary_fidelity"] = fidelity
        if "dividends" in out:
            rec["summary_coalition"] = {k: _mean(r["coalition"][k] for r in rows)
                                        for k in rows[0]["coalition"] if k != "dividends"}
            if "coalition_closed" in rows[0]:
                rec["summary_coalition_closed"] = {k: _mean(r["coalition_closed"][k] for r in rows)
                                                   for k in rows[0]["coalition_closed"] if k != "dividends"}
        if h is not None and hasattr(h.learner, "n_parameters"):
            rec["n_parameters"] = h.learner.n_parameters()
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


def _train_facts(cell: DataCell, train_eps) -> list:
    """Training episodes re-run with a snapshot before every step (identical actions, checked)."""
    facts = []
    for i, ep in enumerate(train_eps):
        f = ab.factual_rollout(cell.config_for("train", i), lambda i=i: cell.team("train", i))
        if not np.array_equal(f.actions, ep.actions):
            raise RuntimeError(f"factual rollout of training episode {i} did not reproduce its actions")
        facts.append(f)
    return facts


def table_credit(cfg, mix_eps, k: int, n_test: int) -> dict:
    """Presence-only table on the ``mix`` data: Lasso of return on [all of S present], |S| <= k.

    It never looks at a trajectory, so its dividends and its Shapley split are the same for every test
    episode. Credit is placed at step 0 (it has no temporal read-out).
    """
    n = cfg.n_agents
    div = ab.fit_table([ep.present for ep in mix_eps], [ep.total for ep in mix_eps], n, k, alpha=TABLE_ALPHA)
    phi = np.zeros(n)
    for s in range(1, 1 << n):
        members = gtm.agents_of(s)
        phi[list(members)] += div[s] / len(members)
    per_step = np.zeros((n_test, cfg.horizon, n))
    per_step[:, 0] = phi
    return {"per_step": per_step, "dividends": np.repeat(div[None], n_test, 0)}


def atd_credit(cell: DataCell, spec: AtdSpec, train_facts: list, test_facts: list) -> dict:
    """Persistent-ablation critic: intervention data -> value critic on the global state -> read-out.

    Read-out uses factual test states only (no interventions at test time): the critic at s_t with agent
    i's acting flag switched off gives W_hat_{-i}(s_t); per-step credit is the ablation-TD credit.
    Coalition level: the dividends of u(S) = W_hat_S(s_0) - W_hat_empty(s_0), per test episode.
    """
    cfg = cell.config()
    n, k = cfg.n_agents, len(cfg.events)
    data = ab.intervention_dataset(cfg, lambda i: (lambda: cell.team("train", i)), train_facts, spec.n_int,
                                   seed=cell.seed, null=spec.null, sampler=spec.sampler)
    if spec.critic == "mlp":
        predict = fit_mlp(data.X, data.y, seed=cell.seed, steps=spec.steps)
    elif spec.critic == "set":
        predict = fit_set_critic(data.X, data.y, n, k, seed=cell.seed, steps=spec.steps)
    else:
        raise ValueError(f"unknown critic {spec.critic!r}")
    per_step, dividends = [], []
    all_stopped = [tuple(i for i in range(n) if not m >> i & 1) for m in range(1 << n)]
    for f in test_facts:
        W = ab.learned_ablation_values(predict, f, null=spec.null)
        per_step.append(ab.ablation_td_credit(f.rewards, W).T)
        v0 = predict(np.stack([ab.stopped_state(f.states[0], n, st, spec.null) for st in all_stopped]))
        dividends.append(gtm.harsanyi_dividends(v0 - v0[0]))
    return {"per_step": np.stack(per_step), "dividends": np.stack(dividends),
            "info": {"atd": asdict(spec), "data": data.info, "n_rows": int(len(data.y)),
                     "final_loss": predict.final_loss, "n_parameters": predict.n_parameters}}


def _enabler_critical(ep) -> bool:
    """The event completed in the first step its gate allowed: the enabler was the bottleneck."""
    rec = ep.event_log[0]
    return rec.completion_time is not None and rec.completion_time == rec.gate_open_time + 1


def _h0(truth: TestTruth) -> dict:
    """Protocol sensitivity of the ground truth: L1 between Shapley shares under removal vs no-op."""
    diffs = []
    for a, b in zip(truth.shapley, truth.noop_shapley):
        sa, sb = normalise_shares(np.array(a)), normalise_shares(np.array(b))
        if sa is not None and sb is not None:
            diffs.append(float(np.abs(sa - sb).sum()))
    return {"remove_vs_noop_l1_mean": _mean(diffs), "remove_vs_noop_l1_max": max(diffs) if diffs else None}
