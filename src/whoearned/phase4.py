"""v3 Phase 4.1-4.3: does a critic trained under random agent dropout give persistent-ablation credit for free?

A MAPPO run trains with ``dropout_p`` (``PlantedEPyMARLEnv``): in a fraction of episodes some agents stop for
good at a random step and the global state's acting flags show it. Its centralised V(s) is then trained on
stopped states as a side effect of robustness training. For each saved checkpoint we:

1. roll out clean test episodes with the trained policy (no dropout; the policy's own sampling noise);
2. compute the **exact** persistent-ablation truth for *that* policy by branching it (env clone + policy RNG,
   RNN hidden state and batch buffer: :class:`BranchableMacTeam`), under the no-op null;
3. read out ablation-TD credit from the MAPPO critic at stopped states (masked read-out) and MAPPO's standard
   shared advantage, and score both against the exact truth.
"""

from __future__ import annotations

import copy

import numpy as np
import torch as th

from whoearned import ablation as ab
from whoearned.epymarl_bridge import build_harness, episodes_to_batch
from whoearned.experiments import masked_credit
from whoearned.metrics import evaluate_against, pearson
from whoearned.mode_t import MacTeam
from whoearned.readouts import episode_credit, shared_advantage
from whoearned.replay import Episode


class BranchableMacTeam(MacTeam):
    """A trained EPyMARL controller whose full internal state can be snapshotted and restored.

    The snapshot is (sampling RNG, RNN hidden state, per-episode batch buffer, step); with ``env.clone()`` it
    determines the rest of the episode, so stopped continuations share the factual episode's random draws.
    Sampling draws one uniform per agent per step whether or not the agent is stopped, so streams stay aligned.
    """

    def get_rng_state(self):
        hidden = self.h.mac.hidden_states
        return {"rng": copy.deepcopy(self.rng.bit_generator.state),
                "hidden": None if hidden is None else hidden.clone(),
                "batch": copy.deepcopy(self.batch), "t": self.t}

    def set_rng_state(self, st):
        self.rng.bit_generator.state = copy.deepcopy(st["rng"])
        self.h.mac.hidden_states = None if st["hidden"] is None else st["hidden"].clone()
        self.batch = copy.deepcopy(st["batch"])
        self.t = st["t"]


def lanes_env_args(K: int, horizon: int = 25, lane_length: int = 4, n_bystanders: int = 0) -> dict:
    return {"map_name": "multi_event_layout",
            "layout_kwargs": {"n_events": K, "delay": 0, "horizon": horizon, "lane_length": lane_length,
                              "n_bystanders": n_bystanders}}


@th.no_grad()
def evaluate_checkpoint(path: str, K: int, n_test: int = 30, seed: int = 0, epsilon: float = 0.0,
                        n_bystanders: int = 0) -> dict:
    h = build_harness("mappo", env_args=lanes_env_args(K, n_bystanders=n_bystanders), seed=seed)
    h.learner.load_models(path)
    cfg = h.env.config
    n = cfg.n_agents
    rows_masked, rows_std, kh, kt, returns = [], [], [], [], []
    facts, episodes, truths = [], [], []
    for s in range(n_test):
        make = lambda s=s: BranchableMacTeam(h, epsilon=epsilon, seed=10_000 * seed + s)
        f = ab.factual_rollout(cfg, make)
        W = ab.oracle_ablation_values(cfg, make, f, null="noop")
        facts.append(f)
        truths.append(W)
        episodes.append(Episode(cfg, f.actions, f.rewards, f.event_log, (1 << n) - 1))
        returns.append(f.total)
    batch = episodes_to_batch(h, episodes)
    masked = masked_credit(h, "mappo", batch, [cfg] * n_test, "noop")["per_step"]       # (B, T, n)
    std = shared_advantage(h, batch)
    enablers = sorted({e for ev in cfg.events for e in ev.enablers})
    bystanders = list(cfg.bystanders)
    uniform_rows = []
    for b, (f, W) in enumerate(zip(facts, truths)):
        k_true = ab.ablation_td_credit(f.rewards, W)          # (n, T)
        loo = k_true.sum(1)
        rows_masked.append(evaluate_against(masked[b].sum(0), loo, enablers, bystanders).as_dict())
        rows_std.append(evaluate_against(std[b].sum(0), loo, enablers, bystanders).as_dict())
        uniform_rows.append(evaluate_against(np.ones(n), loo, enablers, bystanders).as_dict())
        kh.append(masked[b].T)
        kt.append(k_true)

    def mean(rows, key):
        v = [r[key] for r in rows if r["valid"] and r[key] is not None]
        return float(np.mean(v)) if v else None
    return {"return_mean": float(np.mean(returns)), "success": float(np.mean(returns)) / (10.0 * K),
            "masked_l1": mean(rows_masked, "l1"), "standard_l1": mean(rows_std, "l1"),
            "uniform_l1": mean(uniform_rows, "l1"), "masked_leakage": mean(rows_masked, "leakage"),
            "masked_step_r": pearson(np.stack(kh), np.stack(kt)),
            "n_valid": sum(r["valid"] for r in rows_masked)}
