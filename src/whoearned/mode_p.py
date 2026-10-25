"""Mode P: train only the credit-bearing component on fixed scripted data.

- VDN / QMIX: the whole Q-learner (agent utilities + mixer) is the critic, so we call
  ``learner.train``. Note this learns Q* (greedy targets), not the value of the
  scripted policy; the two agree closely on the near-optimal scripted path.
- COMA: only the centralised Q critic (``_train_critic``), with EPyMARL's target updates.
- MAPPO: only the centralised V critic (``train_critic_sequential``), with its target updates.

Actors are never trained. Read-outs that need a policy (COMA baseline, MAPPO SNR)
use the scripted behaviour probabilities recorded in each episode.
"""

from __future__ import annotations

import numpy as np
import torch as th

from whoearned.epymarl_bridge import EpisodeBatch, Harness
from whoearned.readouts import (
    coma_credit,
    episode_credit,
    policy_gradient_snr,
    q_difference_credit,
    shared_advantage,
)
from whoearned.replay import Episode

Q_ALGS = ("vdn", "qmix")
ALGS = ("vdn", "qmix", "coma", "mappo")


def _masks(batch: EpisodeBatch):
    terminated = batch["terminated"][:, :-1].float()
    mask = batch["filled"][:, :-1].float()
    mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
    return terminated, mask


def _rewards(harness: Harness, batch: EpisodeBatch) -> th.Tensor:
    rewards = batch["reward"][:, :-1]
    if harness.args.standardise_rewards:
        harness.learner.rew_ms.update(rewards)
        rewards = (rewards - harness.learner.rew_ms.mean) / th.sqrt(harness.learner.rew_ms.var)
    return rewards


def _update_targets(learner, args):
    learner.critic_training_steps += 1
    tau = args.target_update_interval_or_tau
    if tau > 1 and (learner.critic_training_steps - learner.last_target_update_step) / tau >= 1.0:
        learner._update_targets_hard()
        learner.last_target_update_step = learner.critic_training_steps
    elif tau <= 1.0:
        learner._update_targets_soft(tau)


def train_step(harness: Harness, alg: str, batch: EpisodeBatch, step: int) -> float:
    """One critic update on ``batch``; returns the critic loss."""
    learner, args = harness.learner, harness.args
    if alg in Q_ALGS:
        before = len(harness.logger.stats.get("loss", []))
        args.learner_log_interval = 0  # log every step so we can return the loss
        learner.train(batch, t_env=step * batch.batch_size, episode_num=step * batch.batch_size)
        losses = harness.logger.stats["loss"]
        return float(losses[-1][1]) if len(losses) > before else float("nan")
    if alg == "coma":
        terminated, mask = _masks(batch)
        rewards = _rewards(harness, batch)
        _, log = learner._train_critic(
            batch, rewards, terminated, batch["actions"], batch["avail_actions"][:, :-1],
            mask.clone(), batch.batch_size, batch.max_seq_length,
        )
        _update_targets(learner, args)
        return float(log["critic_loss"][-1])
    if alg == "mappo":
        _, mask = _masks(batch)
        rewards = _rewards(harness, batch).expand(-1, -1, args.n_agents)
        mask = mask.repeat(1, 1, args.n_agents)
        _, log = learner.train_critic_sequential(learner.critic, learner.target_critic, batch, rewards, mask)
        _update_targets(learner, args)
        return float(log["critic_loss"][-1])
    raise ValueError(f"unknown algorithm {alg!r}")


def behaviour_probs(episodes: list[Episode]) -> np.ndarray:
    """Stack recorded behaviour probabilities to (B, T, n, A)."""
    if any(ep.behaviour_probs is None for ep in episodes):
        raise ValueError("episodes need behaviour_probs (roll out with the scripted team)")
    return np.stack([ep.behaviour_probs for ep in episodes])


def read_credit(harness: Harness, alg: str, batch: EpisodeBatch, episodes: list[Episode]) -> dict:
    """Per-step credit (B, T, n) for ``alg``; MAPPO also returns its SNR diagnostic."""
    if alg in Q_ALGS:
        return {"per_step": q_difference_credit(harness, batch)}
    pi = behaviour_probs(episodes)
    if alg == "coma":
        return {"per_step": coma_credit(harness, batch, pi=pi)}
    if alg == "mappo":
        adv = shared_advantage(harness, batch)
        return {"per_step": adv, "snr": policy_gradient_snr(harness, batch, adv, pi=pi)}
    raise ValueError(f"unknown algorithm {alg!r}")


__all__ = ["ALGS", "Q_ALGS", "behaviour_probs", "episode_credit", "read_credit", "train_step"]
