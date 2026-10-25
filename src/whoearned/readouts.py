"""Credit read-outs from trained EPyMARL learners (problem statement v2, Section 2.4).

Every read-out returns **per-step** credit of shape (B, T, n), so callers can
sum over the whole episode or over one event's time window. Compare:

- ``q_difference_credit`` (VDN, QMIX) and ``coma_credit`` with leave-one-out targets;
- ``shared_advantage`` (MAPPO) is identical for all agents by construction. It is a
  reference only; use ``policy_gradient_snr`` as MAPPO's diagnostic instead.
"""

from __future__ import annotations

import numpy as np
import torch as th

from whoearned.envs import NOOP
from whoearned.epymarl_bridge import EpisodeBatch, Harness


def _mac_out(mac, batch: EpisodeBatch) -> th.Tensor:
    """Agent outputs for t = 0..T-1, shape (B, T, n, A): Q-values or action probabilities."""
    outs = []
    mac.init_hidden(batch.batch_size)
    for t in range(batch.max_seq_length - 1):
        outs.append(mac.forward(batch, t=t))
    return th.stack(outs, dim=1)


@th.no_grad()
def q_difference_credit(harness: Harness, batch: EpisodeBatch, null_action: int = NOOP) -> np.ndarray:
    """Q_tot(s_t, a_t) - Q_tot(s_t, a0_i, a_-i,t) for each agent i, using the learned mixer.

    Offset-invariant (unlike raw utilities Q_i). For VDN it reduces to Q_i(a_i) - Q_i(a0).
    """
    learner = harness.learner
    if learner.mixer is None:
        raise ValueError("q_difference_credit needs a mixer (VDN or QMIX)")
    q_all = _mac_out(learner.mac, batch)                       # (B, T, n, A)
    actions = batch["actions"][:, :-1]                         # (B, T, n, 1)
    chosen = th.gather(q_all, dim=3, index=actions).squeeze(3)  # (B, T, n)
    null = q_all[..., null_action]                             # (B, T, n)
    states = batch["state"][:, :-1]
    q_tot = learner.mixer(chosen, states)                      # (B, T, 1)
    n = chosen.shape[2]
    credit = th.zeros_like(chosen)
    for i in range(n):
        swapped = chosen.clone()
        swapped[:, :, i] = null[:, :, i]
        credit[:, :, i] = (q_tot - learner.mixer(swapped, states)).squeeze(2)
    return credit.cpu().numpy()


def _policy(harness: Harness, batch: EpisodeBatch, pi: np.ndarray | None) -> th.Tensor:
    """Action probabilities (B, T, n, A): the given behaviour policy, else the learner's actor."""
    if pi is None:
        return _mac_out(harness.learner.mac, batch)
    return th.as_tensor(pi, dtype=th.float32, device=batch.device)


@th.no_grad()
def coma_credit(harness: Harness, batch: EpisodeBatch, pi: np.ndarray | None = None) -> np.ndarray:
    """COMA's counterfactual advantage Q(s, a) - sum_a' pi_i(a'|tau_i) Q(s, (a', a_-i)), per agent.

    ``pi`` (B, T, n, A) overrides the actor; in Mode P it is the scripted behaviour policy.
    """
    learner = harness.learner
    q_vals = learner.critic(batch)[:, :-1]                     # (B, T, n, A)
    pi = _policy(harness, batch, pi)
    actions = batch["actions"][:, :-1]
    q_taken = th.gather(q_vals, dim=3, index=actions).squeeze(3)
    baseline = (pi * q_vals).sum(-1)
    return (q_taken - baseline).cpu().numpy()


@th.no_grad()
def shared_advantage(harness: Harness, batch: EpisodeBatch) -> np.ndarray:
    """MAPPO's advantage: n-step return minus the centralised V(s), as in PPOLearner.

    Rewards are standardised with the learner's running statistics (without
    updating them) when ``standardise_rewards`` is on, because V was trained on
    that scale.
    """
    learner = harness.learner
    args = harness.args
    rewards = batch["reward"][:, :-1]
    if args.standardise_rewards:
        rewards = (rewards - learner.rew_ms.mean) / th.sqrt(learner.rew_ms.var)
    rewards = rewards.expand(-1, -1, args.n_agents)
    terminated = batch["terminated"][:, :-1].float()
    mask = batch["filled"][:, :-1].float()
    mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
    mask = mask.repeat(1, 1, args.n_agents)
    values = learner.critic(batch).squeeze(3)                  # (B, T+1, n)
    returns = learner.nstep_returns(rewards, mask, values, args.q_nstep)
    return ((returns - values[:, :-1]) * mask).cpu().numpy()


@th.no_grad()
def policy_gradient_snr(
    harness: Harness, batch: EpisodeBatch, advantages: np.ndarray, pi: np.ndarray | None = None
) -> dict[str, np.ndarray]:
    """Per-agent signal and noise of the policy gradient w.r.t. the action logits.

    For softmax policies d log pi(a)/d logits = onehot(a) - pi, so each episode's
    gradient for agent i is g_i = sum_t A_t,i (onehot(a_t,i) - pi_t,i). Across the
    B episodes we report the squared norm of the mean (signal), the total variance
    (noise), and signal / noise. A bystander under a shared advantage should have
    signal ~ 0 but non-zero noise. ``pi`` overrides the actor, as in :func:`coma_credit`.
    """
    pi = _policy(harness, batch, pi).cpu().numpy()            # (B, T, n, A)
    actions = batch["actions"][:, :-1, :, 0].cpu().numpy()     # (B, T, n)
    onehot = np.eye(pi.shape[-1])[actions]                     # (B, T, n, A)
    g = (advantages[..., None] * (onehot - pi)).sum(axis=1)   # (B, n, A)
    mean = g.mean(axis=0)
    signal = (mean ** 2).sum(-1)
    noise = g.var(axis=0, ddof=1).sum(-1) if g.shape[0] > 1 else np.full(g.shape[1], np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        snr = signal / noise
    return {"signal": signal, "noise": noise, "snr": snr}


def episode_credit(per_step: np.ndarray, start: int = 0, stop: int | None = None) -> np.ndarray:
    """Sum per-step credit (B, T, n) over t in [start, stop) -> (B, n)."""
    return per_step[:, start:stop].sum(axis=1)
