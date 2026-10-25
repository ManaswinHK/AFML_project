"""Rollouts, open-/closed-loop replay, and replay-based ground-truth credit.

See docs/environment_design.md, "Replay ground truth".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np

from whoearned import ground_truth as gt
from whoearned.envs.gridworld import (
    N_ACTIONS,
    NOOP,
    EventRecord,
    GridConfig,
    PlantedCoalitionGridworld,
)

NULL_POLICIES = ("remove", "noop", "random")


class Team(Protocol):
    def reset(self, env: PlantedCoalitionGridworld) -> None: ...
    def act(self, env: PlantedCoalitionGridworld) -> list[int]: ...


@dataclass
class Episode:
    config: GridConfig
    actions: np.ndarray        # (T, n) actions as sent to the environment
    rewards: np.ndarray        # (T,)
    event_log: list[EventRecord]
    present: int
    behaviour_probs: np.ndarray | None = None  # (T, n, A) if the team exposes `last_probs`

    @property
    def total(self) -> float:
        return float(self.rewards.sum())

    @property
    def event_payouts(self) -> np.ndarray:
        """Payout of each event in this episode (0 if it never completed)."""
        return np.array(
            [rec.payout if rec.completion_time is not None else 0.0 for rec in self.event_log]
        )


def _run(env: PlantedCoalitionGridworld, choose: Callable[[int], list[int]], present: int) -> Episode:
    cfg = env.config
    actions = np.zeros((cfg.horizon, cfg.n_agents), dtype=np.int64)
    rewards = np.zeros(cfg.horizon)
    for t in range(cfg.horizon):
        actions[t] = choose(t)
        _, rewards[t], done, _ = env.step(actions[t])
        if done:
            break
    return Episode(cfg, actions, rewards, env.event_log, present)


def rollout(config: GridConfig, team: Team, present: int | None = None) -> Episode:
    """Closed-loop episode: ``team`` chooses actions from the live environment.

    If the team sets ``last_probs`` on each ``act`` call (the scripted team does),
    the behaviour-policy probabilities are recorded in ``Episode.behaviour_probs``.
    """
    env = PlantedCoalitionGridworld(config)
    env.reset(present)
    team.reset(env)
    probs = []

    def choose(t):
        actions = team.act(env)
        if hasattr(team, "last_probs"):
            probs.append(np.array(team.last_probs))
        return actions

    ep = _run(env, choose, env.present_mask)
    if len(probs) == config.horizon:
        ep.behaviour_probs = np.stack(probs)
    return ep


def replay(
    config: GridConfig,
    actions: np.ndarray,
    present: int | None = None,
    null: str = "remove",
    null_seed: int = 0,
) -> Episode:
    """Open-loop replay of recorded ``actions`` with only ``present`` acting.

    Agents outside ``present`` follow the null policy: ``remove`` (off the grid),
    ``noop`` (on the grid, doing nothing) or ``random`` (on the grid, uniform
    random actions from a per-agent stream seeded by ``(null_seed, agent)``).
    """
    if null not in NULL_POLICIES:
        raise ValueError(f"null must be one of {NULL_POLICIES}, got {null!r}")
    n = config.n_agents
    full = (1 << n) - 1
    present = full if present is None else present & full
    actions = np.array(actions, dtype=np.int64, copy=True)
    absent = [i for i in range(n) if not present >> i & 1]
    if null == "noop":
        actions[:, absent] = NOOP
    elif null == "random":
        for i in absent:
            actions[:, i] = np.random.default_rng([null_seed, i]).integers(N_ACTIONS, size=len(actions))
    env = PlantedCoalitionGridworld(config)
    env.reset(present if null == "remove" else full)
    return _run(env, lambda t: actions[t], present)


@dataclass
class GroundTruth:
    """Ground-truth credit, in total and per event.

    ``v`` has shape (2**n,); ``v_events`` has shape (K, 2**n) with v = v_events.sum(0).
    Per-event arrays (``event_*``) have a leading axis of length K.
    """

    v: np.ndarray
    v_events: np.ndarray
    dividends: np.ndarray
    shapley: np.ndarray
    leave_one_out: np.ndarray
    event_dividends: np.ndarray
    event_shapley: np.ndarray
    event_leave_one_out: np.ndarray
    clipped: bool

    @classmethod
    def from_event_values(cls, v_events: np.ndarray, clipped: bool) -> GroundTruth:
        v_events = np.asarray(v_events, dtype=np.float64)
        v = v_events.sum(axis=0)
        return cls(
            v=v,
            v_events=v_events,
            dividends=gt.harsanyi_dividends(v),
            shapley=gt.shapley_values(v),
            leave_one_out=gt.leave_one_out(v),
            event_dividends=np.array([gt.harsanyi_dividends(x) for x in v_events]),
            event_shapley=np.array([gt.shapley_values(x) for x in v_events]),
            event_leave_one_out=np.array([gt.leave_one_out(x) for x in v_events]),
            clipped=clipped,
        )

    def nonzero_dividends(self, tol: float = 1e-9) -> dict[tuple[int, ...], float]:
        return gt.nonzero_dividends(self.v, tol)

    def matches(self, planted: dict[tuple[int, ...], float], tol: float = 1e-6) -> bool:
        """True if the realised dividends equal ``planted`` (see :func:`planted_dividends`)."""
        realised = self.nonzero_dividends(tol)
        return realised.keys() == planted.keys() and all(
            abs(realised[g] - planted[g]) <= tol for g in planted
        )


def planted_dividends(config: GridConfig) -> dict[tuple[int, ...], float]:
    """Dividends implied by the configuration alone, assuming every event completes.

    All-or-nothing event k: Delta(G_k) = r_k. Partial event k: each finisher f
    contributes r_k / |F_k| once all enablers are present, so
    Delta(E_k + {f}) = r_k / |F_k|. Compare with the replay ground truth to count
    episodes whose realised structure differs from the plan.
    """
    out: dict[tuple[int, ...], float] = {}
    for ev in config.events:
        if ev.partial:
            groups = [tuple(sorted({*ev.enablers, f})) for f in ev.finishers]
            amounts = [ev.reward / len(ev.finishers)] * len(groups)
        else:
            groups, amounts = [ev.coalition], [ev.reward]
        for g, r in zip(groups, amounts):
            out[g] = out.get(g, 0.0) + r
    return out


def _collect(config: GridConfig, episode_for: Callable[[int], Episode]) -> GroundTruth:
    n = config.n_agents
    v_events = np.zeros((len(config.events), 1 << n))
    clipped = False
    for s in range(1 << n):
        ep = episode_for(s)
        v_events[:, s] = ep.event_payouts
        if s == (1 << n) - 1:
            clipped = any(rec.clipped for rec in ep.event_log)
    return GroundTruth.from_event_values(v_events, clipped)


class _Intervention:
    """Wraps a team: agent ``agent`` takes ``action`` at step ``t0``; everything else is the team's."""

    def __init__(self, team: Team, t0: int, agent: int, action: int):
        self.team, self.t0, self.agent, self.action = team, t0, agent, action

    def reset(self, env):
        self.team.reset(env)

    def act(self, env):
        actions = list(self.team.act(env))
        if env.t == self.t0:
            actions[self.agent] = self.action
        return actions


def _snapshot(env: PlantedCoalitionGridworld):
    """Everything that determines the rest of an episode under a stateless, step-aligned team."""
    return (
        env.t,
        tuple(env.positions),
        tuple(tuple(p.values()) for p in env.pulled),
        tuple(env.done_events),
        tuple(sorted(env._pending.items())),
    )


@dataclass
class StepTruth:
    """Per-step counterfactual ground truth for one episode.

    ``returns[t, i, a]`` is the **discounted return from step t**,
    sum_{k >= t} gamma^(k-t) r_k, if agent i takes action a at step t and the team
    then continues with its own policy (closed loop, common random numbers). This
    is the quantity critics estimate. Entries for the action actually taken equal
    ``factual_to_go[t]``.

    Discounting matters: with a team that recovers from one-step deviations, the
    *undiscounted* return is often unchanged by any single action, so the whole
    per-step effect is the cost of delay under gamma.
    """

    returns: np.ndarray            # (T, n, A)
    factual_to_go: np.ndarray      # (T,) factual discounted return from t
    actions: np.ndarray            # (T, n) actions actually taken
    behaviour_probs: np.ndarray    # (T, n, A)
    gamma: float

    @property
    def factual(self) -> float:
        """Undiscounted factual return (for reference)."""
        return float(self._undiscounted)

    def noop_effect(self) -> np.ndarray:
        """G_t - G_t(i does noop at t): what the Q_tot-difference read-out (VDN/QMIX) estimates."""
        return self.factual_to_go[:, None] - self.returns[:, :, NOOP]

    def baseline_effect(self) -> np.ndarray:
        """G_t - sum_a pi(a) G_t(i does a at t): what COMA's counterfactual advantage estimates."""
        return self.factual_to_go[:, None] - (self.behaviour_probs * self.returns).sum(-1)


def step_counterfactuals(config: GridConfig, make_team: Callable[[], Team], gamma: float = 0.99) -> StepTruth:
    """Per-step counterfactual discounted returns for every agent, step and alternative action.

    The factual episode is rolled out once. For each (t, i, a) with a different from
    the action taken, the team is re-run with that single intervention. A
    counterfactual run stops early once its full state re-merges with the factual
    trajectory at the same step: from there on the outcome is identical, because the
    scripted team is stateless apart from its step-aligned random streams.
    """
    factual_env = PlantedCoalitionGridworld(config)
    factual_env.reset()
    team = make_team()
    team.reset(factual_env)
    T, n = config.horizon, config.n_agents
    actions = np.zeros((T, n), dtype=np.int64)
    probs = np.zeros((T, n, N_ACTIONS))
    snapshots = [_snapshot(factual_env)]
    rewards = np.zeros(T)
    for t in range(T):
        actions[t] = team.act(factual_env)
        probs[t] = team.last_probs
        _, rewards[t], _, _ = factual_env.step(actions[t])
        snapshots.append(_snapshot(factual_env))
    to_go = np.zeros(T + 1)  # to_go[t] = sum_{k >= t} gamma^(k-t) r_k; to_go[T] = 0
    for t in range(T - 1, -1, -1):
        to_go[t] = rewards[t] + gamma * to_go[t + 1]

    returns = np.repeat(to_go[:T, None, None], n, axis=1).repeat(N_ACTIONS, axis=2)
    for t in range(T):
        for i in range(n):
            if not factual_env.present[i]:
                continue
            for a in range(N_ACTIONS):
                if a != actions[t, i]:
                    returns[t, i, a] = _intervened_to_go(config, make_team, t, i, a, snapshots, to_go, gamma)
    truth = StepTruth(returns, to_go[:T], actions, probs, gamma)
    truth._undiscounted = rewards.sum()
    return truth


def _intervened_to_go(config, make_team, t0, agent, action, snapshots, to_go, gamma) -> float:
    """Discounted return from t0 when ``agent`` takes ``action`` at ``t0``."""
    env = PlantedCoalitionGridworld(config)
    env.reset()
    team = _Intervention(make_team(), t0, agent, action)
    team.reset(env)
    g, discount = 0.0, 1.0
    for t in range(config.horizon):
        _, r, _, _ = env.step(team.act(env))
        if t < t0:
            continue
        g += discount * r
        discount *= gamma
        if _snapshot(env) == snapshots[t + 1]:
            # Re-merged with the factual trajectory: the rest is the factual rest.
            return g + discount * to_go[t + 1]
    return g


def ground_truth_open_loop(
    config: GridConfig, actions: np.ndarray, null: str = "remove", null_seed: int = 0
) -> GroundTruth:
    """Default protocol: replay the recorded full-team actions for every subset."""
    return _collect(config, lambda s: replay(config, actions, s, null, null_seed))


def ground_truth_closed_loop(config: GridConfig, make_team: Callable[[], Team]) -> GroundTruth:
    """Sensitivity protocol: re-run a fresh team for every subset (true removal)."""
    return _collect(config, lambda s: rollout(config, make_team(), s))
