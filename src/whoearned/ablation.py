"""Persistent-ablation credit: exact (oracle) and data for learned critics.

Definitions (docs/v3/theory.md has the derivations; docs/assessment_and_plan.md, Section 3):

* A *stop* is a persistent intervention: from step t on, the stopped agents take no-op
  (``null="noop"``, on the grid, teammates not told) or leave the grid (``null="remove"``).
* The *ablation value* ``W_S(s_t)`` is the (undiscounted) return-to-go from the factual state
  s_t when only the agents in S keep acting. ``W_N`` is the ordinary value.
* *Ablation-TD credit* of the stopped set C at step t:
      kappa_C(t) = r_t + W_{N-C}(s_{t+1}) - W_{N-C}(s_t).
  Identity 1: sum_t kappa_C(t) = G - W_{N-C}(s_0), the realised leave-C-out credit.
* *Commitment split*: kappa_i(t) = delta(t) + c_i(t) with the team TD error
      delta(t) = r_t + V(s_{t+1}) - V(s_t)            (shared surprise, identical for every agent)
      c_i(t)   = D_i(s_t) - D_i(s_{t+1}),  D_i = V - W_{N-i}   (change in the team's dependence on i)
  so sum_t c_i(t) = V(s_0) - W_{N-i}(s_0): expected leave-one-out credit.
* *Pending game* at step t: u_t(S) = W_S(s_t) - W_{empty}(s_t), the value that still needs S to
  keep acting. u_0 is the closed-loop coalition game, u_T = 0. Any linear credit rule phi
  (Shapley, Harsanyi dividends, leave-one-out) gives step credit phi(u_t) - phi(u_{t+1}) that sums
  over the episode to phi(u_0). Shapley step credit is efficient at every step: it splits the value
  *committed* at step t, u_t(N) - u_{t+1}(N), which is non-zero when an outcome becomes certain
  (the joint finish), not when its delayed payout arrives.

With the exact simulator and common random numbers, all of these are computed exactly for one
episode (``oracle_*``). A learned critic replaces W by W_hat(state with the agents' acting flags
switched off), queried on factual states only.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from whoearned import ground_truth as gt
from whoearned.envs.gridworld import GridConfig, PlantedCoalitionGridworld

NULLS = ("noop", "remove")
SAMPLERS = ("before_done", "through_payout", "stratified")

MakeTeam = Callable[[], object]   # fresh team with reset/act and get_rng_state/set_rng_state


# --- factual episodes and branching -----------------------------------------------------------------


@dataclass
class Factual:
    """One full-team episode with a snapshot before every step, so continuations can branch."""

    config: GridConfig
    states: np.ndarray            # (T + 1, ds): global state before each step, then the terminal state
    rewards: np.ndarray           # (T,)
    actions: np.ndarray           # (T, n)
    snaps: list = field(repr=False, default_factory=list)  # per step: (env clone, team RNG state)
    event_log: list = field(repr=False, default_factory=list)
    t_done: int = 0               # first step t at which every event is done before acting (T if never)
    t_last_payout: int = 0        # 1 + last step with a non-zero reward (0 if none)

    @property
    def total(self) -> float:
        return float(self.rewards.sum())

    @property
    def horizon(self) -> int:
        return len(self.rewards)

    def to_go(self) -> np.ndarray:
        """(T + 1,) undiscounted return-to-go; to_go[T] = 0."""
        return np.concatenate([np.cumsum(self.rewards[::-1])[::-1], [0.0]])


def factual_rollout(config: GridConfig, make_team: MakeTeam, keep_snaps: bool = True) -> Factual:
    env = PlantedCoalitionGridworld(config)
    env.reset()
    team = make_team()
    team.reset(env)
    T = config.horizon
    snaps, states, rewards, actions = [], [], np.zeros(T), np.zeros((T, config.n_agents), dtype=np.int64)
    t_done = T
    for t in range(T):
        if keep_snaps:
            snaps.append((env.clone(), team.get_rng_state()))
        states.append(env.get_state())
        if t_done == T and all(env.done_events):
            t_done = t
        actions[t] = team.act(env)
        _, rewards[t], _, _ = env.step(actions[t])
    states.append(env.get_state())
    nz = np.flatnonzero(rewards)
    return Factual(config, np.array(states), rewards, actions, snaps, env.event_log, t_done,
                   int(nz[-1]) + 1 if len(nz) else 0)


def continue_stopped(config: GridConfig, make_team: MakeTeam, snap, stopped, null: str = "noop",
                     record: bool = False):
    """Closed-loop continuation from ``snap`` with ``stopped`` agents stopped for good (common random numbers).

    Returns the undiscounted return-to-go, or with ``record`` a tuple (return, states, rewards) where
    states[k] is the global state before continuation step k (it shows the stop).
    """
    if null not in NULLS:
        raise ValueError(f"null must be one of {NULLS}")
    env0, rng_state = snap
    env = env0.clone()
    env.stop(stopped, null)
    team = make_team()
    team.reset(env)
    team.set_rng_state(rng_state)
    g, states, rewards = 0.0, [], []
    for _ in range(env.t, config.horizon):
        if not record and all(env.done_events):
            return g + sum(env._pending.values())         # only already-committed payouts remain
        if record:
            states.append(env.get_state())
        _, r, _, _ = env.step(team.act(env))
        g += r
        rewards.append(r)
    if record:
        return g, np.array(states), np.array(rewards)
    return g


def stopped_state(state: np.ndarray, n: int, stopped, null: str = "noop") -> np.ndarray:
    """The global state the environment would report right after ``env.stop(stopped, null)``."""
    s = np.array(state, copy=True)
    for i in stopped:
        s[..., 3 * n + i] = 0.0
        if null == "remove":
            s[..., 2 * i:2 * i + 2] = 0.0
            s[..., 2 * n + i] = 0.0
    return s


# --- exact (oracle) ablation values and credit ------------------------------------------------------


def oracle_ablation_values(config: GridConfig, make_team: MakeTeam, f: Factual, stopped_sets=None,
                           null: str = "noop") -> np.ndarray:
    """W[c, t], t = 0..T: return-to-go from factual s_t if the agents in stopped_sets[c] stop at t.

    Default ``stopped_sets``: each single agent (leave-one-out). W[c, T] = 0. Exact under common
    random numbers: this is the realised counterfactual for this episode's random draws.
    """
    n, T = config.n_agents, config.horizon
    stopped_sets = [(i,) for i in range(n)] if stopped_sets is None else [tuple(s) for s in stopped_sets]
    togo = f.to_go()
    W = np.zeros((len(stopped_sets), T + 1))
    for t in range(T):
        for c, stopped in enumerate(stopped_sets):
            if t >= f.t_done or not stopped:            # nothing outstanding / nobody stopped: factual
                W[c, t] = togo[t]
            else:
                W[c, t] = continue_stopped(config, make_team, f.snaps[t], stopped, null)
    return W


def ablation_td_credit(rewards: np.ndarray, W: np.ndarray) -> np.ndarray:
    """kappa[c, t] = r_t + W[c, t+1] - W[c, t]  (undiscounted)."""
    return rewards[None, :] + W[:, 1:] - W[:, :-1]


def commitment_split(rewards: np.ndarray, V: np.ndarray, W: np.ndarray):
    """Split ablation-TD credit into shared surprise and structural credit.

    ``V`` (T + 1,) is the full-team value along the trajectory, ``W`` (C, T + 1) the ablation values.
    Returns (delta (T,), c (C, T)) with kappa = delta[None] + c exactly.
    """
    delta = rewards + V[1:] - V[:-1]
    D = V[None, :] - W
    return delta, D[:, :-1] - D[:, 1:]


def all_masks_values(config: GridConfig, make_team: MakeTeam, f: Factual, null: str = "noop") -> np.ndarray:
    """Wall[mask, t]: return-to-go from s_t when only the agents in bitmask ``mask`` keep acting.

    Wall[full] is the factual return-to-go; Wall[:, 0] is the closed-loop coalition game v(S).
    Cost: 2^n continuations per step before t_done.
    """
    n = config.n_agents
    full = (1 << n) - 1
    stopped = [tuple(i for i in range(n) if not m >> i & 1) for m in range(1 << n)]
    W = oracle_ablation_values(config, make_team, f, stopped, null)
    assert np.allclose(W[full], f.to_go())
    return W


def pending_game(Wall: np.ndarray) -> np.ndarray:
    """u[mask, t] = W_mask(s_t) - W_empty(s_t): value that still needs ``mask`` to keep acting."""
    return Wall - Wall[0][None, :]


def step_semivalue_credit(Wall: np.ndarray):
    """Per-step Shapley, leave-one-out and dividend credit from the pending game.

    Returns dict with (T,)-indexed arrays:
      ``shapley`` (n, T): phi_i(u_t) - phi_i(u_{t+1}); sums over t to the closed-loop Shapley value.
      ``loo``     (n, T): leave-one-out analogue (= structural credit c_i of :func:`commitment_split`
                          when V and W come from the same oracle).
      ``dividends`` (2^n, T): Harsanyi dividend released at each step; sums to the closed-loop dividends.
      ``committed`` (T,): u_t(N) - u_{t+1}(N), the value committed at step t (= sum_i shapley[:, t]).
    """
    u = pending_game(Wall)
    T = u.shape[1] - 1
    phi = np.stack([gt.shapley_values(u[:, t]) for t in range(T + 1)], axis=1)        # (n, T+1)
    loo = np.stack([gt.leave_one_out(u[:, t]) for t in range(T + 1)], axis=1)
    div = np.stack([gt.harsanyi_dividends(u[:, t]) for t in range(T + 1)], axis=1)    # (2^n, T+1)
    full = u.shape[0] - 1
    return {
        "shapley": phi[:, :-1] - phi[:, 1:],
        "loo": loo[:, :-1] - loo[:, 1:],
        "dividends": div[:, :-1] - div[:, 1:],
        "committed": u[full, :-1] - u[full, 1:],
    }


# --- intervention data for learned critics -----------------------------------------------------------


@dataclass
class InterventionData:
    X: np.ndarray                 # (M, ds) global states (acting flags show stops)
    y: np.ndarray                 # (M,) undiscounted return-to-go
    episode: np.ndarray           # (M,) training-episode index each row comes from
    is_intervention: np.ndarray   # (M,) bool
    interventions: list           # (episode index, t0, stopped tuple, return-to-go at t0)
    info: dict


def sample_stopped(rng: np.random.Generator, n: int, single_prob: float = 0.5) -> tuple[int, ...]:
    """Half single agents; otherwise a random subset with keep-probability ~ U(0.1, 0.9) (every size)."""
    if rng.random() < single_prob:
        return (int(rng.integers(n)),)
    p = rng.uniform(0.1, 0.9)
    stopped = tuple(int(i) for i in np.flatnonzero(rng.random(n) < p))
    return stopped or (int(rng.integers(n)),)


def sample_stop_time(rng: np.random.Generator, f: Factual, sampler: str) -> int:
    """When the stop happens.

    ``before_done``: uniform over steps before every event is done (round-2 default).
    ``through_payout``: uniform up to the last payout, covering the 'committed, payout pending' wait.
    ``stratified``: half ``before_done``; half uniform between the last completion and the last payout
    (falls back to ``before_done`` when there is no such wait, e.g. d = 0 or nothing completed).
    """
    t_done = max(f.t_done, 1)
    if sampler == "before_done":
        return int(rng.integers(0, t_done))
    t_pay = max(f.t_last_payout, t_done)
    if sampler == "through_payout":
        return int(rng.integers(0, t_pay))
    if sampler == "stratified":
        if rng.random() < 0.5 or t_pay <= f.t_done:
            return int(rng.integers(0, t_done))
        return int(rng.integers(f.t_done, t_pay))
    raise ValueError(f"sampler must be one of {SAMPLERS}")


def intervention_dataset(config: GridConfig, make_team_for: Callable[[int], MakeTeam], facts: list[Factual],
                         n_int: int, seed: int, null: str = "noop", sampler: str = "stratified",
                         include_full: bool = True, single_prob: float = 0.5, plan=None) -> InterventionData:
    """Monte-Carlo regression data: the full-team episodes plus ``n_int`` stopped continuations.

    ``make_team_for(idx)`` returns the factory for training episode idx (its random streams).
    ``plan``: optional explicit list of (idx, t0, stopped) instead of random sampling (used by
    active designs and by the table baseline's 'all stops at t = 0' variant).
    """
    rng = np.random.default_rng([seed, 777])
    n = config.n_agents
    X, y, ep, isint, record = [], [], [], [], []
    if include_full:
        for idx, f in enumerate(facts):
            X.append(f.states[:-1])
            y.append(f.to_go()[:-1])
            ep.append(np.full(f.horizon, idx))
            isint.append(np.zeros(f.horizon, dtype=bool))
    if plan is None:
        plan = []
        for _ in range(n_int):
            idx = int(rng.integers(len(facts)))
            plan.append((idx, sample_stop_time(rng, facts[idx], sampler), sample_stopped(rng, n, single_prob)))
    n_zero = 0
    for idx, t0, stopped in plan:
        g, states, rewards = continue_stopped(config, make_team_for(idx), facts[idx].snaps[t0], stopped, null,
                                              record=True)
        togo = np.cumsum(rewards[::-1])[::-1]
        X.append(states)
        y.append(togo)
        ep.append(np.full(len(togo), idx))
        isint.append(np.ones(len(togo), dtype=bool))
        record.append((idx, t0, tuple(stopped), float(g)))
        n_zero += g == 0
    return InterventionData(np.concatenate(X).astype(np.float32), np.concatenate(y).astype(np.float32),
                            np.concatenate(ep), np.concatenate(isint), record,
                            {"n_int": len(plan), "int_return_zero": n_zero / max(len(plan), 1),
                             "null": null, "sampler": sampler})


# --- read-outs from a learned critic --------------------------------------------------------------------


def learned_ablation_values(predict: Callable[[np.ndarray], np.ndarray], f: Factual, stopped_sets=None,
                            null: str = "noop") -> np.ndarray:
    """W_hat[c, t] from factual states only: the critic at s_t with ``stopped_sets[c]`` switched off.

    W_hat[c, T] = 0 (terminal). ``predict`` maps (B, ds) states to (B,) return-to-go.
    """
    n, T = f.config.n_agents, f.horizon
    stopped_sets = [(i,) for i in range(n)] if stopped_sets is None else [tuple(s) for s in stopped_sets]
    W = np.zeros((len(stopped_sets), T + 1))
    for c, stopped in enumerate(stopped_sets):
        W[c, :T] = predict(stopped_state(f.states[:T], n, stopped, null))
    return W


def learned_value(predict, f: Factual) -> np.ndarray:
    """V_hat(s_t), t = 0..T (V_hat[T] = 0)."""
    return np.concatenate([predict(f.states[:-1]), [0.0]])


def learned_all_masks(predict, f: Factual, null: str = "noop") -> np.ndarray:
    n = f.config.n_agents
    stopped = [tuple(i for i in range(n) if not m >> i & 1) for m in range(1 << n)]
    return learned_ablation_values(predict, f, stopped, null)


# --- presence-only table (sparse Moebius regression) ------------------------------------------------------


def presence_features(masks: np.ndarray, n: int, k: int) -> tuple[np.ndarray, list[tuple[int, ...]]]:
    """[all of S present] for every coalition S with 1 <= |S| <= k."""
    subsets = [c for o in range(1, k + 1) for c in itertools.combinations(range(n), o)]
    masks = np.asarray(masks, dtype=np.int64)
    F = np.array([[float(all(m >> i & 1 for i in c)) for c in subsets] for m in masks])
    return F, subsets


def fit_table(masks: np.ndarray, returns: np.ndarray, n: int, k: int, alpha: float = 0.025) -> np.ndarray:
    """Lasso fit of return on presence monomials up to order k (no intercept). Returns dividends (2^n,)."""
    from sklearn.linear_model import Lasso
    F, subsets = presence_features(masks, n, k)
    beta = Lasso(alpha=alpha, fit_intercept=False, max_iter=50_000).fit(F, np.asarray(returns, float)).coef_
    div = np.zeros(1 << n)
    for c, b in zip(subsets, beta):
        div[gt.mask_of(c)] = b
    return div
