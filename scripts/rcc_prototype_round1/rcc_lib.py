"""Removal-conditioned critic (RCC) prototype on the team's Planted Coalition Gridworld.

Nothing in the repository is modified. This file adds:
  * mid-episode agent removal (drop an agent at step t and keep going),
  * a scripted team that treats an already-pulled lever as banked (needed for mid-episode removal),
  * data generation with removal interventions,
  * exact persistent-removal values V^{-i}(t) from the simulator (ground truth for the new read-out),
  * a value regressor on (global state incl. presence flags) -> discounted return-to-go.
"""
from __future__ import annotations

import os
import sys

import numpy as np

# Locate the repository: WHOEARNED_REPO, or two levels up when this folder sits in <repo>/experiments/.
_HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("WHOEARNED_REPO") or os.path.abspath(os.path.join(_HERE, "..", ".."))
if not os.path.isdir(os.path.join(REPO, "src", "whoearned")):
    raise RuntimeError("Set WHOEARNED_REPO to the AFML_project checkout (the folder that contains src/whoearned).")
sys.path.insert(0, os.path.join(REPO, "src"))

from whoearned import ground_truth as gt  # noqa: E402
from whoearned.envs.gridworld import PlantedCoalitionGridworld  # noqa: E402
from whoearned.scripted import ScriptedTeam  # noqa: E402

GAMMA = 0.99
AUG_PENDING_AMOUNT = False   # if True, append the pending payout amount / 10 to the global state


def state_of(env):
    s = env.get_state()
    if AUG_PENDING_AMOUNT:
        s = np.append(s, np.float32(sum(env._pending.values()) / 10.0))
    return s


class BankedScriptedTeam(ScriptedTeam):
    """ScriptedTeam, except that a finisher task stays feasible when an absent enabler
    has already pulled its lever. Identical to ScriptedTeam for removal at t = 0."""

    def _finish_feasible(self, env, k) -> bool:
        ev = self.config.events[k]
        if not all(env.present[e] or env.pulled[k][e] for e in ev.enablers):
            return False
        return ev.partial or all(env.present[f] for f in ev.finishers)


def make_team(cell, split: str, index: int) -> BankedScriptedTeam:
    base = cell.seed * 100_000 + (0 if split == "train" else 50_000)
    return BankedScriptedTeam(cell.config(), seed=base + index, epsilon=cell.epsilon,
                              finish_not_before=cell.finish_not_before, enabler_start=cell.enabler_start)


def drop_agents(env: PlantedCoalitionGridworld, agents) -> None:
    for i in agents:
        if env.present[i]:
            env.present[i] = False
            env.positions[i] = None
            env._present_mask &= ~(1 << i)


def run(cfg, team, drop_time=None, drop_set=(), start_mask=None):
    """One closed-loop episode. Returns states (T+1, S), rewards (T,), actions (T, n).
    states[t] is the global state *after* any removal scheduled for step t."""
    env = PlantedCoalitionGridworld(cfg)
    env.reset(start_mask)
    team.reset(env)
    T = cfg.horizon
    states = np.zeros((T + 1, env.state_size + int(AUG_PENDING_AMOUNT)), dtype=np.float32)
    rewards = np.zeros(T)
    actions = np.zeros((T, cfg.n_agents), dtype=np.int64)
    for t in range(T):
        if drop_time is not None and t == drop_time:
            drop_agents(env, drop_set)
        states[t] = state_of(env)
        actions[t] = team.act(env)
        _, rewards[t], _, _ = env.step(actions[t])
    states[T] = state_of(env)
    return states, rewards, actions


def to_go(rewards, gamma=GAMMA):
    g = np.zeros(len(rewards) + 1)
    for t in range(len(rewards) - 1, -1, -1):
        g[t] = rewards[t] + gamma * g[t + 1]
    return g


def remove_from_state(state: np.ndarray, n: int, agents) -> np.ndarray:
    """The global state with ``agents`` absent: position zeroed, presence flag 0 (layout from get_state)."""
    s = np.array(state, copy=True)
    for i in agents:
        s[..., 2 * i:2 * i + 2] = 0.0
        s[..., 2 * n + i] = 0.0
    return s


def sample_intervention(rng, n, T, p_drop=0.5):
    """(drop_time, drop_set) or (None, ()). Half of the interventions remove from t = 0, half at a
    uniform random step; half remove one agent, half a random non-empty proper-or-full subset."""
    if rng.random() >= p_drop:
        return None, ()
    t = 0 if rng.random() < 0.5 else int(rng.integers(0, T))
    if rng.random() < 0.5:
        s = (int(rng.integers(n)),)
    else:
        while True:
            bits = rng.random(n) < 0.5
            if bits.any():
                break
        s = tuple(int(i) for i in np.flatnonzero(bits))
    return t, s


def training_set(cell, p_drop=0.5, n_episodes=None, seed_offset=0):
    """(X, y, info): states and discounted returns-to-go from ``n_train`` episodes, a fraction
    ``p_drop`` of which carry one removal intervention. p_drop = 0 gives the team's own data."""
    cfg = cell.config()
    n, T = cfg.n_agents, cfg.horizon
    rng = np.random.default_rng([cell.seed, 777 + seed_offset])
    X, y, yu, returns, n_int = [], [], [], [], 0
    for e in range(n_episodes or cell.n_train):
        dt, ds = sample_intervention(rng, n, T, p_drop)
        n_int += dt is not None
        states, rewards, _ = run(cfg, make_team(cell, "train", e), dt, ds)
        X.append(states[:T])
        y.append(to_go(rewards, cell.gamma)[:T])
        yu.append(to_go(rewards, 1.0)[:T])
        returns.append(rewards.sum())
    Y = np.stack([np.concatenate(y), np.concatenate(yu)], 1)   # discounted, undiscounted return-to-go
    return np.concatenate(X), Y, {"interventions": n_int, "mean_return": float(np.mean(returns))}


def true_persistent_values(cell, index: int, split="test"):
    """Exact V^{-i}(t): discounted return from step t when agent i is removed at step t and the rest of
    the scripted team continues (closed loop, common random numbers). Shape (T+1, n); row T is 0.
    Also returns the factual (states, rewards, actions)."""
    cfg = cell.config()
    n, T = cfg.n_agents, cfg.horizon
    states, rewards, actions = run(cfg, make_team(cell, split, index))
    V = np.zeros((T + 1, n))
    for i in range(n):
        for t in range(T):
            _, r, _ = run(cfg, make_team(cell, split, index), t, (i,))
            V[t, i] = to_go(r, cell.gamma)[t]
    return V, states, rewards, actions


def persistent_credit(V_minus, rewards, gamma=GAMMA):
    """c[t, i] = r_t + gamma * V^{-i}(t+1) - V^{-i}(t). Its gamma^t-weighted sum over t telescopes to
    G_0 - V^{-i}(0) exactly."""
    return rewards[:, None] + gamma * V_minus[1:] - V_minus[:-1]


def closed_loop_values(cell, index: int, split="test"):
    """v(S) for every S under closed-loop removal from t = 0 (the repo's sensitivity protocol),
    discounted from t = 0 and undiscounted."""
    cfg = cell.config()
    n = cfg.n_agents
    vd, vu = np.zeros(1 << n), np.zeros(1 << n)
    for s in range(1 << n):
        _, r, _ = run(cfg, make_team(cell, split, index), start_mask=s)
        vd[s], vu[s] = to_go(r, cell.gamma)[0], r.sum()
    return vd, vu


def shares(x):
    return gt.normalise_shares(np.asarray(x, dtype=np.float64))


def l1(a, b):
    sa, sb = shares(a), shares(b)
    return None if sa is None or sb is None else float(np.abs(sa - sb).sum())


# --- a torch-free mirror of whoearned.experiments.DataCell (same layouts, seeds and defaults) ----
from dataclasses import dataclass  # noqa: E402

from whoearned.envs import coalition_layout, overlap_example  # noqa: E402


@dataclass(frozen=True)
class Cell:
    delay: int
    n_bystanders: int
    seed: int
    horizon: int = 60
    n_train: int = 200
    n_test: int = 30
    epsilon: float = 0.2
    finish_not_before: int = 0
    enabler_start: int = 0
    gamma: float = 0.99
    layout: str = "coalition"

    @property
    def key(self) -> str:
        crit = f"_es{self.enabler_start}" if self.enabler_start else ""
        prefix = "" if self.layout == "coalition" else f"{self.layout}_"
        return f"{prefix}d{self.delay}_b{self.n_bystanders}{crit}_s{self.seed}"

    def config(self):
        if self.layout == "overlap":
            return overlap_example(delay=self.delay, horizon=self.horizon)
        return coalition_layout(self.n_bystanders, delay=self.delay, horizon=self.horizon,
                                partial=self.layout == "partial")

    def team(self, split: str, index: int) -> ScriptedTeam:
        base = self.seed * 100_000 + (0 if split == "train" else 50_000)
        return ScriptedTeam(self.config(), seed=base + index, epsilon=self.epsilon,
                            finish_not_before=self.finish_not_before, enabler_start=self.enabler_start)


def sparse_dividends(v_hat, cv=5, seed=0):
    """Denoised Harsanyi dividends: Lasso fit of v_hat(S) - v_hat(empty) = sum_{A subset S} Delta(A) on the
    full Moebius basis (no order limit), penalty chosen by cross-validation over coalitions. No ground
    truth is used. Returns an array of length 2**n (entry 0 is 0)."""
    from sklearn.linear_model import LassoCV
    v = np.asarray(v_hat, dtype=np.float64)
    v = v - v[0]
    m = len(v)
    M = np.array([[1.0 if (a & s) == a else 0.0 for a in range(1, m)] for s in range(1, m)])
    try:      # scikit-learn >= 1.7 takes the number of penalties as ``alphas``
        model = LassoCV(cv=cv, fit_intercept=False, random_state=seed, max_iter=50000, alphas=40)
        model.fit(M, v[1:])
    except (TypeError, ValueError):   # older versions
        model = LassoCV(cv=cv, fit_intercept=False, random_state=seed, max_iter=50000, n_alphas=40)
        model.fit(M, v[1:])
    d = np.zeros(m)
    d[1:] = model.coef_
    return d

