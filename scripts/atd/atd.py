"""Ablation-TD credit: oracle and learned versions, on the Planted Coalition Gridworld.

Persistent-ablation value  W_S(s)  = expected return-to-go from state s when only the agents in S
keep acting from now on (the others are removed and stay removed; the remaining team reacts).

Ablation-TD credit of agent i at step t (undiscounted, gamma = 1):
    kappa[i, t] = r_t + W_{N-i}(s_{t+1}) - W_{N-i}(s_t)
Summed over t this telescopes to R(tau) - W_{N-i}(s_0) = v(N) - v(N - i): leave-one-out removal credit.

Nothing here imports torch or EPyMARL except `fit_critic`.
"""
from __future__ import annotations

import copy
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Repo root: set WHOEARNED_REPO, or drop this folder into <repo>/scripts/atd/.
REPO = os.environ.get("WHOEARNED_REPO", str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, os.path.join(REPO, "src"))
from whoearned.envs.gridworld import PlantedCoalitionGridworld  # noqa: E402
from whoearned.envs.layouts import coalition_layout, overlap_example  # noqa: E402
from whoearned.scripted import ScriptedTeam  # noqa: E402


@dataclass(frozen=True)
class Cell:
    """Same seeding and layouts as whoearned.experiments.DataCell (re-implemented to avoid EPyMARL)."""
    delay: int
    b: int
    seed: int
    es: int = 0                 # enabler_start: 0 = slack, 12 = "critical"
    layout: str = "coalition"
    eps: float = 0.2
    n_train: int = 200
    n_test: int = 30

    @property
    def horizon(self) -> int:
        return 80 if self.delay == 50 else 60

    def config(self):
        if self.layout == "overlap":
            return overlap_example(delay=self.delay, horizon=self.horizon)
        return coalition_layout(self.b, delay=self.delay, horizon=self.horizon, partial=self.layout == "partial")

    def team(self, split: str, index: int) -> ScriptedTeam:
        base = self.seed * 100_000 + (0 if split == "train" else 50_000)
        return ScriptedTeam(self.config(), seed=base + index, epsilon=self.eps, enabler_start=self.es)

    @property
    def key(self) -> str:
        crit = f"_es{self.es}" if self.es else ""
        prefix = "" if self.layout == "coalition" else f"{self.layout}_"
        return f"{prefix}d{self.delay}_b{self.b}{crit}_s{self.seed}"


def clone_env(env: PlantedCoalitionGridworld) -> PlantedCoalitionGridworld:
    e = PlantedCoalitionGridworld.__new__(PlantedCoalitionGridworld)
    e.__dict__.update(env.__dict__)
    e.present = list(env.present)
    e.positions = list(env.positions)
    e.pulled = [dict(p) for p in env.pulled]
    e.done_events = list(env.done_events)
    e.event_log = copy.deepcopy(env.event_log)
    e._pending = dict(env._pending)
    return e


def remove_agent(env: PlantedCoalitionGridworld, i: int) -> None:
    """Take agent i off the grid from now on. Levers it already pulled stay pulled."""
    env.present[i] = False
    env.positions[i] = None
    env._present_mask &= ~(1 << i)


def edit_state(state: np.ndarray, n: int, removed) -> np.ndarray:
    """The global state the environment would report if `removed` left right now."""
    s = np.array(state, copy=True)
    for i in removed:
        s[2 * i:2 * i + 2] = 0.0
        s[2 * n + i] = 0.0
    return s


@dataclass
class Factual:
    cfg: object
    snaps: list            # per step t: (env clone before the step, rng states before the step)
    states: np.ndarray     # (T+1, ds) global state before each step, plus the terminal state
    rewards: np.ndarray    # (T,)
    actions: np.ndarray    # (T, n)
    event_log: list
    t_done: int            # first step at which every event is done (T if never)

    @property
    def total(self) -> float:
        return float(self.rewards.sum())

    def to_go(self) -> np.ndarray:
        g = np.zeros(len(self.rewards) + 1)
        for t in range(len(self.rewards) - 1, -1, -1):
            g[t] = self.rewards[t] + g[t + 1]
        return g


def factual(cell: Cell, split: str, index: int, keep_snaps: bool = True) -> Factual:
    cfg = cell.config()
    env = PlantedCoalitionGridworld(cfg)
    env.reset()
    team = cell.team(split, index)
    team.reset(env)
    snaps, states, rewards, actions = [], [], [], []
    t_done = cfg.horizon
    for t in range(cfg.horizon):
        if keep_snaps:
            snaps.append((clone_env(env), [r.bit_generator.state for r in team.rngs]))
        states.append(env.get_state())
        if t_done == cfg.horizon and all(env.done_events):
            t_done = t
        a = team.act(env)
        _, r, _, _ = env.step(a)
        rewards.append(r)
        actions.append(a)
    states.append(env.get_state())
    return Factual(cfg, snaps, np.array(states), np.array(rewards), np.array(actions), env.event_log, t_done)


def aug(state: np.ndarray, n: int, inactive=()) -> np.ndarray:
    """Global state plus an explicit 'still acting' mask (needed for the no-op null, where presence flags do not change)."""
    m = np.ones(n, dtype=np.float32)
    m[list(inactive)] = 0.0
    return np.concatenate([state, m])


def continue_ablated(cell: Cell, split: str, index: int, snap, removed, record: bool = False, null: str = "remove"):
    """Closed-loop continuation from a snapshot with `removed` agents taken off the grid (common random numbers).

    Returns the undiscounted return-to-go; with record=True also the states and rewards of the continuation.
    """
    env0, rng_states = snap
    env = clone_env(env0)
    n = env.n_agents
    if null == "remove":
        for i in removed:
            remove_agent(env, i)
    elif null != "noop":
        raise ValueError(null)
    team = cell.team(split, index)
    team.reset(env)
    for r, st in zip(team.rngs, rng_states):
        r.bit_generator.state = st
    g, states, rewards = 0.0, [], []
    for _ in range(env.t, env.config.horizon):
        if not record and all(env.done_events):
            return g + sum(env._pending.values())      # only already-earned payouts remain
        if record:
            states.append(aug(env.get_state(), n, removed))
        a = team.act(env)
        if null == "noop":                      # the agent stays on the grid, inert; teammates are not told
            for i in removed:
                a[i] = 0
        _, r, _, _ = env.step(a)
        g += r
        rewards.append(r)
    if record:
        return g, np.array(states), np.array(rewards)
    return g


def oracle_ablation_values(cell: Cell, split: str, index: int, f: Factual, coalitions=None, null: str = "remove") -> np.ndarray:
    """W[c, t] for t = 0..T: return-to-go from the factual state s_t if coalition c's complement leaves at t.

    `coalitions` is a list of removed-agent tuples; default: each single agent (leave-one-out).
    """
    n, T = f.cfg.n_agents, f.cfg.horizon
    coalitions = [(i,) for i in range(n)] if coalitions is None else coalitions
    W = np.zeros((len(coalitions), T + 1))
    togo = f.to_go()
    for t in range(T):
        if t >= f.t_done:                       # everything is already earned; nobody is needed any more
            W[:, t] = togo[t]
            continue
        for c, removed in enumerate(coalitions):
            W[c, t] = continue_ablated(cell, split, index, f.snaps[t], removed, null=null)
    return W


def atd_credit(rewards: np.ndarray, W: np.ndarray) -> np.ndarray:
    """kappa[c, t] = r_t + W[c, t+1] - W[c, t]."""
    return rewards[None, :] + W[:, 1:] - W[:, :-1]


# ----------------------------------------------------------------------------------------------
# Learned ablation critic
# ----------------------------------------------------------------------------------------------

def ablation_dataset(cell: Cell, n_abl: int, seed: int = 0, single_prob: float = 0.5, drop_prob: float | None = None,
                     include_full: bool = True, null: str = "remove", until: str = "done"):
    """(X, y): global state -> undiscounted return-to-go, from the cell's 200 full-team training episodes
    plus `n_abl` closed-loop continuations in which a random set of agents leaves at a random step.

    The leave step is uniform over the steps before the source episode has finished all its events
    (after that nothing is outstanding and every ablation is a no-op).
    """
    rng = np.random.default_rng([seed, 777])
    cfg = cell.config()
    n = cfg.n_agents
    X, y = [], []
    facts = {}

    def get(idx):
        if idx not in facts:
            facts[idx] = factual(cell, "train", idx)
        return facts[idx]

    if include_full:
        for idx in range(cell.n_train):
            f = get(idx)
            X.append(np.stack([aug(st, n) for st in f.states[:-1]]))
            y.append(f.to_go()[:-1])
    n_zero = 0
    for _ in range(n_abl):
        idx = int(rng.integers(cell.n_train))
        f = get(idx)
        t_end = f.t_done
        if until == "payout" and f.rewards.any():      # also cover the wait between completion and payout
            t_end = int(np.flatnonzero(f.rewards)[-1]) + 1
        t0 = int(rng.integers(0, max(t_end, 1)))
        if rng.random() < single_prob:
            removed = (int(rng.integers(n)),)
        else:
            p = drop_prob if drop_prob is not None else rng.uniform(0.1, 0.9)   # None: cover every coalition size
            removed = tuple(int(i) for i in np.flatnonzero(rng.random(n) < p))
            if not removed:
                removed = (int(rng.integers(n)),)
        g, states, rewards = continue_ablated(cell, "train", idx, f.snaps[t0], removed, record=True, null=null)
        togo = np.cumsum(rewards[::-1])[::-1]
        X.append(states)
        y.append(togo)
        n_zero += g == 0
    return np.concatenate(X), np.concatenate(y), {"n_abl": n_abl, "abl_return_zero": n_zero / max(n_abl, 1)}


def fit_critic(X, y, seed=0, hidden=128, steps=3000, batch=256, lr=1e-3, scale=10.0):
    import torch as th
    th.manual_seed(seed)
    th.set_num_threads(1)
    net = th.nn.Sequential(th.nn.Linear(X.shape[1], hidden), th.nn.ReLU(), th.nn.Linear(hidden, hidden), th.nn.ReLU(),
                           th.nn.Linear(hidden, 1))
    opt = th.optim.Adam(net.parameters(), lr=lr)
    Xt, yt = th.tensor(X, dtype=th.float32), th.tensor(y / scale, dtype=th.float32)
    g = th.Generator().manual_seed(seed)
    for _ in range(steps):
        idx = th.randint(0, len(Xt), (batch,), generator=g)
        loss = ((net(Xt[idx]).squeeze(-1) - yt[idx]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    def predict(S):
        with th.no_grad():
            return net(th.tensor(np.asarray(S), dtype=th.float32)).squeeze(-1).numpy() * scale
    predict.final_loss = float(loss.detach())
    return predict


def learned_ablation_values(predict, f: Factual, coalitions=None, null: str = "remove") -> np.ndarray:
    """W_hat[c, t] from factual states only: the critic evaluated at the state with the agents edited out."""
    n, T = f.cfg.n_agents, f.cfg.horizon
    coalitions = [(i,) for i in range(n)] if coalitions is None else coalitions
    W = np.zeros((len(coalitions), T + 1))
    for c, removed in enumerate(coalitions):
        W[c, :T] = predict(np.stack([aug(edit_state(s, n, removed) if null == "remove" else s, n, removed) for s in f.states[:T]]))
    return W


def fit_mobius_critic(X, y, n, k=3, seed=0, hidden=128, steps=3000, batch=256, lr=1e-3, scale=10.0, sparsity=0.0):
    """State-conditioned low-order dividend critic for the no-op null.

    W(s, m) = sum over coalitions S with |S| <= k and S inside the active set m of g_S(s).
    The network sees the raw state only (the last n columns of X are the active mask and are used as the gate),
    so its outputs ARE the Harsanyi dividends of the state game w_s(m) = W(s, m), with nothing above order k.
    """
    import itertools
    import torch as th
    th.manual_seed(seed)
    th.set_num_threads(1)
    subsets = [c for o in range(1, k + 1) for c in itertools.combinations(range(n), o)]
    member = np.zeros((len(subsets), n), dtype=np.float32)
    for j, c in enumerate(subsets):
        member[j, list(c)] = 1.0
    member_t = th.tensor(member)
    S, M = X[:, :-n], X[:, -n:]
    net = th.nn.Sequential(th.nn.Linear(S.shape[1], hidden), th.nn.ReLU(), th.nn.Linear(hidden, hidden), th.nn.ReLU(),
                           th.nn.Linear(hidden, len(subsets)))
    opt = th.optim.Adam(net.parameters(), lr=lr)
    St, Mt, yt = th.tensor(S, dtype=th.float32), th.tensor(M, dtype=th.float32), th.tensor(y / scale, dtype=th.float32)

    def gate(m):                                   # (B, n_subsets): 1 if every member of S is active
        return ((1.0 - m) @ member_t.T == 0).float()

    g = th.Generator().manual_seed(seed)
    for _ in range(steps):
        idx = th.randint(0, len(St), (batch,), generator=g)
        terms = net(St[idx]) * gate(Mt[idx])
        pred = terms.sum(-1)
        loss = ((pred - yt[idx]) ** 2).mean() + sparsity * terms.abs().sum(-1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    def dividends(states):                         # (B, n_subsets) in reward units
        with th.no_grad():
            return net(th.tensor(np.asarray(states), dtype=th.float32)).numpy() * scale
    dividends.subsets = subsets
    dividends.member = member
    dividends.final_loss = float(loss.detach())
    return dividends


def mobius_ablation_values(dividends, f: Factual) -> np.ndarray:
    """Leave-one-out W_hat[i, t] = sum of the dividends of coalitions that do not contain i."""
    n, T = f.cfg.n_agents, f.cfg.horizon
    g = dividends(f.states[:T])                    # (T, n_subsets)
    W = np.zeros((n, T + 1))
    for i in range(n):
        W[i, :T] = (g * (1.0 - dividends.member[:, i])[None]).sum(-1)
    return W
