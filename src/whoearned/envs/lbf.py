"""Deterministic Level-Based Foraging (v4 Part B), with persistent stops, for exact credit ground truth.

Rules follow `lbforaging` 2.0.0 (`lbforaging/foraging/environment.py`), which the tests use as the reference:

* Grid of food items with levels; agents with levels. Actions NONE, NORTH, SOUTH, WEST, EAST, LOAD.
* A move is valid if it stays on the grid and the target cell holds no food. Agents whose targets coincide
  (including a target occupied by an agent that stays) all fail to move; otherwise moves happen (swaps allowed).
* LOAD is valid next to food. The loading agents adjacent to a food item load it if their level sum >= the food's
  level; the food disappears and the team earns 10 x food level / total food level spawned (the package pays each
  loading agent level x food / (level sum x total); the team sum is the same).
* The episode ends when all food is gone or at the horizon.

One deliberate difference: the package processes loading agents by popping a Python ``set`` of objects, whose
order depends on memory addresses, so a copied environment can resolve simultaneous loads differently. Exact
common-random-number truth needs determinism, so here loading agents are processed in index order. The two agree
whenever no loading agent is adjacent to two foods (tested against the package).

The interface duck-types ``PlantedCoalitionGridworld`` so ``whoearned.ablation`` works unchanged: ``reset``,
``step``, ``stop``, ``clone``, ``get_state`` (positions, presence and acting flags in the same leading layout),
``t``, ``done_events`` (one per food), ``_pending`` (always empty: rewards are immediate), ``config``.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

NONE, NORTH, SOUTH, WEST, EAST, LOAD = range(6)
N_ACTIONS = 6
FOOD_FEATS = 5      # per food item in the global state: row, col, level, reward share, present
_MOVES = {NORTH: (-1, 0), SOUTH: (1, 0), WEST: (0, -1), EAST: (0, 1)}


@dataclass(frozen=True)
class LBFConfig:
    """One episode's layout (positions and levels are drawn per episode from ``spawn_seed``)."""

    n_agents: int = 4
    size: int = 8
    n_food: int = 3
    min_agent_level: int = 1
    max_agent_level: int = 2
    min_food_level: int = 1
    max_food_level: int | None = None     # None: sum of the 3 lowest agent levels (package default)
    # v4 Part B: "lowest3" (package default cap, used when max_food_level is None) or "all" (cap = sum of ALL agent
    # levels, so heavy food can need every agent; any 3 of 4 agents can lift any food under the default cap)
    food_cap: str = "lowest3"
    horizon: int = 50
    spawn_seed: tuple = (0,)

    @property
    def bystanders(self) -> tuple:   # interface compatibility: LBF has no planted bystanders
        return ()

    @property
    def events(self) -> tuple:       # interface compatibility: one "event" per food item
        return tuple(range(self.n_food))


class LBFEnv:
    def __init__(self, config: LBFConfig):
        self.config = config
        self.n_agents = config.n_agents
        self.n_events = config.n_food
        self.state_size = 4 * self.n_agents + self.n_agents + FOOD_FEATS * config.n_food + 1
        self.reset()

    # --- episode control ------------------------------------------------------------------------

    def reset(self, present=None):
        cfg = self.config
        rng = np.random.default_rng(list(cfg.spawn_seed))
        n, S = cfg.n_agents, cfg.size
        full = (1 << n) - 1
        self._present_mask = full if present is None else present & full
        self.levels = rng.integers(cfg.min_agent_level, cfg.max_agent_level + 1, size=n)
        if cfg.max_food_level is not None:
            cap = cfg.max_food_level
        elif cfg.food_cap == "all":
            cap = int(self.levels.sum())
        elif cfg.food_cap == "lowest3":
            cap = int(np.sort(self.levels)[:3].sum())
        else:
            raise ValueError(cfg.food_cap)
        self.field = np.zeros((S, S), dtype=np.int64)
        self.food_pos, self.food_level = [], []
        attempts = 0
        while len(self.food_pos) < cfg.n_food and attempts < 1000:
            attempts += 1
            r, c = int(rng.integers(1, S - 1)), int(rng.integers(1, S - 1))
            if self.field[max(r - 1, 0):r + 2, max(c - 1, 0):c + 2].sum() > 0:       # no adjacent food
                continue
            if self.field[max(r - 2, 0):r + 3, c].sum() + self.field[r, max(c - 2, 0):c + 3].sum() > 0:
                continue
            lvl = int(rng.integers(cfg.min_food_level, cap + 1))
            self.field[r, c] = lvl
            self.food_pos.append((r, c))
            self.food_level.append(lvl)
        self.total_food = float(sum(self.food_level))
        self.positions = [None] * n
        occupied = set(self.food_pos)
        for i in range(n):
            while True:
                cell = (int(rng.integers(0, S)), int(rng.integers(0, S)))
                if cell not in occupied:
                    occupied.add(cell)
                    self.positions[i] = cell
                    break
        self.present = [bool(self._present_mask >> i & 1) for i in range(n)]
        for i in range(n):
            if not self.present[i]:
                self.positions[i] = None
        self.acting = list(self.present)
        self.food_present = [True] * len(self.food_pos)
        self.t = 0
        self._pending = {}
        self.event_log = []
        return None

    @property
    def done_events(self) -> list:
        return [not p for p in self.food_present]

    @property
    def present_mask(self) -> int:
        return self._present_mask

    def stop(self, agents, null: str = "noop") -> None:
        for i in agents:
            i = int(i)
            self.acting[i] = False
            if null == "remove":
                self.present[i] = False
                self.positions[i] = None
                self._present_mask &= ~(1 << i)
            elif null != "noop":
                raise ValueError(null)

    def clone(self) -> "LBFEnv":
        e = LBFEnv.__new__(LBFEnv)
        e.__dict__.update(self.__dict__)
        e.positions = list(self.positions)
        e.present = list(self.present)
        e.acting = list(self.acting)
        e.field = self.field.copy()
        e.food_present = list(self.food_present)
        e._pending = {}
        e.event_log = []
        return e

    # --- dynamics -------------------------------------------------------------------------------------

    def _valid(self, i: int, a: int) -> bool:
        p = self.positions[i]
        if p is None or not self.acting[i]:
            return a == NONE
        if a == NONE:
            return True
        S = self.config.size
        if a in _MOVES:
            r, c = p[0] + _MOVES[a][0], p[1] + _MOVES[a][1]
            return 0 <= r < S and 0 <= c < S and self.field[r, c] == 0
        if a == LOAD:
            return self._adjacent_food(p) is not None
        raise ValueError(a)

    def _adjacent_food(self, p):
        """First adjacent food in the package's N, S, W, E order (with its bounds checks)."""
        r, c = p
        S = self.config.size
        if r > 1 and self.field[r - 1, c] > 0:
            return (r - 1, c)
        if r < S - 1 and self.field[r + 1, c] > 0:
            return (r + 1, c)
        if c > 1 and self.field[r, c - 1] > 0:
            return (r, c - 1)
        if c < S - 1 and self.field[r, c + 1] > 0:
            return (r, c + 1)
        return None

    def get_avail_actions(self):
        return [[1 if self._valid(i, a) else 0 for a in range(N_ACTIONS)] for i in range(self.n_agents)]

    def step(self, actions):
        n = self.n_agents
        actions = [int(a) if self._valid(i, int(a)) else NONE for i, a in enumerate(actions)]
        targets = {}
        loading = []
        for i, a in enumerate(actions):
            p = self.positions[i]
            if p is None:
                continue
            tgt = (p[0] + _MOVES[a][0], p[1] + _MOVES[a][1]) if a in _MOVES else p
            targets.setdefault(tgt, []).append(i)
            if a == LOAD:
                loading.append(i)
        for tgt, who in targets.items():
            if len(who) == 1:
                self.positions[who[0]] = tgt
        reward = 0.0
        remaining = list(loading)                       # index order: deterministic
        while remaining:
            i = remaining.pop(0)
            f = self._adjacent_food(self.positions[i])
            if f is None:
                continue
            adj = [j for j in range(n) if self.positions[j] is not None and
                   abs(self.positions[j][0] - f[0]) + abs(self.positions[j][1] - f[1]) == 1]
            group = [j for j in adj if j in remaining or j == i]
            remaining = [j for j in remaining if j not in group]
            if sum(int(self.levels[j]) for j in group) < self.field[f]:
                continue
            reward += 10.0 * float(self.field[f]) / self.total_food
            self.field[f] = 0
            self.food_present[self.food_pos.index(f)] = False
        self.t += 1
        done = self.t >= self.config.horizon or not any(self.food_present)
        return None, reward, done, {}

    # --- global state (critic input) ---------------------------------------------------------------------

    def get_state(self) -> np.ndarray:
        """[positions 2n | present n | acting n | levels n | per food (row, col, level, share, present) | t / T].

        The first 4n entries use the gridworld's layout, so ``ablation.stopped_state`` applies unchanged. Agent and
        food levels share one absolute scale (n x max agent level, the largest possible food level), so whether a set
        of agents can load a food is readable from the state. ``share`` is the food's fraction of the episode's total
        reward (10 x level / total spawned), which the remaining return depends on once some food is gone.
        """
        n, S, cfg = self.n_agents, self.config.size, self.config
        scale = float(n * cfg.max_agent_level)
        s = np.zeros(self.state_size, dtype=np.float32)
        for i, p in enumerate(self.positions):
            if p is not None:
                s[2 * i:2 * i + 2] = (p[0] / (S - 1), p[1] / (S - 1))
                s[2 * n + i] = 1.0
            s[3 * n + i] = float(self.acting[i])
            s[4 * n + i] = self.levels[i] / scale
        o = 5 * n
        for k, (pos, lvl) in enumerate(zip(self.food_pos, self.food_level)):
            if self.food_present[k]:
                s[o:o + FOOD_FEATS] = (pos[0] / (S - 1), pos[1] / (S - 1), lvl / scale, lvl / self.total_food, 1.0)
            o += FOOD_FEATS
        s[-1] = self.t / cfg.horizon
        return s


# --- scripted coalition-forming team -----------------------------------------------------------------------


def _bfs_first_step(env: LBFEnv, start, goals: set, blocked: set):
    """First action of a shortest path from start to any goal cell (avoiding food and blocked cells)."""
    if start in goals:
        return NONE
    S = env.config.size
    parent = {start: None}
    q = deque([start])
    while q:
        cur = q.popleft()
        if cur in goals:
            break
        for a, (dr, dc) in _MOVES.items():
            nxt = (cur[0] + dr, cur[1] + dc)
            if nxt in parent or not (0 <= nxt[0] < S and 0 <= nxt[1] < S):
                continue
            if env.field[nxt] > 0 or (nxt in blocked and nxt not in goals):
                continue
            parent[nxt] = (cur, a)
            q.append(nxt)
    hit = next((g for g in goals if g in parent), None)
    if hit is None:
        return NONE
    node, first = hit, NONE
    while parent[node] is not None:
        prev, a = parent[node]
        first = a
        node = prev
    return first


class ScriptedLBFTeam:
    """Centralised greedy coalition planner with epsilon noise, re-planned every step (stateless).

    Food items are handled in a fixed order (highest level first, then position). For each, the nearest available
    agents are added until their level sum covers the food; they walk to free cells adjacent to it and all LOAD once
    every member is adjacent. Agents stopped under the no-op null still count as available (teammates are not told);
    removed agents do not. Per-agent random streams draw the same numbers every step, so streams stay aligned under
    stops (common random numbers), as in ``ScriptedTeam``.
    """

    def __init__(self, config: LBFConfig, seed: int = 0, epsilon: float = 0.2):
        self.config, self.seed, self.epsilon = config, seed, epsilon
        self.rngs = []

    def reset(self, env):
        self.rngs = [np.random.default_rng([self.seed, i]) for i in range(self.config.n_agents)]

    def get_rng_state(self):
        return [r.bit_generator.state for r in self.rngs]

    def set_rng_state(self, states):
        for r, st in zip(self.rngs, states):
            r.bit_generator.state = st

    def plan(self, env: LBFEnv) -> list[int]:
        n = env.n_agents
        avail = {i for i in range(n) if env.present[i]}
        intent = [NONE] * n
        foods = sorted((k for k in range(len(env.food_pos)) if env.food_present[k]),
                       key=lambda k: (-env.food_level[k], env.food_pos[k]))
        for k in foods:
            f = env.food_pos[k]
            dist = lambda i: abs(env.positions[i][0] - f[0]) + abs(env.positions[i][1] - f[1])
            members, total = [], 0
            for i in sorted(avail, key=lambda i: (dist(i), i)):
                if total >= env.food_level[k]:
                    break
                members.append(i)
                total += int(env.levels[i])
            if total < env.food_level[k]:
                continue
            avail -= set(members)
            S = env.config.size
            adj = [(f[0] + dr, f[1] + dc) for dr, dc in _MOVES.values()]
            adj = [c for c in adj if 0 <= c[0] < S and 0 <= c[1] < S and env.field[c] == 0]
            ready = all(dist(i) == 1 for i in members)
            others = {env.positions[j] for j in range(n) if env.present[j]}
            for i in members:
                if ready:
                    intent[i] = LOAD
                elif dist(i) == 1:
                    intent[i] = NONE
                else:
                    free = {c for c in adj if c not in others or c == env.positions[i]}
                    intent[i] = _bfs_first_step(env, env.positions[i], free or set(adj), others - {env.positions[i]})
        return intent

    def act(self, env: LBFEnv) -> list[int]:
        intent = self.plan(env)
        out = []
        for i, rng in enumerate(self.rngs):
            u, a_rand = rng.random(), int(rng.integers(N_ACTIONS))
            a = intent[i]
            if u < self.epsilon:
                a = a_rand
            out.append(a if env.present[i] else NONE)
        return out
