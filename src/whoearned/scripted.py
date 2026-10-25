"""Scripted team for Mode P data. See docs/environment_design.md, "Scripted team"."""

from __future__ import annotations

from collections import deque

import numpy as np

from whoearned.envs.gridworld import (
    DOWN,
    INTERACT,
    LEFT,
    N_ACTIONS,
    NOOP,
    RIGHT,
    UP,
    Cell,
    GridConfig,
    PlantedCoalitionGridworld,
)

_STEPS = ((UP, (-1, 0)), (DOWN, (1, 0)), (LEFT, (0, -1)), (RIGHT, (0, 1)))
ENABLE, FINISH = "enable", "finish"
# wander: random moves, never onto a coalition member's planned route (default, keeps roles as planted)
# roam:   random moves anywhere, so a bystander can obstruct (planted vs. realised roles)
# idle:   never moves
BYSTANDER_MODES = ("wander", "roam", "idle")


class ScriptedTeam:
    """Privileged controller for all agents.

    Interface shared with learned teams: ``reset(env)`` after ``env.reset``,
    then ``act(env) -> list[int]`` once per step.
    """

    def __init__(
        self,
        config: GridConfig,
        seed: int = 0,
        epsilon: float = 0.0,
        enabler_start: int = 0,
        finish_not_before: int = 0,
        bystander_mode: str = "wander",
    ):
        if bystander_mode not in BYSTANDER_MODES:
            raise ValueError(f"bystander_mode must be one of {BYSTANDER_MODES}, got {bystander_mode!r}")
        self.config = config
        self.seed = seed
        self.epsilon = epsilon
        self.enabler_start = enabler_start
        self.finish_not_before = finish_not_before
        self.bystander_mode = bystander_mode
        self.bystanders = set(config.bystanders)
        self.tasks = [
            sorted(
                [(k, ENABLE) for k, ev in enumerate(config.events) if i in ev.enablers]
                + [(k, FINISH) for k, ev in enumerate(config.events) if i in ev.finishers]
            )
            for i in range(config.n_agents)
        ]
        self.reserved = self._planned_route_cells()
        self.rngs: list[np.random.Generator] = []

    def _planned_route_cells(self) -> set[Cell]:
        """Cells on every coalition member's route start -> station -> station ... (ignoring agents)."""
        cfg = self.config
        cells: set[Cell] = set()
        for i, tasks in enumerate(self.tasks):
            if not tasks:
                continue
            pos = cfg.start_cells[i]
            cells.add(pos)
            for k, role in tasks:
                ev = cfg.events[k]
                station = ev.lever_cells[i] if role == ENABLE else ev.goal_cells[i]
                cells.update(_shortest_path(cfg, pos, station, set()) or [])
                pos = station
        return cells

    def reset(self, env: PlantedCoalitionGridworld):
        self.rngs = [np.random.default_rng([self.seed, i]) for i in range(self.config.n_agents)]

    def act(self, env: PlantedCoalitionGridworld) -> list[int]:
        actions = []
        for i, rng in enumerate(self.rngs):
            # Fixed draws every step, present or not, so streams stay aligned under removal.
            u_noise = rng.random()
            u_action = int(rng.integers(N_ACTIONS))
            u_wander = int(rng.integers(5))
            if not env.present[i]:
                actions.append(NOOP)
                continue
            if i in self.bystanders:
                actions.append(self._bystander_action(env, i, u_wander))
                continue
            action = self._intended(env, i)
            if u_noise < self.epsilon:
                action = u_action
            actions.append(action)
        return actions

    def _bystander_action(self, env, i, u_wander: int) -> int:
        if self.bystander_mode == "idle" or u_wander == NOOP:
            return NOOP
        if self.bystander_mode == "roam":
            return u_wander
        r, c = env.positions[i]
        dr, dc = dict(_STEPS)[u_wander]
        return NOOP if (r + dr, c + dc) in self.reserved else u_wander

    # --- task logic ----------------------------------------------------------------

    def _current_task(self, env, i):
        for k, role in self.tasks[i]:
            if env.done_events[k]:
                continue
            if role == ENABLE and not env.pulled[k][i]:
                return k, role
            if role == FINISH and self._finish_feasible(env, k):
                return k, role
        return None

    def _finish_feasible(self, env, k) -> bool:
        ev = self.config.events[k]
        if not all(env.present[e] for e in ev.enablers):
            return False
        return ev.partial or all(env.present[f] for f in ev.finishers)

    def _intended(self, env, i) -> int:
        task = self._current_task(env, i)
        if task is None:
            return NOOP
        k, role = task
        ev = self.config.events[k]
        pos = env.positions[i]
        if role == ENABLE:
            if env.t < self.enabler_start:
                return NOOP
            target = ev.lever_cells[i]
            return INTERACT if pos == target else self._step_towards(env, i, target)
        target = ev.goal_cells[i]
        if pos != target:
            return self._step_towards(env, i, target)
        ready = (
            env.gate_open(k)
            and env.t >= self.finish_not_before
            and all(env.positions[f] == ev.goal_cells[f] for f in ev.finishers if env.present[f])
        )
        return INTERACT if ready else NOOP

    def _step_towards(self, env, i, target: Cell) -> int:
        start = env.positions[i]
        others = {p for j, p in enumerate(env.positions) if p is not None and j != i}
        blocked = others if self.config.collisions else set()
        first = _first_step(self.config, start, target, blocked)
        if first is None:
            first = _first_step(self.config, start, target, set())
        if first is None:
            return NOOP
        action, cell = first
        return NOOP if cell in blocked else action


def _first_step(config: GridConfig, start: Cell, target: Cell, blocked: set[Cell]):
    """First (action, cell) of a shortest path from start to target, or None."""
    path = _shortest_path(config, start, target, blocked)
    if not path or len(path) < 2:
        return None
    nxt = path[1]
    action = next(a for a, (dr, dc) in _STEPS if (start[0] + dr, start[1] + dc) == nxt)
    return action, nxt


def _shortest_path(config: GridConfig, start: Cell, target: Cell, blocked: set[Cell]):
    """Cells of a shortest path start..target (inclusive), avoiding walls and ``blocked``; None if unreachable."""
    parent: dict[Cell, Cell | None] = {start: None}
    queue = deque([start])
    while queue:
        cell = queue.popleft()
        if cell == target:
            break
        for _, (dr, dc) in _STEPS:
            nxt = (cell[0] + dr, cell[1] + dc)
            if nxt in parent or not (0 <= nxt[0] < config.height and 0 <= nxt[1] < config.width):
                continue
            if nxt in config.walls or (nxt in blocked and nxt != target):
                continue
            parent[nxt] = cell
            queue.append(nxt)
    if target not in parent:
        return None
    path = [target]
    while path[-1] != start:
        path.append(parent[path[-1]])
    return path[::-1]
