"""Planted Coalition Gridworld. See docs/environment_design.md for the full specification."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

Cell = tuple[int, int]

NOOP, UP, DOWN, LEFT, RIGHT, INTERACT = range(6)
N_ACTIONS = 6
_MOVES = {UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1)}


@dataclass(frozen=True)
class EventSpec:
    """One planted coalition event.

    ``lever_cells`` maps each enabler to its lever; ``goal_cells`` maps each
    finisher to its goal cell.
    """

    enablers: tuple[int, ...]
    finishers: tuple[int, ...]
    reward: float
    lever_cells: dict[int, Cell]
    goal_cells: dict[int, Cell]
    partial: bool = False

    @property
    def coalition(self) -> tuple[int, ...]:
        return tuple(sorted(set(self.enablers) | set(self.finishers)))


@dataclass(frozen=True)
class GridConfig:
    n_agents: int
    height: int
    width: int
    horizon: int
    delay: int
    start_cells: tuple[Cell, ...]
    events: tuple[EventSpec, ...]
    walls: frozenset[Cell] = field(default_factory=frozenset)
    collisions: bool = True

    def __post_init__(self):
        if len(self.start_cells) != self.n_agents:
            raise ValueError("need one start cell per agent")
        if self.delay < 0 or self.horizon < 1:
            raise ValueError("delay must be >= 0 and horizon >= 1")
        if self.collisions and len(set(self.start_cells)) != self.n_agents:
            raise ValueError("start cells must be distinct when collisions are on")
        for cell in self.start_cells:
            self._check_cell(cell)
        for k, ev in enumerate(self.events):
            if not ev.finishers:
                raise ValueError(f"event {k} has no finishers")
            if set(ev.enablers) & set(ev.finishers):
                raise ValueError(f"event {k}: an agent cannot be enabler and finisher")
            if set(ev.lever_cells) != set(ev.enablers):
                raise ValueError(f"event {k}: need exactly one lever cell per enabler")
            if set(ev.goal_cells) != set(ev.finishers):
                raise ValueError(f"event {k}: need exactly one goal cell per finisher")
            for a in ev.coalition:
                if not 0 <= a < self.n_agents:
                    raise ValueError(f"event {k}: unknown agent {a}")
            for cell in [*ev.lever_cells.values(), *ev.goal_cells.values()]:
                self._check_cell(cell)

    def _check_cell(self, cell: Cell):
        r, c = cell
        if not (0 <= r < self.height and 0 <= c < self.width):
            raise ValueError(f"cell {cell} is outside the grid")
        if cell in self.walls:
            raise ValueError(f"cell {cell} is a wall")

    @property
    def bystanders(self) -> tuple[int, ...]:
        involved = {a for ev in self.events for a in ev.coalition}
        return tuple(a for a in range(self.n_agents) if a not in involved)


@dataclass
class EventRecord:
    pull_times: dict[int, int] = field(default_factory=dict)
    gate_open_time: int | None = None
    completion_time: int | None = None
    participants: tuple[int, ...] = ()
    payout_time: int | None = None
    payout: float = 0.0
    clipped: bool = False


class PlantedCoalitionGridworld:
    def __init__(self, config: GridConfig):
        self.config = config
        self.n_agents = config.n_agents
        self.n_events = len(config.events)
        self.obs_size = 4 * self.n_agents + 7 * self.n_events + 4
        # positions (2n), presence (n), acting (n); per event: gate, done, pending, countdown,
        # pending amount, pulled-by-enabler (n); per agent per event: is-enabler, is-finisher (the
        # agents' "unit types", which their own observations already carry); then t / T.
        # See docs/environment_design.md.
        self.state_size = (4 * self.n_agents + self.n_events * (5 + self.n_agents)
                           + 2 * self.n_agents * self.n_events + 1)
        self._present_mask = 0
        self.reset()

    # --- episode control -------------------------------------------------------

    def reset(self, present: int | None = None) -> list[np.ndarray]:
        """Start an episode with only the agents in bitmask ``present`` (default: all)."""
        n = self.n_agents
        full = (1 << n) - 1
        self._present_mask = full if present is None else present & full
        self.present = [bool(self._present_mask >> i & 1) for i in range(n)]
        # ``acting[i]`` is False once agent i has been stopped (``stop``); absent agents never act.
        self.acting = list(self.present)
        self.positions: list[Cell | None] = [
            cell if self.present[i] else None for i, cell in enumerate(self.config.start_cells)
        ]
        self.t = 0
        self.pulled = [{e: False for e in ev.enablers} for ev in self.config.events]
        self.done_events = [False] * self.n_events
        self.event_log = [EventRecord() for _ in self.config.events]
        self._pending: dict[int, float] = {}
        return self.get_obs()

    @property
    def present_mask(self) -> int:
        return self._present_mask

    def stop(self, agents, null: str = "noop") -> None:
        """Stop ``agents`` for the rest of the episode (a persistent intervention).

        ``null="noop"``: the agent stays on the grid and only takes NOOP; teammates are not
        told (their observations are unchanged). ``null="remove"``: the agent leaves the grid,
        as with ``reset(present=...)``, and teammates see it gone. Either way, levers it has
        already pulled stay pulled and events it has completed stay completed. Only the global
        state's ``acting`` flags reveal a no-op stop.
        """
        if null not in ("noop", "remove"):
            raise ValueError(f"null must be 'noop' or 'remove', got {null!r}")
        for i in agents:
            i = int(i)
            if not 0 <= i < self.n_agents:
                raise ValueError(f"unknown agent {i}")
            self.acting[i] = False
            if null == "remove":
                self.present[i] = False
                self.positions[i] = None
                self._present_mask &= ~(1 << i)

    @property
    def acting_mask(self) -> int:
        return sum(1 << i for i, a in enumerate(self.acting) if a)

    def clone(self) -> "PlantedCoalitionGridworld":
        """An independent copy of the full environment state (for branching counterfactual continuations)."""
        e = PlantedCoalitionGridworld.__new__(PlantedCoalitionGridworld)
        e.__dict__.update(self.__dict__)
        e.present = list(self.present)
        e.acting = list(self.acting)
        e.positions = list(self.positions)
        e.pulled = [dict(p) for p in self.pulled]
        e.done_events = list(self.done_events)
        e.event_log = [EventRecord(dict(r.pull_times), r.gate_open_time, r.completion_time, tuple(r.participants),
                                   r.payout_time, r.payout, r.clipped) for r in self.event_log]
        e._pending = dict(self._pending)
        return e

    def gate_open(self, k: int) -> bool:
        return all(self.pulled[k].values())

    def step(self, actions) -> tuple[list[np.ndarray], float, bool, dict]:
        if self.t >= self.config.horizon:
            raise RuntimeError("episode is over; call reset()")
        actions = [int(a) for a in actions]
        if len(actions) != self.n_agents:
            raise ValueError(f"expected {self.n_agents} actions, got {len(actions)}")
        actions = [a if self.acting[i] else NOOP for i, a in enumerate(actions)]
        cfg = self.config
        t = self.t

        self._move(actions)

        gate_before = [self.gate_open(k) for k in range(self.n_events)]
        interacting = [self.present[i] and actions[i] == INTERACT for i in range(self.n_agents)]

        for k, ev in enumerate(cfg.events):
            for e in ev.enablers:
                if interacting[e] and not self.pulled[k][e] and self.positions[e] == ev.lever_cells[e]:
                    self.pulled[k][e] = True
                    self.event_log[k].pull_times[e] = t
            if self.event_log[k].gate_open_time is None and self.gate_open(k):
                self.event_log[k].gate_open_time = t

        for k, ev in enumerate(cfg.events):
            if self.done_events[k] or not gate_before[k]:
                continue
            at_goal = tuple(
                f for f in ev.finishers if interacting[f] and self.positions[f] == ev.goal_cells[f]
            )
            complete = bool(at_goal) if ev.partial else len(at_goal) == len(ev.finishers)
            if complete:
                self._complete(k, ev, at_goal, t)

        reward = self._pending.pop(t, 0.0)
        self.t += 1
        done = self.t >= cfg.horizon
        info = {"t": t, "events_done": list(self.done_events)}
        return self.get_obs(), reward, done, info

    def _move(self, actions: list[int]):
        cfg = self.config
        occupied = {p for p in self.positions if p is not None}
        for i, a in enumerate(actions):
            if not self.present[i] or a not in _MOVES:
                continue
            r, c = self.positions[i]
            dr, dc = _MOVES[a]
            target = (r + dr, c + dc)
            if not (0 <= target[0] < cfg.height and 0 <= target[1] < cfg.width):
                continue
            if target in cfg.walls:
                continue
            if cfg.collisions and target in occupied:
                continue
            occupied.discard((r, c))
            occupied.add(target)
            self.positions[i] = target

    def _complete(self, k: int, ev: EventSpec, participants: tuple[int, ...], t: int):
        horizon = self.config.horizon
        payout = ev.reward * len(participants) / len(ev.finishers)
        pay_t = t + self.config.delay
        rec = self.event_log[k]
        rec.completion_time = t
        rec.participants = participants
        rec.payout = payout
        rec.clipped = pay_t >= horizon
        rec.payout_time = min(pay_t, horizon - 1)
        self.done_events[k] = True
        self._pending[rec.payout_time] = self._pending.get(rec.payout_time, 0.0) + payout

    # --- observations ------------------------------------------------------------

    def get_avail_actions(self) -> list[list[int]]:
        return [[1] * N_ACTIONS if a else [1] + [0] * (N_ACTIONS - 1) for a in self.acting]

    def _norm(self, cell: Cell) -> tuple[float, float]:
        return cell[0] / max(self.config.height - 1, 1), cell[1] / max(self.config.width - 1, 1)

    def _rel(self, src: Cell, dst: Cell) -> tuple[float, float]:
        return (
            (dst[0] - src[0]) / max(self.config.height - 1, 1),
            (dst[1] - src[1]) / max(self.config.width - 1, 1),
        )

    def get_obs_agent(self, i: int) -> np.ndarray:
        n, cfg = self.n_agents, self.config
        obs = np.zeros(self.obs_size, dtype=np.float32)
        if not self.present[i]:
            return obs
        me = self.positions[i]
        obs[i] = 1.0
        o = n
        obs[o:o + 3] = (*self._norm(me), 1.0)
        o += 3
        for j in range(n):
            if self.present[j]:
                obs[o:o + 3] = (*self._rel(me, self.positions[j]), 1.0)
            o += 3
        for k, ev in enumerate(cfg.events):
            station = ev.lever_cells.get(i) or ev.goal_cells.get(i)
            obs[o:o + 7] = (
                float(self.gate_open(k)),
                float(self.done_events[k]),
                float(self.pulled[k].get(i, False)),
                float(i in ev.enablers),
                float(i in ev.finishers),
                *(self._rel(me, station) if station else (0.0, 0.0)),
            )
            o += 7
        obs[o] = self.t / cfg.horizon
        return obs

    def get_obs(self) -> list[np.ndarray]:
        return [self.get_obs_agent(i) for i in range(self.n_agents)]

    def get_state(self) -> np.ndarray:
        n = self.n_agents
        state = np.zeros(self.state_size, dtype=np.float32)
        for i, p in enumerate(self.positions):
            if p is not None:
                state[2 * i:2 * i + 2] = self._norm(p)
                state[2 * n + i] = 1.0
            state[3 * n + i] = float(self.acting[i])
        o = 4 * n
        for k, ev in enumerate(self.config.events):
            # Payout countdown: makes the state Markov under delay (reward is a function
            # of the state). Only the global state carries it; agent observations do not.
            pay_t = self.event_log[k].payout_time
            pending = pay_t is not None and pay_t >= self.t
            state[o] = float(self.gate_open(k))
            state[o + 1] = float(self.done_events[k])
            state[o + 2] = float(pending)
            state[o + 3] = (pay_t - self.t) / self.config.horizon if pending else 0.0
            # Pending amount / event reward: a partial event pays r * (participants / finishers), so
            # without it the state is not Markov while a partial payout is pending (v3 Phase 0.1).
            state[o + 4] = self.event_log[k].payout / ev.reward if pending else 0.0
            for e in ev.enablers:
                state[o + 5 + e] = float(self.pulled[k][e])
            o += 5 + n
        K = self.n_events
        for i in range(n):
            for k, ev in enumerate(self.config.events):
                state[o + 2 * K * i + 2 * k] = float(i in ev.enablers)
                state[o + 2 * K * i + 2 * k + 1] = float(i in ev.finishers)
        o += 2 * n * K
        state[o] = self.t / self.config.horizon
        return state

    # --- debugging -----------------------------------------------------------------

    def render(self) -> str:
        """ASCII view: agents as A..Z, levers 'l', goals 'g', walls '#'."""
        cfg = self.config
        grid = [["." for _ in range(cfg.width)] for _ in range(cfg.height)]
        for r, c in cfg.walls:
            grid[r][c] = "#"
        for ev in cfg.events:
            for r, c in ev.lever_cells.values():
                grid[r][c] = "l"
            for r, c in ev.goal_cells.values():
                grid[r][c] = "g"
        for i, p in enumerate(self.positions):
            if p is not None:
                grid[p[0]][p[1]] = chr(ord("A") + i)
        return "\n".join("".join(row) for row in grid)
