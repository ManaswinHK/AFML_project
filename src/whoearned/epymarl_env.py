"""EPyMARL adapter for the Planted Coalition Gridworld.

Implements EPyMARL's ``MultiAgentEnv`` interface by duck typing, so this module
does not import EPyMARL. ``scripts/run_epymarl.py`` registers :func:`make_env`
under the name ``"planted"``.

The horizon is a true terminal state (time t/T is in every observation), so the
last step is stored as terminated rather than as a truncated time limit: there
is nothing to bootstrap from after T.
"""

from __future__ import annotations

from whoearned.envs import (
    N_ACTIONS,
    NOOP,
    PlantedCoalitionGridworld,
    coalition_layout,
    overlap_example,
    running_example,
)
from whoearned.envs.layouts import double_enabler_layout, multi_event_layout
from whoearned.envs.gridworld import GridConfig

LAYOUTS = {
    "running_example": running_example,
    "overlap_example": overlap_example,
    "coalition_layout": coalition_layout,
    "multi_event_layout": multi_event_layout,
    "double_enabler_layout": double_enabler_layout,
}
# v3 Phase 3. "team": one shared reward (EPyMARL common_reward=True). "team_vector": every agent receives
# the team reward as its own (common_reward=False), the control for "difference". "difference": exact
# difference rewards, r - r^(-i), where r^(-i) comes from a shadow copy of the environment fed the same
# joint actions except that agent i takes no-op from the start of the episode (Wolpert & Tumer's
# clamping, open loop, in lockstep: the repo's open-loop replay run online).
# v4: "expected" - expected-value credit for lane layouts: agent i's reward is its own lane's reward plus the other
# lanes' TD errors under per-lane value tables learned online (TD(0)) in this environment worker:
#     kappa_i(t) = r_t + W_{-i}(s_{t+1}) - W_{-i}(s_t),  W_{-i}(s) = sum_{j != lane(i)} V_j(s_j)
# (stopping i forfeits its own lane's future reward; lanes are independent). It differs from "difference"
# (= own lane's reward exactly) by exactly the other lanes' TD errors: their *luck*. docs/v4/theory note.
REWARD_MODES = ("team", "team_vector", "difference", "expected")


def build_config(layout: str, **layout_kwargs) -> GridConfig:
    if layout not in LAYOUTS:
        raise ValueError(f"unknown layout {layout!r}; choose from {sorted(LAYOUTS)}")
    return LAYOUTS[layout](**layout_kwargs)


class PlantedEPyMARLEnv:
    def __init__(
        self,
        map_name: str = "running_example",
        layout_kwargs: dict | None = None,
        seed: int | None = None,
        common_reward: bool = True,
        reward_scalarisation: str = "sum",
        present: int | None = None,
        reward_mode: str = "team",
        dropout_p: float = 0.0,
        dropout_null: str = "noop",
        dropout_seed: int | None = None,
    ):
        if reward_mode not in REWARD_MODES:
            raise ValueError(f"reward_mode must be one of {REWARD_MODES}")
        if (reward_mode == "team") != bool(common_reward):
            raise ValueError("reward_mode 'team' needs common_reward=True; per-agent modes need common_reward=False")
        self.reward_mode = reward_mode
        # EPyMARL names runs by env_args.map_name, so the layout is selected through it.
        self.config = build_config(map_name, **(layout_kwargs or {}))
        self.env = PlantedCoalitionGridworld(self.config)
        self.n_agents = self.config.n_agents
        self.episode_limit = self.config.horizon
        self.present = present
        self._seed = seed
        self._return = 0.0
        self.shadows = ([PlantedCoalitionGridworld(self.config) for _ in range(self.n_agents)]
                        if reward_mode == "difference" else [])
        self.lane_alpha = 0.1
        self.lane_tables = [dict() for _ in self.config.events] if reward_mode == "expected" else []
        self.lane_of = {a: k for k, ev in enumerate(self.config.events) for a in ev.coalition}
        if reward_mode == "expected" and len(self.lane_of) != self.n_agents:
            raise ValueError("reward_mode 'expected' needs every agent in exactly one event (lane layouts)")
        # v3 Phase 4.1: with probability dropout_p per episode, a random set of agents (as in the intervention
        # data: half single agents, half random subsets) stops for good at a uniformly random step. The global
        # state's acting flags show it, so the centralised critic of an actor-critic learner is trained on
        # stopped states, i.e. it becomes an ablation critic as a side effect of robustness training.
        import numpy as _np
        self.dropout_p, self.dropout_null = float(dropout_p), dropout_null
        self._drop_rng = _np.random.default_rng(dropout_seed if dropout_seed is not None else seed)
        self._drop_plan = None

    # --- MultiAgentEnv interface ---------------------------------------------------

    def reset(self, seed=None, options=None):
        obs = self.env.reset(self.present)
        self._drop_plan = None
        if self.dropout_p > 0 and self._drop_rng.random() < self.dropout_p:
            from whoearned.ablation import sample_stopped
            t0 = int(self._drop_rng.integers(self.episode_limit))
            self._drop_plan = (t0, sample_stopped(self._drop_rng, self.n_agents))
        for i, sh in enumerate(self.shadows):
            sh.reset(self.present)
            sh.stop([i], "noop")
        self._return = 0.0
        return obs, {}

    def step(self, actions):
        if self._drop_plan is not None and self.env.t == self._drop_plan[0]:
            self.env.stop(self._drop_plan[1], self.dropout_null)
        actions = [int(a) for a in actions]
        before = (([self._lane_state(k) for k in range(len(self.lane_tables))], self.env.t)
                  if self.reward_mode == "expected" else None)
        obs, reward, done, _ = self.env.step(actions)
        self._return += reward
        # EPyMARL sums info values across episodes, so only numeric stats go here.
        info = ({"events_completed": float(sum(self.env.done_events)),
                 "success_rate": float(sum(self.env.done_events)) / max(self.env.n_events, 1)} if done else {})
        if self.reward_mode == "team":
            return obs, float(reward), bool(done), False, info
        if self.reward_mode == "team_vector":
            return obs, [float(reward)] * self.n_agents, bool(done), False, info
        if self.reward_mode == "expected":
            return obs, self._expected_credit(before, float(reward), done), bool(done), False, info
        rewards = [float(reward - sh.step(actions)[1]) for sh in self.shadows]   # shadow i overrides i to no-op
        return obs, rewards, bool(done), False, info

    # --- v4 expected-value credit (lane layouts) ---------------------------------------------------

    def _lane_state(self, k: int):
        ev = self.config.events[k]
        return (self.env.t, tuple(self.env.positions[a] for a in ev.coalition),
                tuple(self.env.pulled[k].values()), self.env.done_events[k])

    def _lane_rewards(self, t: int) -> list[float]:
        return [rec.payout if rec.payout_time == t and rec.completion_time is not None else 0.0
                for rec in self.env.event_log]

    def _expected_credit(self, before, reward: float, done: bool) -> list[float]:
        states_before, t = before
        r_lane = self._lane_rewards(t)
        td = []
        for k, table in enumerate(self.lane_tables):
            v0 = table.get(states_before[k], 0.0)
            v1 = 0.0 if done else table.get(self._lane_state(k), 0.0)
            delta = r_lane[k] + v1 - v0
            td.append(delta)
            table[states_before[k]] = v0 + self.lane_alpha * delta          # TD(0), after using v0 for credit
        total_td = sum(td)
        return [float(r_lane[self.lane_of[i]] + total_td - td[self.lane_of[i]]) for i in range(self.n_agents)]

    def get_obs(self):
        return self.env.get_obs()

    def get_obs_agent(self, agent_id):
        return self.env.get_obs_agent(agent_id)

    def get_obs_size(self):
        return self.env.obs_size

    def get_state(self):
        return self.env.get_state()

    def get_state_size(self):
        return self.env.state_size

    def get_avail_actions(self):
        return self.env.get_avail_actions()

    def get_avail_agent_actions(self, agent_id):
        return self.env.get_avail_actions()[agent_id]

    def get_total_actions(self):
        return N_ACTIONS

    def get_env_info(self):
        return {
            "state_shape": self.get_state_size(),
            "obs_shape": self.get_obs_size(),
            "n_actions": self.get_total_actions(),
            "n_agents": self.n_agents,
            "episode_limit": self.episode_limit,
        }

    def get_stats(self):
        return {}

    def render(self):
        print(self.env.render())

    def close(self):
        pass

    def seed(self, seed=None):
        self._seed = seed

    def save_replay(self):
        pass


def make_env(**kwargs) -> PlantedEPyMARLEnv:
    """EPyMARL REGISTRY entry: called with env_args plus common_reward / reward_scalarisation."""
    return PlantedEPyMARLEnv(**kwargs)
