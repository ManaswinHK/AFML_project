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
    PlantedCoalitionGridworld,
    coalition_layout,
    overlap_example,
    running_example,
)
from whoearned.envs.gridworld import GridConfig

LAYOUTS = {
    "running_example": running_example,
    "overlap_example": overlap_example,
    "coalition_layout": coalition_layout,
}


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
    ):
        if not common_reward:
            raise ValueError("the planted gridworld only has a shared team reward; set common_reward=True")
        # EPyMARL names runs by env_args.map_name, so the layout is selected through it.
        self.config = build_config(map_name, **(layout_kwargs or {}))
        self.env = PlantedCoalitionGridworld(self.config)
        self.n_agents = self.config.n_agents
        self.episode_limit = self.config.horizon
        self.present = present
        self._seed = seed
        self._return = 0.0

    # --- MultiAgentEnv interface ---------------------------------------------------

    def reset(self, seed=None, options=None):
        obs = self.env.reset(self.present)
        self._return = 0.0
        return obs, {}

    def step(self, actions):
        obs, reward, done, _ = self.env.step([int(a) for a in actions])
        self._return += reward
        # EPyMARL sums info values across episodes, so only numeric stats go here.
        info = {"events_completed": float(sum(self.env.done_events))} if done else {}
        return obs, float(reward), bool(done), False, info

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
