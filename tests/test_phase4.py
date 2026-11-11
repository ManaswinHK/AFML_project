"""Branching a learned EPyMARL policy is exact: an unstopped continuation reproduces the factual rest."""

import numpy as np

from whoearned import ablation as ab
from whoearned.epymarl_bridge import build_harness
from whoearned.phase4 import BranchableMacTeam, lanes_env_args


def test_branchable_mac_team_continuations_are_exact():
    h = build_harness("mappo", env_args=lanes_env_args(2), seed=0)
    cfg = h.env.config
    make = lambda: BranchableMacTeam(h, epsilon=0.3, seed=7)
    f = ab.factual_rollout(cfg, make)
    togo = f.to_go()
    for t in (0, 3, 11, 20):
        g, states, rewards = ab.continue_stopped(cfg, make, f.snaps[t], (), "noop", record=True)
        np.testing.assert_array_equal(rewards, f.rewards[t:])
        np.testing.assert_array_equal(states[0], f.states[t])
        assert g == togo[t]
    # a stop changes the continuation only through the stopped agent's lane
    W = ab.oracle_ablation_values(cfg, make, f, null="noop")
    assert W.shape == (cfg.n_agents, cfg.horizon + 1)
