"""Mode T: rolling out a (here untrained) EPyMARL controller as a gridworld team."""

import os

import numpy as np
import pytest

EPYMARL = os.path.join(os.path.dirname(os.path.dirname(__file__)), "external", "epymarl", "src")
needs_epymarl = pytest.mark.skipif(not os.path.isdir(EPYMARL), reason="EPyMARL not cloned into external/")

ENV_ARGS = {"map_name": "coalition_layout", "layout_kwargs": {"n_bystanders": 1, "delay": 0, "horizon": 20}}


@needs_epymarl
@pytest.mark.parametrize("alg", ["qmix", "mappo"])
def test_mac_team_rolls_out_with_valid_behaviour_probs(alg):
    from whoearned.epymarl_bridge import build_harness, episodes_to_batch
    from whoearned.mode_t import MacTeam
    from whoearned.replay import replay, rollout

    h = build_harness(alg, env_args=ENV_ARGS, seed=0)
    cfg = h.env.config
    ep = rollout(cfg, MacTeam(h, epsilon=0.2, seed=1))
    assert ep.behaviour_probs.shape == (cfg.horizon, cfg.n_agents, 6)
    np.testing.assert_allclose(ep.behaviour_probs.sum(-1), 1.0, atol=1e-6)
    assert np.all(ep.behaviour_probs >= 0.2 / 6 - 1e-9)  # epsilon-uniform floor
    assert replay(cfg, ep.actions).total == ep.total
    episodes_to_batch(h, [ep])  # replays to the recorded rewards or raises


@needs_epymarl
def test_greedy_q_team_is_deterministic():
    from whoearned.epymarl_bridge import build_harness
    from whoearned.mode_t import MacTeam
    from whoearned.replay import rollout

    h = build_harness("qmix", env_args=ENV_ARGS, seed=0)
    a = rollout(h.env.config, MacTeam(h, epsilon=0.0, seed=1)).actions
    b = rollout(h.env.config, MacTeam(h, epsilon=0.0, seed=2)).actions
    np.testing.assert_array_equal(a, b)
