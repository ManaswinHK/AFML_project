import os

import numpy as np
import pytest

EPYMARL = os.path.join(os.path.dirname(os.path.dirname(__file__)), "external", "epymarl", "src")
pytestmark = pytest.mark.skipif(not os.path.isdir(EPYMARL), reason="EPyMARL not cloned into external/")

if os.path.isdir(EPYMARL):
    import torch as th

    from whoearned.epymarl_bridge import build_harness, episodes_to_batch
    from whoearned.readouts import (
        _mac_out,
        coma_credit,
        episode_credit,
        policy_gradient_snr,
        q_difference_credit,
        shared_advantage,
    )

from whoearned.envs import NOOP, PlantedCoalitionGridworld, running_example
from whoearned.epymarl_env import PlantedEPyMARLEnv
from whoearned.replay import rollout
from whoearned.scripted import ScriptedTeam

N, T = 5, 60


@pytest.fixture(scope="module")
def episodes():
    cfg = running_example()
    return [rollout(cfg, ScriptedTeam(cfg, seed=s, epsilon=0.1)) for s in range(4)]


@pytest.fixture(scope="module")
def harnesses(episodes):
    out = {}
    for alg in ("vdn", "qmix", "coma", "mappo"):
        h = build_harness(alg, seed=0)
        out[alg] = (h, episodes_to_batch(h, episodes))
    return out


# --- adapter ----------------------------------------------------------------------


def test_adapter_env_info_and_step_contract():
    env = PlantedEPyMARLEnv(map_name="running_example", seed=0)
    info = env.get_env_info()
    assert info == {"state_shape": 4 * 5 + 1 * (5 + 5) + 2 * 5 * 1 + 1, "obs_shape": 31, "n_actions": 6, "n_agents": 5, "episode_limit": 60}
    obs, _ = env.reset()
    assert len(obs) == 5
    for t in range(T):
        obs, r, terminated, truncated, info = env.step([NOOP] * 5)
        assert not truncated
        assert terminated == (t == T - 1)
    assert info == {"events_completed": 0.0, "success_rate": 0.0}  # numeric only: EPyMARL sums info values


def test_adapter_rejects_unknown_layout_and_individual_rewards():
    with pytest.raises(ValueError):
        PlantedEPyMARLEnv(map_name="nope")
    with pytest.raises(ValueError):
        PlantedEPyMARLEnv(common_reward=False)


# --- episodes -> EpisodeBatch ------------------------------------------------------


def test_batch_matches_the_recorded_episodes(harnesses, episodes):
    _, batch = harnesses["vdn"]
    assert batch.batch_size == 4 and batch.max_seq_length == T + 1
    rewards = batch["reward"][:, :, 0].numpy()
    for b, ep in enumerate(episodes):
        np.testing.assert_allclose(rewards[b, :T], ep.rewards)
        np.testing.assert_array_equal(batch["actions"][b, :T, :, 0].numpy(), ep.actions)
    term = batch["terminated"][:, :, 0].numpy()
    assert term[:, T - 1].all() and not term[:, : T - 1].any()
    assert batch["filled"][:, :T].all()


def test_batch_observations_equal_the_environment(harnesses, episodes):
    _, batch = harnesses["vdn"]
    ep = episodes[0]
    env = PlantedCoalitionGridworld(ep.config)
    env.reset()
    for t in range(5):
        np.testing.assert_allclose(batch["obs"][0, t].numpy(), np.stack(env.get_obs()))
        np.testing.assert_allclose(batch["state"][0, t].numpy(), env.get_state())
        env.step(ep.actions[t])


def test_batch_rejects_tampered_episode(episodes):
    h = build_harness("vdn")
    bad = episodes[0]
    bad = type(bad)(bad.config, bad.actions, bad.rewards + 1.0, bad.event_log, bad.present)
    with pytest.raises(ValueError):
        episodes_to_batch(h, [bad])


# --- read-outs ------------------------------------------------------------------------


@pytest.mark.parametrize("alg", ["vdn", "qmix"])
def test_q_difference_is_zero_when_agent_took_the_null_action(harnesses, alg):
    h, batch = harnesses[alg]
    credit = q_difference_credit(h, batch)
    actions = batch["actions"][:, :-1, :, 0].numpy()
    assert credit.shape == (4, T, N)
    np.testing.assert_allclose(credit[actions == NOOP], 0.0, atol=1e-6)
    assert np.abs(credit[actions != NOOP]).max() > 0


def test_vdn_q_difference_reduces_to_utility_difference(harnesses):
    h, batch = harnesses["vdn"]
    with th.no_grad():
        q = _mac_out(h.learner.mac, batch).numpy()
    a = batch["actions"][:, :-1].numpy()
    expected = np.take_along_axis(q, a, axis=3)[..., 0] - q[..., NOOP]
    np.testing.assert_allclose(q_difference_credit(h, batch), expected, atol=1e-5)


def test_coma_credit_shape_and_finite(harnesses):
    h, batch = harnesses["coma"]
    credit = coma_credit(h, batch)
    assert credit.shape == (4, T, N) and np.isfinite(credit).all()


def test_coma_critic_row_ignores_own_action(harnesses):
    # COMA's critic masks agent i's own action, which is what makes its baseline counterfactual.
    h, batch = harnesses["coma"]
    with th.no_grad():
        q1 = h.learner.critic(batch)[:, :-1].numpy()
        batch["actions_onehot"][:, :, 0] = th.roll(batch["actions_onehot"][:, :, 0], 1, dims=-1)
        q2 = h.learner.critic(batch)[:, :-1].numpy()
        batch["actions_onehot"][:, :, 0] = th.roll(batch["actions_onehot"][:, :, 0], -1, dims=-1)
    np.testing.assert_allclose(q1[:, :, 0], q2[:, :, 0], atol=1e-6)      # agent 0's row unchanged
    assert not np.allclose(q1[:, :, 1:], q2[:, :, 1:])                     # others' rows change


def test_mappo_shared_advantage_and_snr(harnesses):
    h, batch = harnesses["mappo"]
    adv = shared_advantage(h, batch)
    assert adv.shape == (4, T, N) and np.isfinite(adv).all()
    snr = policy_gradient_snr(h, batch, adv)
    for key in ("signal", "noise", "snr"):
        assert snr[key].shape == (N,)
    assert (snr["noise"] > 0).all()


def test_episode_credit_window():
    per_step = np.ones((2, 10, 3))
    np.testing.assert_array_equal(episode_credit(per_step), np.full((2, 3), 10.0))
    np.testing.assert_array_equal(episode_credit(per_step, 2, 5), np.full((2, 3), 3.0))


# --- offline (Mode P) training plumbing ------------------------------------------------


def test_vdn_learner_trains_on_scripted_batch(harnesses):
    h, batch = harnesses["vdn"]
    for k in range(30):
        h.learner.train(batch, t_env=k * T, episode_num=k)
    losses = [v for _, v in h.logger.stats.get("loss", [])]
    assert losses and all(np.isfinite(losses))
