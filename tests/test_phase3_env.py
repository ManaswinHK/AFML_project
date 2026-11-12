"""v3 Phase 3: multi-event lanes and exact difference rewards."""

import numpy as np
import pytest

from whoearned.envs import INTERACT, NOOP, RIGHT, PlantedCoalitionGridworld
from whoearned.envs.layouts import multi_event_layout
from whoearned.epymarl_env import PlantedEPyMARLEnv


def _solve_lane_actions(K, solved):
    """Scripted joint actions that complete exactly the lanes in ``solved`` (lane length 4)."""
    seq_e = [RIGHT, RIGHT, INTERACT] + [NOOP] * 22
    seq_f = [RIGHT, RIGHT, RIGHT, NOOP, INTERACT] + [NOOP] * 20
    out = []
    for t in range(25):
        a = []
        for k in range(K):
            a += [seq_e[t], seq_f[t]] if k in solved else [NOOP, NOOP]
        out.append(a)
    return out


def test_lanes_are_independent_and_pay_10_each():
    K = 4
    cfg = multi_event_layout(K)
    env = PlantedCoalitionGridworld(cfg)
    env.reset()
    total = sum(env.step(a)[1] for a in _solve_lane_actions(K, {0, 2}))
    assert total == pytest.approx(20.0) and env.done_events == [True, False, True, False]


def test_difference_reward_is_each_agents_own_lane():
    K = 3
    env = PlantedEPyMARLEnv(map_name="multi_event_layout", layout_kwargs={"n_events": K}, common_reward=False,
                            reward_mode="difference")
    env.reset()
    totals = np.zeros(2 * K)
    team = 0.0
    for a in _solve_lane_actions(K, {0, 2}):
        _, r, _, _, _ = env.step(a)
        totals += r
    np.testing.assert_allclose(totals, [10, 10, 0, 0, 10, 10])


def test_team_vector_gives_everyone_the_team_reward():
    env = PlantedEPyMARLEnv(map_name="multi_event_layout", layout_kwargs={"n_events": 2}, common_reward=False,
                            reward_mode="team_vector")
    env.reset()
    tot = np.zeros(4)
    for a in _solve_lane_actions(2, {1}):
        tot += env.step(a)[1]
    np.testing.assert_allclose(tot, [10, 10, 10, 10])


def test_reward_mode_must_match_common_reward():
    with pytest.raises(ValueError):
        PlantedEPyMARLEnv(map_name="multi_event_layout", common_reward=True, reward_mode="difference")


def test_dropout_wrapper_stops_agents_and_flags_them_in_state():
    env = PlantedEPyMARLEnv(map_name="multi_event_layout", layout_kwargs={"n_events": 2}, dropout_p=1.0, dropout_seed=3)
    stopped_seen = 0
    for _ in range(20):
        env.reset()
        for t in range(25):
            env.step([NOOP] * 4)
        n = env.n_agents
        acting = env.get_state()[3 * n:4 * n]
        stopped_seen += acting.min() == 0.0 or env._drop_plan[0] == 24
    assert stopped_seen == 20
    env0 = PlantedEPyMARLEnv(map_name="multi_event_layout", layout_kwargs={"n_events": 2}, dropout_p=0.0)
    env0.reset()
    assert env0._drop_plan is None


def test_lanes_with_bystanders():
    from whoearned.envs.layouts import multi_event_layout
    cfg = multi_event_layout(2, n_bystanders=2)
    assert cfg.n_agents == 6 and cfg.bystanders == (4, 5)
    env = PlantedCoalitionGridworld(cfg)
    env.reset()
    total = 0.0
    for a in _solve_lane_actions(2, {0, 1}):
        total += env.step(a + [RIGHT, RIGHT])[1]
    assert total == pytest.approx(20.0)
    assert all(p[0] == 6 for p in env.positions[4:])     # bystanders stay in their own row


def test_expected_credit_is_own_lane_reward_plus_other_lanes_td_errors():
    K = 3
    env = PlantedEPyMARLEnv(map_name="multi_event_layout", layout_kwargs={"n_events": K}, common_reward=False,
                            reward_mode="expected")
    # With empty value tables, expected credit = own lane reward + other lanes' rewards (= team reward).
    env.reset()
    tot = np.zeros(2 * K)
    for a in _solve_lane_actions(K, {0, 2}):
        tot += env.step(a)[1]
    np.testing.assert_allclose(tot, [20] * 6)
    # After many identical episodes the tables learn the (deterministic) lane values, other lanes' TD errors
    # vanish, and expected credit converges to the exact difference reward (own lane).
    for _ in range(400):
        env.reset()
        tot = np.zeros(2 * K)
        for a in _solve_lane_actions(K, {0, 2}):
            tot += env.step(a)[1]
    np.testing.assert_allclose(tot, [10, 10, 0, 0, 10, 10], atol=0.05)
