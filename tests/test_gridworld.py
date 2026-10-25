from dataclasses import replace

import numpy as np
import pytest

from whoearned.envs import (
    DOWN,
    INTERACT,
    LEFT,
    NOOP,
    RIGHT,
    UP,
    EventSpec,
    GridConfig,
    PlantedCoalitionGridworld,
    overlap_example,
    running_example,
)
from whoearned.ground_truth import (
    coalition_values,
    mask_of,
    nonzero_dividends,
    shapley_values,
)

A, B, C, D, E = range(5)
ALL5 = 0b11111


def script(*segments, horizon):
    """Concatenate (action, repeat) segments and pad with NOOP to the horizon."""
    acts = [a for a, k in segments for _ in range(k)]
    return acts + [NOOP] * (horizon - len(acts))


def running_scripts(horizon, a_pull_at=4):
    """A pulls its lever, B and C walk 8 cells and interact together at t=8, D and E wander."""
    wander = [LEFT, RIGHT] * (horizon // 2)
    return [
        script((NOOP, a_pull_at - 4), (RIGHT, 4), (INTERACT, 1), horizon=horizon),
        script((RIGHT, 8), (INTERACT, 1), horizon=horizon),
        script((RIGHT, 8), (INTERACT, 1), horizon=horizon),
        wander[:horizon],
        [RIGHT, LEFT] * (horizon // 2),
    ]


def run(env, scripts, present=None):
    """Open-loop replay: returns total return and the steps at which reward arrived."""
    env.reset(present)
    total, reward_steps = 0.0, []
    for t in range(env.config.horizon):
        _, r, done, _ = env.step([s[t] for s in scripts])
        if r:
            total += r
            reward_steps.append(t)
        if done:
            break
    return total, reward_steps


def replay_values(env, scripts):
    return coalition_values(lambda s: run(env, scripts, s)[0], env.n_agents)


# --- Planted rewards ---------------------------------------------------------------


def test_running_example_full_team_earns_planted_reward_after_delay():
    cfg = running_example(delay=15)
    env = PlantedCoalitionGridworld(cfg)
    total, steps = run(env, running_scripts(cfg.horizon))

    assert total == 10.0
    assert steps == [8 + 15]
    rec = env.event_log[0]
    assert rec.pull_times == {A: 4}
    assert rec.gate_open_time == 4
    assert rec.completion_time == 8
    assert rec.participants == (B, C)
    assert not rec.clipped


@pytest.mark.parametrize("delay", [0, 10, 25, 50])
def test_delay_shifts_payout_only(delay):
    cfg = running_example(delay=delay, horizon=60)
    total, steps = run(PlantedCoalitionGridworld(cfg), running_scripts(cfg.horizon))
    assert total == 10.0
    assert steps == [8 + delay]


def test_payout_past_horizon_is_clipped_and_flagged():
    cfg = running_example(delay=15, horizon=20)
    env = PlantedCoalitionGridworld(cfg)
    total, steps = run(env, running_scripts(cfg.horizon))
    assert total == 10.0
    assert steps == [19]
    assert env.event_log[0].clipped


# --- Removal: replay ground truth equals the planted structure ------------------


def test_removing_any_coalition_member_zeroes_the_event():
    cfg = running_example()
    env = PlantedCoalitionGridworld(cfg)
    scripts = running_scripts(cfg.horizon)
    for removed in (A, B, C):
        assert run(env, scripts, ALL5 & ~(1 << removed))[0] == 0.0
    for removed in (D, E):
        assert run(env, scripts, ALL5 & ~(1 << removed))[0] == 10.0


def test_replay_ground_truth_matches_planted_running_example():
    cfg = running_example()
    env = PlantedCoalitionGridworld(cfg)
    v = replay_values(env, running_scripts(cfg.horizon))

    assert nonzero_dividends(v) == {(A, B, C): pytest.approx(10.0)}
    np.testing.assert_allclose(shapley_values(v), [10 / 3] * 3 + [0, 0])


def test_replay_ground_truth_matches_planted_overlap_example():
    cfg = overlap_example()
    env = PlantedCoalitionGridworld(cfg)
    h = cfg.horizon
    scripts = [
        script((RIGHT, 4), (INTERACT, 1), horizon=h),                            # A: lever 0 at t=4
        script((RIGHT, 5), (INTERACT, 1), (RIGHT, 3), (INTERACT, 1), horizon=h),  # B: event 0 at t=5, event 1 at t=9
        script((RIGHT, 5), (INTERACT, 1), horizon=h),                            # C: event 0 at t=5
        script((RIGHT, 4), (INTERACT, 1), horizon=h),                            # D: lever 1 at t=4
        [NOOP] * h,                                                               # E: bystander
    ]
    v = replay_values(env, scripts)

    assert v[ALL5] == 10.0
    assert nonzero_dividends(v) == {(A, B, C): pytest.approx(6.0), (B, D): pytest.approx(4.0)}
    np.testing.assert_allclose(shapley_values(v), [2, 4, 2, 2, 0])


def test_partial_synergy_pays_fraction_per_finisher():
    cfg = running_example(partial=True)
    env = PlantedCoalitionGridworld(cfg)
    v = replay_values(env, running_scripts(cfg.horizon))

    assert v[mask_of([A, B, C])] == 10.0
    assert v[mask_of([A, B])] == 5.0
    assert v[mask_of([A, C])] == 5.0
    assert nonzero_dividends(v) == {(A, B): pytest.approx(5.0), (A, C): pytest.approx(5.0)}
    np.testing.assert_allclose(shapley_values(v), [5, 2.5, 2.5, 0, 0])


# --- Mechanics ---------------------------------------------------------------------


def test_gate_opened_this_step_cannot_be_used_this_step():
    cfg = running_example()
    env = PlantedCoalitionGridworld(cfg)
    # A pulls at t=8, the same step B and C interact: too late.
    assert run(env, running_scripts(cfg.horizon, a_pull_at=8))[0] == 0.0
    assert env.event_log[0].gate_open_time == 8


def test_all_or_nothing_needs_finishers_in_the_same_step():
    cfg = running_example()
    env = PlantedCoalitionGridworld(cfg)
    h = cfg.horizon
    scripts = running_scripts(h)
    scripts[C] = script((RIGHT, 8), (NOOP, 1), (INTERACT, 1), horizon=h)  # C one step late
    assert run(env, scripts)[0] == 0.0


def test_stations_are_keyed_to_their_agent():
    cfg = running_example()
    env = PlantedCoalitionGridworld(cfg)
    h = cfg.horizon
    scripts = running_scripts(h)
    scripts[D] = script((UP, 4), (INTERACT, 1), horizon=h)  # D reaches A's lever (0,4) and pulls
    total, _ = run(env, scripts, ALL5 & ~(1 << A))
    assert total == 0.0
    assert env.event_log[0].pull_times == {}


def test_bystander_can_obstruct_so_its_realised_credit_is_negative():
    cfg = running_example()
    env = PlantedCoalitionGridworld(cfg)
    h = cfg.horizon
    scripts = running_scripts(h)
    scripts[D] = script((UP, 2), horizon=h)  # D parks at (2,4), in B's lane
    v = replay_values(env, scripts)

    assert v[ALL5] == 0.0                     # B is blocked
    assert v[ALL5 & ~(1 << D)] == 10.0        # without D the event succeeds
    assert shapley_values(v)[D] < 0           # planted bystander, realised obstructor


def test_moves_into_walls_and_edges_fail():
    cfg = replace(running_example(), walls=frozenset({(1, 0)}))
    env = PlantedCoalitionGridworld(cfg)
    env.step([DOWN, UP, NOOP, NOOP, NOOP])  # A and B both try to enter the wall at (1,0)
    assert env.positions[A] == (0, 0)
    assert env.positions[B] == (2, 0)
    env.step([UP, LEFT, NOOP, NOOP, NOOP])  # off the top / left edge
    assert env.positions[A] == (0, 0)
    assert env.positions[B] == (2, 0)


# --- Absent agents and fixed-size observations ----------------------------------


def test_absent_agent_is_off_grid_with_zero_obs_and_only_noop():
    cfg = running_example()
    env = PlantedCoalitionGridworld(cfg)
    obs = env.reset(ALL5 & ~(1 << D))

    assert env.positions[D] is None
    assert not obs[D].any()
    assert env.get_avail_actions()[D] == [1, 0, 0, 0, 0, 0]
    assert env.get_avail_actions()[A] == [1] * 6
    # Others see D with presence 0.
    presence_of_d_in_a = obs[A][5 + 3 + 3 * D + 2]
    assert presence_of_d_in_a == 0.0
    env.step([NOOP, NOOP, NOOP, UP, NOOP])  # D's action is ignored
    assert env.positions[D] is None


@pytest.mark.parametrize("mask", [ALL5, 0b00111, 0b00001, 0])
def test_obs_and_state_sizes_are_fixed(mask):
    env = PlantedCoalitionGridworld(overlap_example())
    obs = env.reset(mask)
    assert all(o.shape == (4 * 5 + 7 * 2 + 4,) for o in obs)
    assert env.get_state().shape == (3 * 5 + 2 * (4 + 5) + 1,)


def test_state_carries_payout_countdown_but_observations_do_not():
    cfg = running_example(delay=15)
    env = PlantedCoalitionGridworld(cfg)
    scripts = running_scripts(cfg.horizon)
    env.reset()
    pending, countdown = 3 * 5 + 2, 3 * 5 + 3
    seen = []
    for t in range(cfg.horizon):
        state = env.get_state()
        seen.append((t, state[pending], state[countdown] * cfg.horizon))
        obs_before = np.concatenate(env.get_obs())
        env.step([s[t] for s in scripts])
    # Event completes at t=8 and pays at t=23: from state time 9 to 23 the countdown runs 14..0.
    assert seen[8][1] == 0.0
    assert seen[9][1] == 1.0 and seen[9][2] == pytest.approx(14)
    assert seen[23][1] == 1.0 and seen[23][2] == pytest.approx(0)
    assert seen[24][1] == 0.0
    assert obs_before.shape == (5 * (4 * 5 + 7 + 4),)  # observation size unchanged


def test_replay_is_deterministic():
    cfg = running_example()
    scripts = running_scripts(cfg.horizon)

    def trajectory():
        env = PlantedCoalitionGridworld(cfg)
        env.reset()
        out = []
        for t in range(cfg.horizon):
            obs, r, _, _ = env.step([s[t] for s in scripts])
            out.append((np.concatenate(obs), env.get_state(), r))
        return out

    for (o1, s1, r1), (o2, s2, r2) in zip(trajectory(), trajectory()):
        np.testing.assert_array_equal(o1, o2)
        np.testing.assert_array_equal(s1, s2)
        assert r1 == r2


def test_step_after_horizon_raises():
    cfg = running_example(horizon=3)
    env = PlantedCoalitionGridworld(cfg)
    for _ in range(3):
        env.step([NOOP] * 5)
    with pytest.raises(RuntimeError):
        env.step([NOOP] * 5)


# --- Config validation -------------------------------------------------------------


def test_config_rejects_enabler_that_is_also_finisher():
    bad = EventSpec(enablers=(A,), finishers=(A, B), reward=1.0,
                    lever_cells={A: (0, 0)}, goal_cells={A: (1, 1), B: (2, 2)})
    with pytest.raises(ValueError):
        replace(running_example(), events=(bad,))


def test_config_rejects_missing_station():
    bad = EventSpec(enablers=(A,), finishers=(B,), reward=1.0, lever_cells={}, goal_cells={B: (2, 2)})
    with pytest.raises(ValueError):
        replace(running_example(), events=(bad,))


def test_config_reports_bystanders():
    assert running_example().bystanders == (D, E)
    assert overlap_example().bystanders == (E,)
