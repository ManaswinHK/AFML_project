from dataclasses import replace

import numpy as np
import pytest

from whoearned.envs import UP, PlantedCoalitionGridworld, overlap_example, running_example
from whoearned.ground_truth import mask_of
from whoearned.replay import (
    ground_truth_closed_loop,
    ground_truth_open_loop,
    planted_dividends,
    replay,
    rollout,
)
from whoearned.scripted import ScriptedTeam

A, B, C, D, E = range(5)
ALL5 = 0b11111
RUNNING_DIVIDENDS = {(A, B, C): pytest.approx(10.0)}
OVERLAP_DIVIDENDS = {(A, B, C): pytest.approx(6.0), (B, D): pytest.approx(4.0)}


def team_factory(cfg, **kw):
    return lambda: ScriptedTeam(cfg, **kw)


# --- Scripted team behaviour ----------------------------------------------------


def test_scripted_team_completes_running_example():
    cfg = running_example(delay=15)
    ep = rollout(cfg, ScriptedTeam(cfg, bystander_mode="idle"))
    rec = ep.event_log[0]
    assert ep.total == 10.0
    assert rec.pull_times == {A: 4}
    assert rec.completion_time == 8
    assert rec.payout_time == 23


def test_finish_not_before_sets_the_enabler_finisher_gap():
    cfg = running_example(delay=5)
    ep = rollout(cfg, ScriptedTeam(cfg, finish_not_before=30, bystander_mode="idle"))
    rec = ep.event_log[0]
    assert rec.pull_times[A] == 4
    assert rec.completion_time == 30
    assert ep.total == 10.0


def test_late_enabler_makes_finishers_wait_for_the_gate():
    cfg = running_example(delay=5)
    ep = rollout(cfg, ScriptedTeam(cfg, enabler_start=20, bystander_mode="idle"))
    rec = ep.event_log[0]
    assert rec.pull_times[A] == 24
    assert rec.completion_time == 25  # gate usable from the step after the pull


def test_scripted_team_completes_overlap_example():
    cfg = overlap_example()
    ep = rollout(cfg, ScriptedTeam(cfg))
    assert ep.total == 10.0
    assert all(rec.completion_time is not None for rec in ep.event_log)


def test_rollout_is_reproduced_exactly_by_open_loop_replay():
    cfg = running_example()
    ep = rollout(cfg, ScriptedTeam(cfg, seed=7, epsilon=0.2))
    rep = replay(cfg, ep.actions)
    np.testing.assert_array_equal(rep.actions, ep.actions)
    np.testing.assert_array_equal(rep.rewards, ep.rewards)


def test_same_seed_same_episode_different_seed_different_episode():
    cfg = running_example()
    a = rollout(cfg, ScriptedTeam(cfg, seed=1, epsilon=0.3)).actions
    b = rollout(cfg, ScriptedTeam(cfg, seed=1, epsilon=0.3)).actions
    c = rollout(cfg, ScriptedTeam(cfg, seed=2, epsilon=0.3)).actions
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_random_streams_do_not_depend_on_who_is_present():
    cfg = running_example()
    full, without_e = ScriptedTeam(cfg, seed=5, epsilon=0.3), ScriptedTeam(cfg, seed=5, epsilon=0.3)
    rollout(cfg, full)
    rollout(cfg, without_e, ALL5 & ~(1 << E))
    for j in range(5):
        assert full.rngs[j].bit_generator.state == without_e.rngs[j].bit_generator.state


# --- Replay ground truth equals the planted structure ---------------------------


@pytest.mark.parametrize("seed", range(10))
def test_open_loop_ground_truth_is_planted_running_example(seed):
    cfg = running_example()
    ep = rollout(cfg, ScriptedTeam(cfg, seed=seed))  # bystanders wander
    truth = ground_truth_open_loop(cfg, ep.actions)
    assert truth.nonzero_dividends() == RUNNING_DIVIDENDS
    np.testing.assert_allclose(truth.shapley, [10 / 3] * 3 + [0, 0])
    np.testing.assert_allclose(truth.leave_one_out, [10, 10, 10, 0, 0])


def test_wandering_bystanders_stay_off_coalition_routes():
    cfg = running_example()
    team = ScriptedTeam(cfg, seed=3)
    ep = rollout(cfg, team)
    env = PlantedCoalitionGridworld(cfg)
    env.reset()
    for t in range(cfg.horizon):
        env.step(ep.actions[t])
        assert env.positions[D] not in team.reserved
        assert env.positions[E] not in team.reserved


def test_roaming_bystander_can_change_realised_roles():
    # Seed 7: a roaming bystander strays into C's lane. Without D (who used to
    # block E), E's recorded moves now put it in C's way, so v(N \ D) = 0:
    # D is a planted bystander with non-zero realised credit.
    cfg = running_example()
    ep = rollout(cfg, ScriptedTeam(cfg, seed=7, bystander_mode="roam"))
    truth = ground_truth_open_loop(cfg, ep.actions)
    assert truth.v[ALL5] == 10.0
    assert truth.v[ALL5 & ~(1 << D)] == 0.0
    assert truth.nonzero_dividends() != RUNNING_DIVIDENDS


def test_open_loop_ground_truth_is_planted_overlap_example():
    cfg = overlap_example()
    truth = ground_truth_open_loop(cfg, rollout(cfg, ScriptedTeam(cfg)).actions)
    assert truth.nonzero_dividends() == OVERLAP_DIVIDENDS
    np.testing.assert_allclose(truth.shapley, [2, 4, 2, 2, 0])


@pytest.mark.parametrize(
    "cfg, expected",
    [(running_example(), RUNNING_DIVIDENDS), (overlap_example(), OVERLAP_DIVIDENDS)],
)
def test_closed_loop_ground_truth_is_planted(cfg, expected):
    truth = ground_truth_closed_loop(cfg, team_factory(cfg, seed=0))
    assert truth.nonzero_dividends() == expected


def test_closed_loop_partial_synergy():
    cfg = running_example(partial=True)
    truth = ground_truth_closed_loop(cfg, team_factory(cfg, bystander_mode="idle"))
    assert truth.v[mask_of([A, B])] == 5.0
    assert truth.nonzero_dividends() == {(A, B): pytest.approx(5.0), (A, C): pytest.approx(5.0)}
    np.testing.assert_allclose(truth.shapley, [5, 2.5, 2.5, 0, 0])


def test_per_event_ground_truth_separates_overlapping_events():
    cfg = overlap_example()
    truth = ground_truth_open_loop(cfg, rollout(cfg, ScriptedTeam(cfg)).actions)

    np.testing.assert_allclose(truth.v_events.sum(axis=0), truth.v)
    np.testing.assert_allclose(truth.event_shapley[0], [2, 2, 2, 0, 0])
    np.testing.assert_allclose(truth.event_shapley[1], [0, 2, 0, 2, 0])
    np.testing.assert_allclose(truth.event_leave_one_out[0], [6, 6, 6, 0, 0])
    np.testing.assert_allclose(truth.event_leave_one_out[1], [0, 4, 0, 4, 0])


@pytest.mark.parametrize("delay", [0, 25, 50])
def test_delay_does_not_change_ground_truth(delay):
    cfg = running_example(delay=delay, horizon=70)
    truth = ground_truth_open_loop(cfg, rollout(cfg, ScriptedTeam(cfg)).actions)
    assert truth.nonzero_dividends() == RUNNING_DIVIDENDS
    assert not truth.clipped


def test_ground_truth_flags_clipped_payouts():
    cfg = running_example(delay=15, horizon=20)
    truth = ground_truth_open_loop(cfg, rollout(cfg, ScriptedTeam(cfg)).actions)
    assert truth.clipped


# --- Null-policy protocols ----------------------------------------------------------


def _obstruction_case():
    """E starts in B's lane at (2,4) and steps out at t=0; B walks straight through."""
    base = running_example()
    actions = rollout(base, ScriptedTeam(base, bystander_mode="idle")).actions
    starts = list(base.start_cells)
    starts[E] = (2, 4)
    cfg = replace(base, start_cells=tuple(starts))
    actions[:, E] = 0
    actions[0, E] = UP
    return cfg, actions


def test_noop_null_policy_can_obstruct_where_removal_does_not():
    cfg, actions = _obstruction_case()
    without_e = ALL5 & ~(1 << E)

    assert replay(cfg, actions).total == 10.0
    assert replay(cfg, actions, without_e, null="remove").total == 10.0
    assert replay(cfg, actions, without_e, null="noop").total == 0.0  # E stands in B's way

    removal = ground_truth_open_loop(cfg, actions, null="remove")
    noop = ground_truth_open_loop(cfg, actions, null="noop")
    assert removal.shapley[E] == pytest.approx(0.0)
    assert noop.shapley[E] > 0  # E "earns" credit just for stepping aside


def test_random_null_policy_is_deterministic_given_seed():
    cfg, actions = _obstruction_case()
    a = ground_truth_open_loop(cfg, actions, null="random", null_seed=3)
    b = ground_truth_open_loop(cfg, actions, null="random", null_seed=3)
    np.testing.assert_array_equal(a.v, b.v)


@pytest.mark.parametrize(
    "cfg",
    [running_example(), overlap_example(), running_example(partial=True)],
    ids=["running", "overlap", "partial"],
)
def test_planted_dividends_match_noise_free_replay(cfg):
    truth = ground_truth_open_loop(cfg, rollout(cfg, ScriptedTeam(cfg)).actions)
    assert truth.matches(planted_dividends(cfg))


def test_planted_dividends_of_partial_event():
    assert planted_dividends(running_example(partial=True)) == {(A, B): 5.0, (A, C): 5.0}


def test_matches_detects_realised_deviation():
    cfg = running_example()
    ep = rollout(cfg, ScriptedTeam(cfg, seed=7, bystander_mode="roam"))
    assert not ground_truth_open_loop(cfg, ep.actions).matches(planted_dividends(cfg))


def test_unknown_null_policy_raises():
    cfg = running_example()
    with pytest.raises(ValueError):
        replay(cfg, np.zeros((cfg.horizon, 5), dtype=int), null="teleport")
