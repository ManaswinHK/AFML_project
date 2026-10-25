import numpy as np
import pytest

from whoearned.envs import NOOP, PlantedCoalitionGridworld, running_example
from whoearned.replay import _Intervention, step_counterfactuals
from whoearned.scripted import ScriptedTeam

A, B, C, D, E = range(5)
GAMMA = 0.99


def _discounted_to_go(config, team, t0):
    env = PlantedCoalitionGridworld(config)
    env.reset()
    team.reset(env)
    g = 0.0
    for t in range(config.horizon):
        _, r, _, _ = env.step(team.act(env))
        if t >= t0:
            g += GAMMA ** (t - t0) * r
    return g


@pytest.fixture(scope="module")
def deterministic():
    cfg = running_example(delay=15)
    return cfg, step_counterfactuals(cfg, lambda: ScriptedTeam(cfg, bystander_mode="idle"), GAMMA)


def test_shapes_and_factual_entries(deterministic):
    cfg, st = deterministic
    assert st.returns.shape == (cfg.horizon, 5, 6)
    assert st.factual == 10.0
    taken = np.take_along_axis(st.returns, st.actions[..., None], axis=2)[..., 0]
    np.testing.assert_allclose(taken, np.repeat(st.factual_to_go[:, None], 5, axis=1))


def test_finisher_noop_at_completion_costs_one_step_of_discounting(deterministic):
    # B and C complete at t=8 (payout t=23). If B waits one step, completion and payout slip by one.
    _, st = deterministic
    expected = 10 * GAMMA ** 15 * (1 - GAMMA)
    assert st.noop_effect()[8, B] == pytest.approx(expected)
    assert st.noop_effect()[8, C] == pytest.approx(expected)


def test_enabler_with_slack_gets_zero_per_step_credit(deterministic):
    # A pulls at t=4 but the finishers are not ready until t=8: one step of delay costs nothing.
    _, st = deterministic
    np.testing.assert_allclose(st.noop_effect()[:, A], 0.0, atol=1e-12)


def test_enabler_on_the_critical_path_gets_credit():
    # A starts late (t=20), so the finishers wait for the gate: delaying the pull delays everything.
    cfg = running_example(delay=15)
    st = step_counterfactuals(cfg, lambda: ScriptedTeam(cfg, enabler_start=20, bystander_mode="idle"), GAMMA)
    assert st.noop_effect()[:, A].sum() > 0


def test_bystanders_have_zero_effect(deterministic):
    _, st = deterministic
    np.testing.assert_allclose(st.noop_effect()[:, [D, E]], 0.0, atol=1e-12)


def test_baseline_effect_is_zero_for_a_deterministic_policy(deterministic):
    # pi is one-hot on the taken action, so G - sum_a pi(a) G(a) = 0 everywhere.
    _, st = deterministic
    np.testing.assert_allclose(st.baseline_effect(), 0.0, atol=1e-12)


def test_early_stop_matches_brute_force_rollouts():
    cfg = running_example(delay=10)
    make = lambda: ScriptedTeam(cfg, seed=4, epsilon=0.2)  # noqa: E731
    st = step_counterfactuals(cfg, make, GAMMA)
    rng = np.random.default_rng(0)
    for _ in range(25):
        t, i, a = int(rng.integers(0, 20)), int(rng.integers(5)), int(rng.integers(6))
        brute = _discounted_to_go(cfg, _Intervention(make(), t, i, a), t)
        assert st.returns[t, i, a] == pytest.approx(brute), (t, i, a)
