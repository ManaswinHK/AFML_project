"""Identities of persistent-ablation credit (docs/v3/theory.md), checked exactly on real episodes."""

import numpy as np
import pytest

from whoearned import ablation as ab
from whoearned import ground_truth as gt
from whoearned.envs import PlantedCoalitionGridworld, coalition_layout, overlap_example
from whoearned.replay import ground_truth_closed_loop
from whoearned.scripted import ScriptedTeam


def _setup(layout="coalition", delay=10, seed=0, eps=0.2, b=2):
    cfg = overlap_example(delay=delay) if layout == "overlap" else coalition_layout(
        b, delay=delay, partial=layout == "partial")
    make = lambda: ScriptedTeam(cfg, seed=seed, epsilon=eps)
    return cfg, make, ab.factual_rollout(cfg, make)


@pytest.mark.parametrize("layout", ["coalition", "overlap", "partial"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_identity1_ablation_td_sums_to_realised_leave_one_out_closed_loop(layout, seed):
    """Removal null, stop at t = 0 is reset(present=N-i): sum_t kappa_i = G - v_closed(N - i)."""
    cfg, make, f = _setup(layout, seed=seed)
    W = ab.oracle_ablation_values(cfg, make, f, null="remove")
    kappa = ab.ablation_td_credit(f.rewards, W)
    truth = ground_truth_closed_loop(cfg, make)
    np.testing.assert_allclose(kappa.sum(1), truth.leave_one_out, atol=1e-9)
    np.testing.assert_allclose(f.total, truth.v[-1])


@pytest.mark.parametrize("null", ["noop", "remove"])
def test_commitment_split_is_exact(null):
    cfg, make, f = _setup(seed=3)
    W = ab.oracle_ablation_values(cfg, make, f, null=null)
    V = f.to_go()      # oracle full-team value under common random numbers = realised return-to-go
    delta, c = ab.commitment_split(f.rewards, V, W)
    np.testing.assert_allclose(delta[None] + c, ab.ablation_td_credit(f.rewards, W), atol=1e-12)
    np.testing.assert_allclose(c.sum(1), V[0] - W[:, 0], atol=1e-12)


@pytest.mark.parametrize("layout", ["coalition", "overlap", "partial"])
def test_pending_game_step_credit_sums_to_closed_loop_shapley_and_dividends(layout):
    cfg, make, f = _setup(layout, seed=1)
    Wall = ab.all_masks_values(cfg, make, f, null="remove")
    out = ab.step_semivalue_credit(Wall)
    truth = ground_truth_closed_loop(cfg, make)
    np.testing.assert_allclose(Wall[:, 0], truth.v, atol=1e-9)
    np.testing.assert_allclose(out["shapley"].sum(1), truth.shapley, atol=1e-9)
    np.testing.assert_allclose(out["dividends"].sum(1), truth.dividends, atol=1e-9)
    # Shapley step credit is efficient at every step: it splits the value committed at that step.
    np.testing.assert_allclose(out["shapley"].sum(0), out["committed"], atol=1e-9)


@pytest.mark.parametrize("delay", [0, 25])
def test_step_credit_lands_on_commitment_not_payout(delay):
    """No-op null, deterministic team: enabler credited at its pull, finishers at the joint finish,
    bystanders never; the payout step (delay later) carries no structural credit."""
    cfg, make, f = _setup(delay=delay, eps=0.0)
    rec = f.event_log[0]
    t_pull, t_fin, t_pay = rec.pull_times[0], rec.completion_time, rec.payout_time
    Wall = ab.all_masks_values(cfg, make, f, null="noop")
    out = ab.step_semivalue_credit(Wall)
    loo = out["loo"]
    assert loo[0, t_pull] == pytest.approx(10) and np.abs(np.delete(loo[0], t_pull)).max() == 0
    for i in (1, 2):
        assert loo[i, t_fin] == pytest.approx(10) and np.abs(np.delete(loo[i], t_fin)).max() == 0
    assert np.abs(loo[3:]).max() == 0 and np.abs(out["shapley"][3:]).max() == 0
    assert out["committed"][t_fin] == pytest.approx(10)
    if delay:
        assert out["committed"][t_pay] == 0
    # Coalition timeline: {A,B,C} is released at the pull and {B,C} at the finish.
    abc, bc = gt.mask_of((0, 1, 2)), gt.mask_of((1, 2))
    assert out["dividends"][abc, t_pull] == pytest.approx(10)
    assert out["dividends"][bc, t_pull] == pytest.approx(-10)
    assert out["dividends"][bc, t_fin] == pytest.approx(10)


@pytest.mark.parametrize("null", ["noop", "remove"])
def test_stopped_state_matches_what_the_environment_reports(null):
    cfg, make, f = _setup(seed=2)
    for t in (0, 5, 17):
        env, _ = f.snaps[t]
        env = env.clone()
        env.stop((1, 3), null)
        np.testing.assert_array_equal(ab.stopped_state(f.states[t], cfg.n_agents, (1, 3), null), env.get_state())


def test_dataset_without_interventions_is_exactly_the_training_episodes():
    cfg = coalition_layout(2, delay=10)
    makes = [lambda i=i: ScriptedTeam(cfg, seed=100 + i, epsilon=0.2) for i in range(4)]
    facts = [ab.factual_rollout(cfg, m) for m in makes]
    d = ab.intervention_dataset(cfg, lambda i: makes[i], facts, n_int=0, seed=0)
    np.testing.assert_array_equal(d.X, np.concatenate([f.states[:-1] for f in facts]).astype(np.float32))
    np.testing.assert_array_equal(d.y, np.concatenate([f.to_go()[:-1] for f in facts]).astype(np.float32))
    d2 = ab.intervention_dataset(cfg, lambda i: makes[i], facts, n_int=12, seed=0, sampler="stratified")
    assert d2.is_intervention.sum() > 0 and len(d2.interventions) == 12
    for idx, t0, stopped, g in d2.interventions:
        assert g == pytest.approx(ab.continue_stopped(cfg, makes[idx], facts[idx].snaps[t0], stopped, "noop"))


def test_stratified_sampler_covers_the_payout_wait():
    cfg = coalition_layout(2, delay=25)
    f = ab.factual_rollout(cfg, lambda: ScriptedTeam(cfg, seed=0, epsilon=0.2))
    rng = np.random.default_rng(0)
    ts = np.array([ab.sample_stop_time(rng, f, "stratified") for _ in range(2000)])
    assert 0.4 < np.mean(ts >= f.t_done) < 0.6 and ts.max() < f.t_last_payout


def test_learned_readout_with_a_perfect_lookup_critic_reproduces_the_oracle():
    """If the critic were exact on stopped states, the learned read-out equals the oracle."""
    cfg, make, f = _setup(seed=4)
    W = ab.oracle_ablation_values(cfg, make, f, null="noop")
    n = cfg.n_agents
    table = {}
    for i in range(n):
        for t in range(cfg.horizon):
            table[ab.stopped_state(f.states[t], n, (i,), "noop").tobytes()] = W[i, t]
    predict = lambda S: np.array([table[s.astype(np.float32).tobytes()] for s in S])
    np.testing.assert_allclose(ab.learned_ablation_values(predict, f, null="noop"), W)
