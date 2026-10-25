from itertools import combinations

import numpy as np
import pytest
from scipy.stats import spearmanr

from whoearned.envs import overlap_example, running_example
from whoearned.metrics import (
    chance_jaccard,
    coalition_recovery,
    credit_error,
    dividend_error,
    evaluate_credit,
    identified_coalitions,
    jaccard,
    leakage,
    missed_credit,
    realised_bystanders,
    return_faithfulness,
    spearman,
    true_dividends,
)
from whoearned.replay import ground_truth_open_loop, rollout
from whoearned.scripted import ScriptedTeam

A, B, C, D, E = range(5)
fs = frozenset


@pytest.fixture(scope="module")
def running():
    cfg = running_example()
    return cfg, ground_truth_open_loop(cfg, rollout(cfg, ScriptedTeam(cfg)).actions)


@pytest.fixture(scope="module")
def overlap():
    cfg = overlap_example()
    return cfg, ground_truth_open_loop(cfg, rollout(cfg, ScriptedTeam(cfg)).actions)


# --- evaluate_credit on the running example (target shares 1/3, 1/3, 1/3, 0, 0) -----------


def test_perfect_credit(running):
    cfg, truth = running
    r = evaluate_credit(np.array([3.0, 3, 3, 0, 0]), truth, cfg)
    assert r.valid
    assert r.l1 == pytest.approx(0.0)
    assert r.spearman == pytest.approx(1.0)
    assert r.leakage == pytest.approx(0.0)
    assert r.missed == pytest.approx(0.0)


def test_shared_signal_credit_is_maximal_leakage_reference(running):
    # MAPPO-style: everyone gets the same credit -> shares 0.2 each.
    cfg, truth = running
    r = evaluate_credit(np.ones(5), truth, cfg)
    assert r.l1 == pytest.approx(3 * (1 / 3 - 0.2) + 2 * 0.2)  # 0.8
    assert r.leakage == pytest.approx(0.4)
    assert r.missed == pytest.approx(1 - 0.2 / (1 / 3))      # 0.4
    assert r.spearman is None                                # constant credit has no ranking


def test_proximity_credit_forgets_the_enabler(running):
    # Only the finishers, who were near the crate, get credit.
    cfg, truth = running
    r = evaluate_credit(np.array([0.0, 5, 5, 0, 0]), truth, cfg)
    assert r.l1 == pytest.approx(1 / 3 + 2 * (0.5 - 1 / 3))  # 2/3
    assert r.missed == pytest.approx(1.0)
    assert r.leakage == pytest.approx(0.0)


def test_negative_credit_counts_as_leakage_by_magnitude(running):
    cfg, truth = running
    r = evaluate_credit(np.array([3.0, 3, 3, -1, 1]), truth, cfg)
    assert r.leakage == pytest.approx(2 / 11)


def test_leave_one_out_target(running):
    cfg, truth = running
    r = evaluate_credit(np.array([10.0, 10, 10, 0, 0]), truth, cfg, target="loo")
    assert r.l1 == pytest.approx(0.0)


def test_zero_credit_is_reported_invalid(running):
    cfg, truth = running
    r = evaluate_credit(np.zeros(5), truth, cfg)
    assert not r.valid and r.l1 is None


def test_unknown_target_raises(running):
    cfg, truth = running
    with pytest.raises(ValueError):
        evaluate_credit(np.ones(5), truth, cfg, target="banzhaf")


def test_per_event_evaluation_uses_event_roles(overlap):
    # Event 1 = enabler D, finisher B. Agents A, C, E are outsiders for this event.
    cfg, truth = overlap
    perfect = evaluate_credit(np.array([0.0, 1, 0, 1, 0]), truth, cfg, event=1)
    assert perfect.l1 == pytest.approx(0.0)
    assert perfect.missed == pytest.approx(0.0)
    assert perfect.leakage == pytest.approx(0.0)

    leaky = evaluate_credit(np.array([1.0, 1, 0, 1, 0]), truth, cfg, event=1)  # A belongs to event 0 only
    assert leaky.leakage == pytest.approx(1 / 3)


# --- individual metric functions ------------------------------------------------------


def test_credit_error_bounds():
    assert credit_error(np.array([1.0, 0]), np.array([0.0, 1])) == pytest.approx(2.0)


def test_missed_credit_is_clipped_at_zero_and_undefined_without_enablers():
    target = np.array([0.5, 0.5])
    assert missed_credit(np.array([0.9, 0.1]), target, [0]) == 0.0      # over-credited enabler
    assert missed_credit(np.array([0.9, 0.1]), np.array([0.0, 1.0]), [0]) is None


def test_leakage_sums_bystanders():
    assert leakage(np.array([0.5, 0.2, -0.3]), [1, 2]) == pytest.approx(0.5)


@pytest.mark.parametrize("seed", range(5))
def test_spearman_matches_scipy(seed):
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 4, size=12).astype(float)  # forces ties
    y = rng.normal(size=12)
    assert spearman(x, y) == pytest.approx(spearmanr(x, y).statistic)


def test_spearman_constant_is_none():
    assert spearman([1, 1, 1], [1, 2, 3]) is None


# --- coalition-level -------------------------------------------------------------------


def test_jaccard():
    assert jaccard({A, B, C}, {B, C}) == pytest.approx(2 / 3)
    assert jaccard({A}, {B}) == 0.0
    assert jaccard(set(), set()) == 1.0


@pytest.mark.parametrize("guess_size", range(0, 6))
def test_chance_jaccard_matches_enumeration(guess_size):
    n, true = 5, {A, B, C}
    guesses = list(combinations(range(n), guess_size))
    brute = np.mean([jaccard(true, g) for g in guesses])
    assert chance_jaccard(n, 3, guess_size) == pytest.approx(brute)


def test_coalition_recovery_recall_and_precision():
    planted = [{A, B, C}]
    r = coalition_recovery([{B, C}], planted)
    assert (r.recall, r.precision) == pytest.approx((2 / 3, 2 / 3))

    r = coalition_recovery([{A, B, C}, {D}], planted)
    assert (r.recall, r.precision) == pytest.approx((1.0, 0.5))  # spurious {D} costs precision

    r = coalition_recovery([], planted)
    assert (r.recall, r.precision) == (0.0, 1.0)


def test_identified_coalitions_thresholds_on_share():
    est = {fs({A, B, C}): 9.0, fs({D}): 0.5, fs({B, C}): 0.5}
    assert identified_coalitions(est, threshold=0.1) == [fs({A, B, C})]
    assert identified_coalitions({}, 0.1) == []


def test_true_dividends_total_and_per_event(overlap):
    _, truth = overlap
    assert true_dividends(truth) == {fs({A, B, C}): pytest.approx(6.0), fs({B, D}): pytest.approx(4.0)}
    assert true_dividends(truth, event=1) == {fs({B, D}): pytest.approx(4.0)}


def test_dividend_error_tells_synergy_from_independence():
    # G4: Shapley cannot tell these apart; dividend error can.
    synergy = {fs({A, B}): 2.0}
    independent = {fs({A}): 1.0, fs({B}): 1.0}
    assert dividend_error(independent, synergy) == pytest.approx(2.0)  # maximal
    assert dividend_error(synergy, synergy) == pytest.approx(0.0)


def test_dividend_error_normalisation_and_order():
    truth = {fs({A, B, C}): 10.0}
    assert dividend_error({fs({A, B, C}): 3.0}, truth) == pytest.approx(0.0)
    assert dividend_error({fs({A, B, C}): 3.0}, truth, normalise=False) == pytest.approx(7.0)
    # A pairwise-only model scored up to order 2: the true triple is invisible -> undefined.
    assert dividend_error({fs({A, B}): 5.0}, truth, max_order=2) is None


# --- across runs ------------------------------------------------------------------------


def test_return_faithfulness_within_and_pooled():
    returns = [1, 2, 3, 1, 2, 3]
    errors = [0.3, 0.2, 0.1, 0.1, 0.2, 0.3]
    groups = ["X", "X", "X", "Y", "Y", "Y"]
    out = return_faithfulness(returns, errors, groups)
    assert out["X"] == pytest.approx(1.0)
    assert out["Y"] == pytest.approx(-1.0)
    assert out["pooled"] == pytest.approx(0.0)  # the two methods cancel when pooled


def test_return_faithfulness_needs_three_runs_per_group():
    out = return_faithfulness([1, 2], [0.1, 0.2], ["X", "X"])
    assert out["X"] is None


# --- realised roles -------------------------------------------------------------------------


def test_realised_bystanders(running):
    _, truth = running
    assert realised_bystanders(truth) == [D, E]


def test_realised_bystanders_drop_an_obstructing_bystander():
    cfg = running_example()
    ep = rollout(cfg, ScriptedTeam(cfg, seed=7, bystander_mode="roam"))
    truth = ground_truth_open_loop(cfg, ep.actions)
    assert D not in realised_bystanders(truth)
