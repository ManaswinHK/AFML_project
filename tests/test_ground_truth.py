import numpy as np
import pytest

from whoearned.ground_truth import (
    agents_of,
    coalition_values,
    harsanyi_dividends,
    leave_one_out,
    mask_of,
    n_agents,
    nonzero_dividends,
    normalise_shares,
    shapley_values,
    shapley_values_bruteforce,
    values_from_dividends,
)

A, B, C, D, E = range(5)


def planted_events(events: list[tuple[set[int], float]]):
    """v(S) for all-or-nothing events: event k pays r_k iff its whole coalition is in S."""
    masks = [(mask_of(g), r) for g, r in events]
    return lambda s: sum(r for g, r in masks if s & g == g)


# --- Worked examples from the problem statement ---------------------------------


def test_running_example():
    # Section 1: A enables, B and C finish, D and E are bystanders; payout +10.
    v = coalition_values(planted_events([({A, B, C}, 10.0)]), 5)

    assert nonzero_dividends(v) == {(A, B, C): pytest.approx(10.0)}
    np.testing.assert_allclose(shapley_values(v), [10 / 3, 10 / 3, 10 / 3, 0, 0])
    np.testing.assert_allclose(leave_one_out(v), [10, 10, 10, 0, 0])
    # Normalised, both targets give {A, B, C} one third each.
    np.testing.assert_allclose(normalise_shares(shapley_values(v)), [1 / 3] * 3 + [0, 0])
    np.testing.assert_allclose(normalise_shares(leave_one_out(v)), [1 / 3] * 3 + [0, 0])


def test_g4_synergy_vs_independence_have_same_shapley():
    # Section 3, G4: identical Shapley values, opposite coordination structure.
    synergy = np.array([0.0, 0.0, 0.0, 2.0])      # v(empty), v(A), v(B), v(AB)
    independent = np.array([0.0, 1.0, 1.0, 2.0])

    np.testing.assert_allclose(shapley_values(synergy), [1, 1])
    np.testing.assert_allclose(shapley_values(independent), [1, 1])
    assert nonzero_dividends(synergy) == {(A, B): pytest.approx(2.0)}
    assert nonzero_dividends(independent) == {(A,): pytest.approx(1.0), (B,): pytest.approx(1.0)}


# --- Where the two individual targets diverge ----------------------------------


def test_overlap_makes_leave_one_out_and_shapley_shares_differ():
    # B belongs to two coalitions of different sizes.
    v = coalition_values(planted_events([({A, B}, 6.0), ({B, C, D}, 6.0)]), 4)

    np.testing.assert_allclose(shapley_values(v), [3, 5, 2, 2])
    np.testing.assert_allclose(leave_one_out(v), [6, 12, 6, 6])
    np.testing.assert_allclose(normalise_shares(shapley_values(v)), [0.25, 5 / 12, 1 / 6, 1 / 6])
    np.testing.assert_allclose(normalise_shares(leave_one_out(v)), [0.2, 0.4, 0.2, 0.2])


def test_partial_synergy():
    # A alone earns 4; A with B earns 6. B adds value only alongside A.
    v = np.array([0.0, 4.0, 0.0, 6.0])

    assert nonzero_dividends(v) == {(A,): pytest.approx(4.0), (A, B): pytest.approx(2.0)}
    np.testing.assert_allclose(shapley_values(v), [5, 1])
    np.testing.assert_allclose(leave_one_out(v), [6, 2])


# --- Properties on random games -------------------------------------------------


@pytest.mark.parametrize("n", range(1, 7))
def test_dividend_based_shapley_matches_bruteforce(n):
    rng = np.random.default_rng(n)
    v = rng.normal(size=1 << n)
    v[0] = 0.0
    np.testing.assert_allclose(shapley_values(v), shapley_values_bruteforce(v), atol=1e-10)


@pytest.mark.parametrize("n", range(1, 8))
def test_moebius_round_trip(n):
    rng = np.random.default_rng(100 + n)
    v = rng.normal(size=1 << n)
    np.testing.assert_allclose(values_from_dividends(harsanyi_dividends(v)), v, atol=1e-9)


@pytest.mark.parametrize("n", range(1, 9))
def test_shapley_efficiency(n):
    rng = np.random.default_rng(200 + n)
    v = rng.normal(size=1 << n)
    assert shapley_values(v).sum() == pytest.approx(v[-1] - v[0])


def test_bystanders_get_zero_under_all_targets():
    # Agent E (index 4) never appears in any event: a null player.
    v = coalition_values(planted_events([({A, B}, 3.0), ({C, D}, 5.0)]), 5)
    assert shapley_values(v)[E] == pytest.approx(0.0)
    assert leave_one_out(v)[E] == pytest.approx(0.0)
    assert all(E not in g for g in nonzero_dividends(v))


# --- Helpers --------------------------------------------------------------------


def test_mask_round_trip():
    assert mask_of([A, B, C]) == 0b111
    assert agents_of(0b10110) == (1, 2, 4)
    assert agents_of(mask_of([0, 3, 7])) == (0, 3, 7)


def test_normalise_shares_rejects_near_zero_total():
    assert normalise_shares(np.array([1e-12, -1e-12])) is None
    np.testing.assert_allclose(normalise_shares(np.array([2.0, -2.0])), [0.5, -0.5])


def test_bad_length_raises():
    with pytest.raises(ValueError):
        n_agents(np.zeros(6))
