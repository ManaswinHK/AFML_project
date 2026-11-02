"""H5: the low-order episodic coalition estimator (structure, not learning quality)."""

import itertools

import numpy as np
import torch as th

from whoearned.coalition_estimator import CoalitionModel, own_feature_index
from whoearned.envs import PlantedCoalitionGridworld, coalition_layout, overlap_example
from whoearned.ground_truth import agents_of, harsanyi_dividends, shapley_values, values_from_dividends


def test_own_features_ignore_other_agents_and_the_environment():
    cfg = coalition_layout(2, delay=0)
    idx = own_feature_index(cfg.n_agents, len(cfg.events))
    env = PlantedCoalitionGridworld(cfg)
    env.reset()
    before = env.get_obs_agent(3)[idx]
    env.positions[0] = (1, 1)   # move A
    env.positions[3] = (4, 9)   # move the agent itself: positions are excluded too
    np.testing.assert_array_equal(env.get_obs_agent(3)[idx], before)


def test_own_features_are_the_identity():
    cfg = overlap_example(delay=0)
    env = PlantedCoalitionGridworld(cfg)
    env.reset()
    idx = own_feature_index(cfg.n_agents, len(cfg.events))
    np.testing.assert_array_equal(env.get_obs_agent(2)[idx], np.eye(cfg.n_agents)[2])


def _model(n=4, k=2, d=8):
    th.manual_seed(0)
    return CoalitionModel(feat_dim=3, n_actions=2, n_agents=n, k=k, d=d)


def test_terms_cover_all_subsets_up_to_order_k():
    m = _model(n=4, k=3)
    assert [len(s) for s in m.subsets] == [4, 6, 4]
    assert m.subsets[1] == list(itertools.combinations(range(4), 2))


def test_absent_agents_zero_their_terms_only():
    m = _model(n=4, k=2)
    h = th.randn(2, 4, 8)
    present = th.tensor([[1.0, 1, 1, 1], [1, 0, 1, 1]])
    singles, pairs = m.terms(h, present)
    assert singles[1, 1] == 0 and singles[1, 0] != 0
    for j, (a, b) in enumerate(m.subsets[1]):
        assert (pairs[1, j] == 0) == (1 in (a, b))
    full_singles, full_pairs = m.terms(h, th.ones(2, 4))
    th.testing.assert_close(singles[0], full_singles[0])
    th.testing.assert_close(pairs[0], full_pairs[0])


def test_terms_are_symmetric_in_members():
    m = _model(n=3, k=2)
    h = th.randn(1, 3, 8)
    _, pairs = m.terms(h, th.ones(1, 3))
    _, swapped = m.terms(h[:, [1, 0, 2]], th.ones(1, 3))
    th.testing.assert_close(pairs[0, 0], swapped[0, 0])  # {0, 1} is the same pair either way


def test_term_game_has_the_terms_as_dividends_and_shapley_as_equal_split():
    # v_hat(M) = sum of terms inside M, so its dividends are the terms and Shapley splits each equally.
    rng = np.random.default_rng(0)
    n = 4
    d = np.zeros(2 ** n)
    for s in range(1, 2 ** n):
        if len(agents_of(s)) <= 2:
            d[s] = rng.normal()
    v = values_from_dividends(d)
    np.testing.assert_allclose(harsanyi_dividends(v), d, atol=1e-12)
    split = np.zeros(n)
    for s in range(1, 2 ** n):
        members = agents_of(s)
        split[list(members)] += d[s] / len(members)
    np.testing.assert_allclose(shapley_values(v), split, atol=1e-12)
