"""Checks behind the v5 corrections (docs/v5/review_response.md)."""
import numpy as np
import torch as th

from whoearned.critics import SetCritic, StateTokens
from whoearned.envs.gridworld import PlantedCoalitionGridworld
from whoearned.experiments import DataCell
from whoearned import ground_truth as gt


def test_shuffled_roles_are_a_relabelling_the_set_critic_cannot_see():
    """Errata E12: a shuffled-roles state is a relabelling of the fixed-layout state (start cells move with roles),
    so ANY set critic gives the same value: v3's P8 shuffled-roles results hold by construction."""
    fixed = DataCell(delay=25, n_bystanders=2, seed=5, horizon=60)
    perm = DataCell(delay=25, n_bystanders=2, seed=5, horizon=60, shuffle_roles=True)
    n = fixed.config().n_agents
    th.manual_seed(0)
    nets = [SetCritic(StateTokens(n, 1, roles=r)) for r in (True, False)]
    for i in range(10):
        ef, ep = PlantedCoalitionGridworld(fixed.config()), PlantedCoalitionGridworld(perm.config_for("test", i))
        ef.stop([0])
        ep.stop([int(perm.role_perm("test", i)[0])])
        S = th.tensor(np.stack([ef.get_state(), ep.get_state()]))
        for net in nets:
            with th.no_grad():
                v = net(S).squeeze(-1)
            assert th.allclose(v[0], v[1], atol=1e-5)


def test_random_start_cells_are_not_a_relabelling():
    """v4 Part D is a genuine generalisation test: random start cells change the state beyond a relabelling."""
    perm = DataCell(delay=25, n_bystanders=2, seed=5, horizon=60, shuffle_roles=True)
    rs = DataCell(delay=25, n_bystanders=2, seed=5, horizon=60, shuffle_roles=True, random_starts=True)
    s0 = np.sort(PlantedCoalitionGridworld(perm.config_for("test", 0)).get_state())
    differ = sum(not np.allclose(np.sort(PlantedCoalitionGridworld(rs.config_for("test", i)).get_state()), s0)
                 for i in range(10))
    assert differ == 10


def test_equal_negative_credit_scores_the_maximum_l1():
    """Errata E11: an equal split with the wrong sign scores L1 = 2 (the maximum), not the equal split's score."""
    truth = np.array([10.0, 10.0, 10.0, 0.0, 0.0])
    l1 = lambda c: np.abs(gt.normalise_shares(c) - gt.normalise_shares(truth)).sum()
    assert np.isclose(l1(-np.ones(5)), 2.0)
    assert np.isclose(l1(np.ones(5)), 0.8)
