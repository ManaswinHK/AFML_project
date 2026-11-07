"""The closed-loop no-op null used to re-test H0 (scripts/sensitivity_loop_h0_h1b.py)."""

import os
import sys

import numpy as np

from whoearned.envs import running_example
from whoearned.replay import ground_truth_closed_loop, rollout
from whoearned.scripted import ScriptedTeam

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts"))
from sensitivity_loop_h0_h1b import ground_truth_closed_loop_noop, rollout_inert  # noqa: E402

E = 4
ALL5 = 0b11111


def test_inert_rollout_with_everyone_present_is_the_ordinary_rollout():
    cfg = running_example()
    a = rollout(cfg, ScriptedTeam(cfg, seed=3, epsilon=0.2))
    b = rollout_inert(cfg, ScriptedTeam(cfg, seed=3, epsilon=0.2), ALL5)
    np.testing.assert_array_equal(a.actions, b.actions)
    assert a.total == b.total


def test_inert_agent_stays_on_the_grid_and_never_acts():
    cfg = running_example()
    ep = rollout_inert(cfg, ScriptedTeam(cfg, seed=0), ALL5 & ~(1 << E))
    assert np.all(ep.actions[:, E] == 0)
    assert ep.total == 10.0  # E is a bystander: the coalition still completes the event


def test_closed_loop_nulls_agree_without_obstruction():
    cfg = running_example()
    make = lambda: ScriptedTeam(cfg, seed=0, bystander_mode="idle")  # noqa: E731
    np.testing.assert_allclose(ground_truth_closed_loop_noop(cfg, make).shapley,
                               ground_truth_closed_loop(cfg, make).shapley)

