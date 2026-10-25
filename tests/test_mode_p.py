import json
import os

import numpy as np
import pytest

from whoearned.envs import NOOP, coalition_layout, running_example
from whoearned.replay import rollout
from whoearned.scripted import ScriptedTeam

EPYMARL = os.path.join(os.path.dirname(os.path.dirname(__file__)), "external", "epymarl", "src")
needs_epymarl = pytest.mark.skipif(not os.path.isdir(EPYMARL), reason="EPyMARL not cloned into external/")


# --- parametric layout -----------------------------------------------------------------


def test_coalition_layout_with_two_bystanders_is_the_running_example():
    assert coalition_layout(2) == running_example()


@pytest.mark.parametrize("b", [0, 2, 4, 5])
def test_coalition_layout_bystander_count(b):
    cfg = coalition_layout(b)
    assert cfg.n_agents == 3 + b
    assert cfg.bystanders == tuple(range(3, 3 + b))


def test_coalition_layout_rejects_too_many_bystanders():
    with pytest.raises(ValueError):
        coalition_layout(6)


# --- behaviour probabilities -----------------------------------------------------------


@pytest.mark.parametrize("mode", ["wander", "roam", "idle"])
def test_behaviour_probs_are_distributions_that_cover_taken_actions(mode):
    cfg = running_example()
    ep = rollout(cfg, ScriptedTeam(cfg, seed=3, epsilon=0.2, bystander_mode=mode))
    p = ep.behaviour_probs
    assert p.shape == (cfg.horizon, 5, 6)
    np.testing.assert_allclose(p.sum(-1), 1.0)
    taken = np.take_along_axis(p, ep.actions[..., None], axis=2)[..., 0]
    assert (taken > 0).all()


def test_coalition_member_probability_of_intended_action():
    cfg = running_example()
    team = ScriptedTeam(cfg, seed=0, epsilon=0.3, bystander_mode="idle")
    ep = rollout(cfg, team)
    # At t=0 agent A intends RIGHT towards its lever.
    p0 = ep.behaviour_probs[0, 0]
    assert p0.max() == pytest.approx(0.7 + 0.3 / 6)
    assert p0.min() == pytest.approx(0.3 / 6)
    # Idle bystanders always NOOP.
    np.testing.assert_allclose(ep.behaviour_probs[:, 3, NOOP], 1.0)


def test_wandering_bystander_probability_folds_reserved_moves_into_noop():
    # D starts at (4,4); moving UP to (3,4) is C's lane (reserved), so it maps to NOOP.
    cfg = running_example()
    ep = rollout(cfg, ScriptedTeam(cfg, seed=0))
    p = ep.behaviour_probs[0, 3]
    assert p[NOOP] == pytest.approx(0.4)   # u=NOOP plus the folded UP
    assert p[1] == 0.0                     # UP itself is impossible


# --- critic-only training and read-outs --------------------------------------------------


@needs_epymarl
@pytest.mark.parametrize("alg", ["vdn", "qmix", "coma", "mappo"])
def test_train_step_updates_critic_and_read_credit_works(alg):
    from whoearned.epymarl_bridge import build_harness, episodes_to_batch
    from whoearned.mode_p import read_credit, train_step

    cfg = running_example()
    eps = [rollout(cfg, ScriptedTeam(cfg, seed=s, epsilon=0.2)) for s in range(6)]
    h = build_harness(alg, seed=0)
    batch = episodes_to_batch(h, eps)
    critic = h.learner.mixer if alg in ("vdn", "qmix") else h.learner.critic
    params = h.learner.mac.parameters() if alg == "vdn" else critic.parameters()  # VDN mixer has no params
    before = [p.detach().clone() for p in params]
    actor_before = [p.detach().clone() for p in h.learner.mac.parameters()]

    losses = [train_step(h, alg, batch, k) for k in range(3)]
    assert all(np.isfinite(losses))

    after = list(h.learner.mac.parameters() if alg == "vdn" else critic.parameters())
    assert any(not np.allclose(a.detach(), b) for a, b in zip(after, before))
    if alg in ("coma", "mappo"):  # actors are never trained in Mode P
        for a, b in zip(h.learner.mac.parameters(), actor_before):
            np.testing.assert_array_equal(a.detach(), b)

    out = read_credit(h, alg, batch, eps)
    assert out["per_step"].shape == (6, cfg.horizon, 5)
    assert np.isfinite(out["per_step"]).all()
    if alg == "mappo":
        assert out["snr"]["snr"].shape == (5,)


@needs_epymarl
def test_run_data_cell_writes_complete_record(tmp_path):
    from whoearned.experiments import DataCell, TrainSpec, run_data_cell

    cell = DataCell(delay=0, n_bystanders=2, seed=0, n_train=8, n_test=3)
    path = run_data_cell(cell, TrainSpec(updates=2, batch_size=4, log_every=1), str(tmp_path))
    with open(path) as f:
        rec = json.load(f)
    assert set(rec["algs"]) == {"vdn", "qmix", "coma", "mappo"}
    assert rec["data"]["test_matches_plan"] >= 0
    for a in rec["algs"].values():
        assert a["summary"]["n_total"] == 3
        assert len(a["episodes"]) == 3
        assert "step_pearson" in a["summary_fidelity"]
    assert "snr" in rec["algs"]["mappo"]
    assert set(rec["semantic_gap"]) == {"noop", "baseline"}
    # With no slack-free enabler, a perfect noop critic never credits A: missed = 1.
    assert rec["semantic_gap"]["noop"]["mean_shares"][0] == pytest.approx(0.0)
