"""Step 7: NA²Q and STAS ports, and the mixer-game coalition read-out."""

import os

import numpy as np
import pytest
import torch as th

from whoearned.envs import coalition_layout
from whoearned.ground_truth import harsanyi_dividends, mask_of, shapley_values
from whoearned.replay import rollout
from whoearned.scripted import ScriptedTeam

EPYMARL = os.path.join(os.path.dirname(os.path.dirname(__file__)), "external", "epymarl", "src")
needs_epymarl = pytest.mark.skipif(not os.path.isdir(EPYMARL), reason="EPyMARL not cloned into external/")


# --- mixer game -------------------------------------------------------------------------


@needs_epymarl
def test_additive_mixer_game_has_only_singleton_dividends():
    from whoearned.na2q import coalition_game, game_dividends

    chosen, null = th.rand(2, 4, 3), th.rand(2, 4, 3)
    d = game_dividends(coalition_game(lambda q: q.sum(-1), chosen, null))
    for i in range(3):
        np.testing.assert_allclose(d[:, mask_of([i])], (chosen - null)[..., i].sum(1).numpy(), atol=1e-5)
    singles = {mask_of([i]) for i in range(3)}
    others = [s for s in range(1, 8) if s not in singles]
    np.testing.assert_allclose(d[:, others], 0.0, atol=1e-5)


@needs_epymarl
def test_pairwise_product_mixer_game_has_exactly_that_pair_dividend():
    from whoearned.na2q import coalition_game, game_dividends

    chosen, null = th.ones(1, 1, 3) * 2.0, th.zeros(1, 1, 3)
    d = game_dividends(coalition_game(lambda q: q[..., 0] * q[..., 1], chosen, null))[0]
    expected = np.zeros(8)
    expected[mask_of([0, 1])] = 4.0
    np.testing.assert_allclose(d, expected, atol=1e-6)


@needs_epymarl
def test_per_step_game_shapley_matches_exact_shapley():
    from whoearned.na2q import game_shapley_per_step

    rng = np.random.default_rng(0)
    game = rng.normal(size=(2, 3, 16))
    game[..., 0] = 0.0
    out = game_shapley_per_step(game)
    for b in range(2):
        for t in range(3):
            np.testing.assert_allclose(out[b, t], shapley_values(game[b, t]), atol=1e-10)


# --- NA²Q ---------------------------------------------------------------------------


def _harness_and_batch(alg, n_bystanders=0, episodes=6, horizon=30):
    from whoearned.epymarl_bridge import episodes_to_batch
    from whoearned.mode_p import make_harness

    cfg = coalition_layout(n_bystanders, delay=0, horizon=horizon)
    eps = [rollout(cfg, ScriptedTeam(cfg, seed=s, epsilon=0.2)) for s in range(episodes)]
    env_args = {"map_name": "coalition_layout",
                "layout_kwargs": {"n_bystanders": n_bystanders, "delay": 0, "horizon": horizon}}
    h = make_harness(alg, env_args=env_args, seed=0)
    return h, episodes_to_batch(h, eps), eps


@needs_epymarl
def test_na2q_mixer_weights_are_a_distribution_and_terms_are_monotone():
    h, batch, _ = _harness_and_batch("na2q")
    net = h.learner
    n, h_dim = net.n_agents, net.args.rnn_hidden_dim
    qs = th.randn(64, n)
    states = batch["state"][0, :1].expand(64, -1)
    latent = th.randn(64, n, h_dim)
    f, w, _, _ = net.mixer.parts(qs, states, latent)
    assert w.shape == (64, n + n * (n - 1) // 2)
    np.testing.assert_allclose(w.sum(-1).detach().numpy(), 1.0, atol=1e-5)
    assert (w >= 0).all()
    f_up, _, _, _ = net.mixer.parts(qs + 1.0, states, latent)
    assert (f_up >= f - 1e-6).all()


@needs_epymarl
def test_na2q_mixer_game_has_no_dividends_above_order_two():
    from whoearned.ground_truth import agents_of

    h, batch, _ = _harness_and_batch("na2q")
    for _ in range(3):
        h.learner.train(batch, episode_num=0)
    game = h.learner.mixer_game(batch)
    d = np.apply_along_axis(harsanyi_dividends, -1, game)
    high = [s for s in range(game.shape[-1]) if len(agents_of(s)) >= 3]
    assert np.abs(d[..., high]).max() < 1e-4
    assert np.abs(d[..., [s for s in range(1, game.shape[-1]) if len(agents_of(s)) <= 2]]).max() > 1e-4


@needs_epymarl
def test_na2q_sarsa_target_fits_without_diverging():
    from whoearned.mode_p import train_step

    h, batch, _ = _harness_and_batch("na2q")
    assert h.learner.args.target == "sarsa"
    losses = [train_step(h, "na2q", batch, k) for k in range(600)]
    assert np.mean(losses[-20:]) < 0.5 * np.mean(losses[:5])
    # Every episode returns 10, so Q_tot must stay near [0, 10]; the official max target
    # on this fixed data passes 100.
    net = h.learner
    with th.no_grad():
        q, hidden = net.agent(net._inputs(batch))
        _, latent, _ = net.vae.encode(hidden[:, :-1])
        chosen = th.gather(q[:, :-1], 3, batch["actions"][:, :-1]).squeeze(3)
        q_tot = net.mixer(chosen, batch["state"][:, :-1], latent)
    assert q_tot.max().item() < 15.0


# --- STAS ---------------------------------------------------------------------------


@needs_epymarl
def test_stas_is_causal_in_time():
    h, batch, _ = _harness_and_batch("stas")
    model = h.learner.model.eval()
    obs, actions, lengths, _ = h.learner._inputs(batch)
    th.manual_seed(1)
    base = model(obs, actions, lengths)
    obs2 = obs.clone()
    obs2[:, :, -1] += 5.0
    th.manual_seed(1)
    moved = model(obs2, actions, lengths)
    np.testing.assert_allclose(base[..., :-1].detach().cpu().numpy(), moved[..., :-1].detach().cpu().numpy(), atol=1e-5)
    assert not th.allclose(base[..., -1], moved[..., -1])


@needs_epymarl
def test_stas_learns_to_decompose_the_return():
    from whoearned.mode_p import read_credit, train_step

    h, batch, eps = _harness_and_batch("stas")
    losses = [train_step(h, "stas", batch, k) for k in range(300)]
    assert np.mean(losses[-20:]) < 0.2 * np.mean(losses[:5])
    # The loss constrains the train-mode (dropout on) sum, so that is what must match the return.
    net = h.learner
    obs, actions, lengths, mask = net._inputs(batch)
    net.model.train()
    with th.no_grad():
        sums = sum(net.model(obs, actions, lengths).sum(dim=(1, 2)) for _ in range(16)) / 16
    np.testing.assert_allclose(sums.cpu().numpy(), [e.total for e in eps], rtol=0.25)
    credit = read_credit(h, "stas", batch, eps)["per_step"]
    assert credit.shape == (len(eps), 30, 3) and np.isfinite(credit).all()


@needs_epymarl
@pytest.mark.parametrize("alg", ["vdn", "qmix", "na2q"])
def test_coalition_algs_return_dividends(alg):
    from whoearned.mode_p import read_credit

    h, batch, eps = _harness_and_batch(alg)
    out = read_credit(h, alg, batch, eps)
    assert out["per_step"].shape == (len(eps), 30, 3)
    assert out["dividends"].shape == (len(eps), 8)
    np.testing.assert_allclose(out["dividends"][:, 0], 0.0, atol=1e-6)
