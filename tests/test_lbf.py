"""v4 Part B: the deterministic LBF matches lbforaging 2.0.0 exactly, and supports exact persistent-ablation truth."""

import numpy as np
import pytest

from whoearned import ablation as ab
from whoearned.envs.lbf import LBFConfig, LBFEnv, ScriptedLBFTeam


def _package_env_like(mine: LBFEnv):
    from lbforaging.foraging.environment import ForagingEnv
    cfg = mine.config
    pk = ForagingEnv(players=cfg.n_agents, min_player_level=1, max_player_level=2, min_food_level=1,
                     max_food_level=None, field_size=(cfg.size, cfg.size), max_num_food=cfg.n_food, sight=cfg.size,
                     max_episode_steps=cfg.horizon, force_coop=False)
    pk.reset(seed=0)
    pk.field = mine.field.copy()
    pk._food_spawned = float(mine.field.sum())
    for p, pos, lvl in zip(pk.players, mine.positions, mine.levels):
        p.position, p.level = tuple(pos), int(lvl)
    pk.current_step = 0
    pk._gen_valid_moves()
    return pk


@pytest.mark.parametrize("seed", range(12))
def test_matches_lbforaging_package_step_for_step(seed):
    cfg = LBFConfig(spawn_seed=(seed, 1))
    mine = LBFEnv(cfg)
    pk = _package_env_like(mine)
    rng = np.random.default_rng(seed)
    team = ScriptedLBFTeam(cfg, seed=seed, epsilon=0.5)
    team.reset(mine)
    for _ in range(cfg.horizon):
        acts = team.act(mine) if rng.random() < 0.7 else list(rng.integers(6, size=cfg.n_agents))
        _, r_pk, done_pk, _, _ = pk.step(list(acts))
        _, r_me, done_me, _ = mine.step(list(acts))
        assert r_me == pytest.approx(10.0 * sum(r_pk))
        assert [tuple(p.position) for p in pk.players] == mine.positions
        np.testing.assert_array_equal(pk.field, mine.field)
        if done_pk:
            assert done_me
            break


def test_scripted_team_collects_most_food():
    tot = []
    for s in range(20):
        cfg = LBFConfig(spawn_seed=(s, 2))
        f = ab.factual_rollout(cfg, lambda: ScriptedLBFTeam(cfg, seed=s, epsilon=0.0), make_env=LBFEnv)
        tot.append(f.total)
    assert np.mean(tot) > 7.0


def test_identity1_on_lbf():
    from whoearned.replay import GroundTruth  # noqa: F401
    cfg = LBFConfig(spawn_seed=(3, 3))
    make = lambda: ScriptedLBFTeam(cfg, seed=3, epsilon=0.2)
    f = ab.factual_rollout(cfg, make, make_env=LBFEnv)
    W = ab.oracle_ablation_values(cfg, make, f, null="noop")
    kappa = ab.ablation_td_credit(f.rewards, W)
    # stop at t = 0 under the no-op null equals a full no-op continuation from the start state
    for i in range(cfg.n_agents):
        g = ab.continue_stopped(cfg, make, f.snaps[0], (i,), "noop")
        assert kappa[i].sum() == pytest.approx(f.total - g)
    Wall = ab.all_masks_values(cfg, make, f, null="noop")
    out = ab.step_semivalue_credit(Wall)
    np.testing.assert_allclose(out["shapley"].sum(0), out["committed"], atol=1e-9)


def test_stopped_state_matches_lbf_env():
    cfg = LBFConfig(spawn_seed=(5, 5))
    f = ab.factual_rollout(cfg, lambda: ScriptedLBFTeam(cfg, seed=5), make_env=LBFEnv)
    for null in ("noop", "remove"):
        env = f.snaps[7][0].clone()
        env.stop((1,), null)
        np.testing.assert_array_equal(ab.stopped_state(f.states[7], cfg.n_agents, (1,), null), env.get_state())


def test_two_set_critic_is_invariant_to_agent_and_food_relabelling():
    import torch as th
    from whoearned.critics import TwoSetCritic
    cfg = LBFConfig(n_agents=4, n_food=3, spawn_seed=(9, 9))
    env = LBFEnv(cfg)
    S = env.get_state()[None]
    n, F = 4, 3
    pa, pf = [2, 0, 3, 1], [1, 2, 0]
    P = S.copy()
    for new, old in enumerate(pa):
        P[:, 2 * new:2 * new + 2] = S[:, 2 * old:2 * old + 2]
        for blk in (2, 3, 4):
            P[:, blk * n + new] = S[:, blk * n + old]
    for new, old in enumerate(pf):
        P[:, 5 * n + 4 * new:5 * n + 4 * new + 4] = S[:, 5 * n + 4 * old:5 * n + 4 * old + 4]
    th.manual_seed(0)
    net = TwoSetCritic(n, F)
    with th.no_grad():
        np.testing.assert_allclose(net(th.tensor(S)).numpy(), net(th.tensor(P)).numpy(), atol=1e-5)
