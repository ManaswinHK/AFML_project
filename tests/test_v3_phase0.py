"""v3 Phase 0: Markov state for partial payouts, persistent stopping, finisher feasibility.

Each test is the "done when" criterion of a Phase 0 item in docs/assessment_and_plan.md, Section 8.
"""

import numpy as np
import pytest

from whoearned import scripted as scripted_mod
from whoearned.envs import NOOP, PlantedCoalitionGridworld, coalition_layout, overlap_example
from whoearned.replay import ground_truth_closed_loop, rollout
from whoearned.scripted import ScriptedTeam

N = 5


def _state_index(env, k, field):
    """Index of a per-event field in the global state (see gridworld.get_state)."""
    n = env.n_agents
    offsets = {"gate": 0, "done": 1, "pending": 2, "countdown": 3, "amount": 4}
    return 4 * n + k * (5 + n) + offsets[field]


def _run(cfg, team, env=None, stop_at=None, stop=(), null="noop"):
    env = env or PlantedCoalitionGridworld(cfg)
    env.reset()
    team.reset(env)
    states, rewards = [], []
    for t in range(cfg.horizon):
        if stop_at is not None and t == stop_at:
            env.stop(stop, null)
        states.append(env.get_state())
        _, r, _, _ = env.step(team.act(env))
        rewards.append(r)
    return np.array(states), np.array(rewards), env


# --- 0.1: pending payout amount ------------------------------------------------------------------


def test_partial_payouts_of_5_and_10_have_different_pending_states():
    """Two partial-layout episodes, one where only B finishes (pays 5) and one where B and C both
    finish (pays 10): while the payout is pending, the states must differ in the amount field."""
    cfg = coalition_layout(2, delay=25, horizon=60, partial=True)
    team = ScriptedTeam(cfg, seed=0)
    full_states, full_r, _ = _run(cfg, team)
    one_states, one_r, _ = _run(cfg, ScriptedTeam(cfg, seed=0), stop_at=0, stop=(2,), null="remove")
    assert full_r.sum() == pytest.approx(10.0) and one_r.sum() == pytest.approx(5.0)
    env = PlantedCoalitionGridworld(cfg)
    pend, amount = _state_index(env, 0, "pending"), _state_index(env, 0, "amount")
    full_pending = full_states[full_states[:, pend] == 1.0]
    one_pending = one_states[one_states[:, pend] == 1.0]
    assert len(full_pending) and len(one_pending)
    assert np.allclose(full_pending[:, amount], 1.0)
    assert np.allclose(one_pending[:, amount], 0.5)


def test_amount_is_zero_when_nothing_is_pending():
    cfg = coalition_layout(2, delay=10)
    states, _, env = _run(cfg, ScriptedTeam(cfg, seed=3))
    pend, amount = _state_index(env, 0, "pending"), _state_index(env, 0, "amount")
    assert np.all(states[states[:, pend] == 0.0][:, amount] == 0.0)


# --- 0.2: persistent stopping --------------------------------------------------------------------


@pytest.mark.parametrize("mask", [0b11110, 0b11101, 0b11011, 0b00111, 0b01011])
@pytest.mark.parametrize("seed", [0, 1])
def test_removal_at_t0_equals_reset_with_present_mask(mask, seed):
    cfg = coalition_layout(2, delay=15)
    ref = rollout(cfg, ScriptedTeam(cfg, seed=seed, epsilon=0.2), present=mask)
    removed = [i for i in range(N) if not mask >> i & 1]
    states, rewards, env = _run(cfg, ScriptedTeam(cfg, seed=seed, epsilon=0.2), stop_at=0, stop=removed, null="remove")
    np.testing.assert_array_equal(rewards, ref.rewards)
    assert env.present_mask == mask


def test_noop_stop_keeps_agent_on_grid_inert_and_flags_it_in_state_only():
    cfg = coalition_layout(2, delay=0)
    env = PlantedCoalitionGridworld(cfg)
    env.reset()
    team = ScriptedTeam(cfg, seed=0)
    team.reset(env)
    for _ in range(3):
        env.step(team.act(env))
    obs_before = [o.copy() for o in env.get_obs()]
    pos = env.positions[1]
    env.stop([1], "noop")
    for o_b, o_a in zip(obs_before, env.get_obs()):
        np.testing.assert_array_equal(o_b, o_a)          # teammates are not told
    assert env.get_state()[3 * N + 1] == 0.0 and env.get_state()[2 * N + 1] == 1.0
    assert env.get_avail_actions()[1] == [1, 0, 0, 0, 0, 0]
    for _ in range(10):
        env.step(team.act(env))
        assert env.positions[1] == pos


def test_noop_stop_of_a_finisher_loses_the_event_and_stop_of_bystander_does_not():
    cfg = coalition_layout(2, delay=0)
    _, r_b, _ = _run(cfg, ScriptedTeam(cfg, seed=0), stop_at=0, stop=(1,), null="noop")
    _, r_d, _ = _run(cfg, ScriptedTeam(cfg, seed=0), stop_at=0, stop=(3,), null="noop")
    assert r_b.sum() == 0.0 and r_d.sum() == pytest.approx(10.0)


def test_clone_is_independent_and_continues_identically():
    cfg = coalition_layout(2, delay=5)
    env = PlantedCoalitionGridworld(cfg)
    env.reset()
    team = ScriptedTeam(cfg, seed=4, epsilon=0.2)
    team.reset(env)
    for _ in range(6):
        env.step(team.act(env))
    snap_env, snap_rng = env.clone(), team.get_rng_state()
    rest = [env.step(team.act(env))[1] for _ in range(cfg.horizon - 6)]
    team2 = ScriptedTeam(cfg, seed=4, epsilon=0.2)
    team2.reset(snap_env)
    team2.set_rng_state(snap_rng)
    rest2 = [snap_env.step(team2.act(snap_env))[1] for _ in range(cfg.horizon - 6)]
    assert rest == rest2 and sum(rest) == pytest.approx(10.0)


# --- 0.3: finishers proceed when an absent enabler has already pulled -----------------------------


def _pull_time(cfg, seed):
    _, _, env = _run(cfg, ScriptedTeam(cfg, seed=seed))
    return env.event_log[0].pull_times[0]


def test_removing_the_enabler_after_its_pull_does_not_lose_the_event():
    cfg = coalition_layout(2, delay=0)
    tp = _pull_time(cfg, 0)
    _, before, _ = _run(cfg, ScriptedTeam(cfg, seed=0), stop_at=tp, stop=(0,), null="remove")
    _, after, _ = _run(cfg, ScriptedTeam(cfg, seed=0), stop_at=tp + 1, stop=(0,), null="remove")
    assert before.sum() == 0.0            # removed before pulling: the gate never opens
    assert after.sum() == pytest.approx(10.0)


def _old_finish_feasible(self, env, k):
    ev = self.config.events[k]
    if not all(env.present[e] for e in ev.enablers):
        return False
    return ev.partial or all(env.present[f] for f in ev.finishers)


@pytest.mark.parametrize("layout", ["coalition", "overlap", "partial"])
def test_closed_loop_coalition_values_are_unchanged_by_the_feasibility_fix(layout, monkeypatch):
    """Removal at t = 0 happens before any pull, so v(S) must be identical under the old rule."""
    cfg = overlap_example(delay=15) if layout == "overlap" else coalition_layout(2, delay=15, partial=layout == "partial")
    for seed in range(3):
        new = ground_truth_closed_loop(cfg, lambda: ScriptedTeam(cfg, seed=seed, epsilon=0.2)).v
        with monkeypatch.context() as m:
            m.setattr(scripted_mod.ScriptedTeam, "_finish_feasible", _old_finish_feasible)
            old = ground_truth_closed_loop(cfg, lambda: ScriptedTeam(cfg, seed=seed, epsilon=0.2)).v
        np.testing.assert_array_equal(new, old)


def test_noop_null_actions_are_overridden_even_if_the_team_sends_moves():
    cfg = coalition_layout(2, delay=0)
    env = PlantedCoalitionGridworld(cfg)
    env.reset()
    env.stop([0], "noop")
    pos = env.positions[0]
    env.step([3, NOOP, NOOP, NOOP, NOOP])   # 3 = LEFT is blocked anyway; try RIGHT (4) next
    env.step([4, NOOP, NOOP, NOOP, NOOP])
    assert env.positions[0] == pos


# --- 2.1: shuffled roles -------------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_permuted_episode_has_the_same_return_and_the_permuted_truth(seed):
    """Relabelling agents changes nothing physical: same return, and v(S) of the permuted layout equals
    v(perm^-1 S) of the fixed layout (closed loop, deterministic team)."""
    from whoearned.ground_truth import mask_of
    rng = np.random.default_rng(seed)
    perm = rng.permutation(5)                      # perm[role] = agent
    base = coalition_layout(2, delay=10)
    cfg = coalition_layout(2, delay=10, role_perm=perm)
    assert cfg.events[0].enablers == (perm[0],)
    v_base = ground_truth_closed_loop(base, lambda: ScriptedTeam(base, seed=0)).v
    v_perm = ground_truth_closed_loop(cfg, lambda: ScriptedTeam(cfg, seed=0)).v
    for m in range(32):
        roles = [r for r in range(5) if m >> r & 1]
        assert v_perm[mask_of([perm[r] for r in roles])] == pytest.approx(v_base[m])
    assert v_perm[-1] == pytest.approx(10.0)


def test_shuffled_data_cell_assigns_roles_per_episode():
    from whoearned.experiments import DataCell
    cell = DataCell(delay=0, n_bystanders=2, seed=5, shuffle_roles=True, n_train=6, n_test=3)
    perms = {tuple(cell.role_perm("train", i)) for i in range(6)}
    assert len(perms) > 1 and cell.key.startswith("perm_")
    eps = cell.episodes("train")
    assert all(ep.total == pytest.approx(10.0) or ep.total == 0 for ep in eps)
    assert [ep.config for ep in eps] == [cell.config_for("train", i) for i in range(6)]
