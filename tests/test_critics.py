"""The set critic is exactly invariant to relabelling agents; both critics fit a tiny problem."""

import numpy as np
import torch as th

from whoearned.critics import SetCritic, StateTokens, fit_mlp, fit_set_critic
from whoearned.envs import PlantedCoalitionGridworld, overlap_example


def permute_state(S, n, k, perm):
    """State of the same situation with agent perm[i] renamed to i (positions, flags, lever flags)."""
    S = np.array(S, copy=True)
    out = S.copy()
    per_event = 5 + n
    for new, old in enumerate(perm):
        out[..., 2 * new:2 * new + 2] = S[..., 2 * old:2 * old + 2]
        out[..., 2 * n + new] = S[..., 2 * n + old]
        out[..., 3 * n + new] = S[..., 3 * n + old]
        for e in range(k):
            out[..., 4 * n + e * per_event + 5 + new] = S[..., 4 * n + e * per_event + 5 + old]
    return out


def test_set_critic_is_permutation_invariant():
    cfg = overlap_example()
    env = PlantedCoalitionGridworld(cfg)
    env.reset()
    rng = np.random.default_rng(0)
    states = []
    for _ in range(20):
        env.step(rng.integers(6, size=5))
        states.append(env.get_state())
    env.stop([2], "noop")
    states.append(env.get_state())
    S = np.array(states)
    th.manual_seed(0)
    net = SetCritic(StateTokens(5, 2))
    perm = rng.permutation(5)
    with th.no_grad():
        a = net(th.tensor(S)).numpy()
        b = net(th.tensor(permute_state(S, 5, 2, perm), dtype=th.float32)).numpy()
    np.testing.assert_allclose(a, b, atol=1e-5)
    assert StateTokens(5, 2).d_state == env.state_size


def test_critics_fit_a_simple_function():
    rng = np.random.default_rng(1)
    n, k = 3, 1
    d = 4 * n + k * (5 + n) + 1
    X = rng.random((2000, d)).astype(np.float32)
    y = 10 * X[:, 3 * n:4 * n].sum(1)          # value = 10 x number of acting agents (symmetric)
    for fit in (lambda: fit_mlp(X, y, steps=1500), lambda: fit_set_critic(X, y, n, k, steps=1500)):
        p = fit()
        assert np.mean((p(X) - y) ** 2) < 1.0
