"""Ground-truth credit from coalition values.

A coalition is an int bitmask over agents 0..n-1 (bit i set <=> agent i present).
Coalition values are a float array ``v`` of length 2**n, where ``v[mask]`` is the
expected team return when only the agents in ``mask`` act and the rest follow
the null policy (Section 2.3 of the problem statement).

From ``v`` we derive:

- Harsanyi dividends  Delta(S) = sum_{A subset S} (-1)^{|S|-|A|} v(A)
  -> which coalition created the value;
- Shapley values      phi_i = sum_{S containing i} Delta(S) / |S|
  -> each agent's fair share;
- leave-one-out       m_i = v(N) - v(N minus i)
  -> the target that counterfactual read-outs (COMA, difference-of-Q_tot) estimate.
"""

from __future__ import annotations

from itertools import combinations
from math import factorial
from typing import Callable, Iterable

import numpy as np

MAX_EXACT_AGENTS = 16  # 2**16 coalitions; the project uses n <= 8


def mask_of(agents: Iterable[int]) -> int:
    """Bitmask for a collection of agent indices."""
    mask = 0
    for i in agents:
        mask |= 1 << i
    return mask


def agents_of(mask: int) -> tuple[int, ...]:
    """Agent indices contained in a bitmask, in increasing order."""
    out = []
    i = 0
    while mask:
        if mask & 1:
            out.append(i)
        mask >>= 1
        i += 1
    return tuple(out)


def coalition_sizes(n: int) -> np.ndarray:
    """|S| for every mask S in 0..2**n - 1."""
    return np.array([bin(s).count("1") for s in range(1 << n)], dtype=np.int64)


def n_agents(v: np.ndarray) -> int:
    """Number of agents implied by a coalition-value array of length 2**n."""
    size = len(v)
    n = size.bit_length() - 1
    if size < 1 or (1 << n) != size:
        raise ValueError(f"coalition-value array must have length 2**n, got {size}")
    return n


def coalition_values(value_fn: Callable[[int], float], n: int) -> np.ndarray:
    """Evaluate ``value_fn(mask)`` on every coalition of ``n`` agents."""
    if not 0 <= n <= MAX_EXACT_AGENTS:
        raise ValueError(f"exact enumeration supports 0..{MAX_EXACT_AGENTS} agents, got {n}")
    return np.array([float(value_fn(s)) for s in range(1 << n)], dtype=np.float64)


def harsanyi_dividends(v: np.ndarray) -> np.ndarray:
    """Harsanyi dividends via the fast Moebius transform, O(n 2**n)."""
    n = n_agents(v)
    d = np.array(v, dtype=np.float64, copy=True)
    for i in range(n):
        bit = 1 << i
        for s in range(1 << n):
            if s & bit:
                d[s] -= d[s ^ bit]
    return d


def values_from_dividends(d: np.ndarray) -> np.ndarray:
    """Inverse of :func:`harsanyi_dividends`: v(S) = sum_{A subset S} Delta(A)."""
    n = n_agents(d)
    v = np.array(d, dtype=np.float64, copy=True)
    for i in range(n):
        bit = 1 << i
        for s in range(1 << n):
            if s & bit:
                v[s] += v[s ^ bit]
    return v


def shapley_values(v: np.ndarray) -> np.ndarray:
    """Shapley values: each dividend split equally among its coalition's members."""
    n = n_agents(v)
    d = harsanyi_dividends(v)
    sizes = coalition_sizes(n)
    phi = np.zeros(n)
    for s in range(1, 1 << n):
        if d[s] != 0.0:
            share = d[s] / sizes[s]
            for i in agents_of(s):
                phi[i] += share
    return phi


def shapley_values_bruteforce(v: np.ndarray) -> np.ndarray:
    """Shapley values from the marginal-contribution formula (reference for tests)."""
    n = n_agents(v)
    phi = np.zeros(n)
    for i in range(n):
        others = [j for j in range(n) if j != i]
        for k in range(n):
            weight = factorial(k) * factorial(n - k - 1) / factorial(n)
            for subset in combinations(others, k):
                s = mask_of(subset)
                phi[i] += weight * (v[s | (1 << i)] - v[s])
    return phi


def leave_one_out(v: np.ndarray) -> np.ndarray:
    """m_i = v(N) - v(N minus i)."""
    n = n_agents(v)
    full = (1 << n) - 1
    return np.array([v[full] - v[full ^ (1 << i)] for i in range(n)])


def normalise_shares(credit: np.ndarray, min_total: float = 1e-8) -> np.ndarray | None:
    """Credit as shares c_i / sum_j |c_j|.

    Returns ``None`` when sum_j |c_j| is below ``min_total``; such episodes are
    counted and reported separately rather than silently dropped (Section 2.4).
    """
    credit = np.asarray(credit, dtype=np.float64)
    total = np.abs(credit).sum()
    if total < min_total:
        return None
    return credit / total


def nonzero_dividends(v: np.ndarray, tol: float = 1e-9) -> dict[tuple[int, ...], float]:
    """Coalitions with a non-negligible dividend, as {agent tuple: dividend}."""
    d = harsanyi_dividends(v)
    return {agents_of(s): float(d[s]) for s in range(1, len(d)) if abs(d[s]) > tol}
