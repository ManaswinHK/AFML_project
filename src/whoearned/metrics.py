"""Credit-faithfulness metrics (problem statement v2, Section 2.5).

Every metric compares a method's credit read-out with a ground-truth target
*after* both are normalised to shares, c_i / sum_j |c_j|. Use the target that
matches the read-out: leave-one-out marginals for counterfactual read-outs
(difference of Q_tot, COMA), Shapley values for redistribution methods (STAS)
and for coalition-level comparisons.

Functions return ``None`` when a quantity is undefined (e.g. no credit to
share); callers count those cases rather than dropping them silently.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import comb
from typing import Iterable, Sequence

import numpy as np

from whoearned.envs.gridworld import GridConfig
from whoearned.ground_truth import agents_of, normalise_shares
from whoearned.replay import GroundTruth

Coalition = frozenset[int]


# --- individual-level ----------------------------------------------------------------


def credit_error(credit_shares: np.ndarray, target_shares: np.ndarray) -> float:
    """L1 distance between share vectors (0 = perfect, at most 2)."""
    return float(np.abs(np.asarray(credit_shares) - np.asarray(target_shares)).sum())


def _average_ranks(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x))
    sorted_x = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    """Spearman rank correlation with average ranks for ties; None if either side is constant."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    rx, ry = _average_ranks(x), _average_ranks(y)
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def leakage(credit_shares: np.ndarray, bystanders: Iterable[int]) -> float:
    """Lambda: total |share| given to bystanders (0 = perfect)."""
    shares = np.asarray(credit_shares)
    return float(sum(abs(shares[i]) for i in bystanders))


def missed_credit(
    credit_shares: np.ndarray, target_shares: np.ndarray, enablers: Iterable[int]
) -> float | None:
    """M = max(0, 1 - sum_E c / sum_E g): fraction of the enablers' deserved share that is missing.

    M > 1 means the enablers received *negative* credit (worse than forgotten);
    it is deliberately not capped. None if the enablers deserve nothing under the target.
    """
    enablers = list(enablers)
    deserved = float(sum(target_shares[i] for i in enablers))
    if deserved <= 0:
        return None
    received = float(sum(credit_shares[i] for i in enablers))
    return max(0.0, 1.0 - received / deserved)


# --- coalition-level -------------------------------------------------------------------


def jaccard(a: Iterable[int], b: Iterable[int]) -> float:
    a, b = set(a), set(b)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def chance_jaccard(n: int, true_size: int, guess_size: int) -> float:
    """Expected Jaccard between a fixed true group and a uniformly random group of ``guess_size``.

    The reference for "above chance" coalition recovery (H3).
    """
    total = comb(n, guess_size)
    expected = 0.0
    for i in range(max(0, true_size + guess_size - n), min(true_size, guess_size) + 1):
        ways = comb(true_size, i) * comb(n - true_size, guess_size - i)
        union = true_size + guess_size - i
        expected += ways / total * (i / union if union else 1.0)
    return expected


@dataclass(frozen=True)
class Recovery:
    recall: float     # mean over planted groups of the best Jaccard with any identified group
    precision: float  # mean over identified groups of the best Jaccard with any planted group


def coalition_recovery(
    identified: Sequence[Iterable[int]], planted: Sequence[Iterable[int]]
) -> Recovery:
    """Match identified groups to planted groups by best Jaccard, in both directions."""
    identified = [frozenset(g) for g in identified]
    planted = [frozenset(g) for g in planted]

    def best(groups, against):
        if not groups:
            return 1.0
        return float(np.mean([max((jaccard(g, h) for h in against), default=0.0) for g in groups]))

    return Recovery(recall=best(planted, identified), precision=best(identified, planted))


def identified_coalitions(
    dividends: dict[Coalition, float], threshold: float = 0.1
) -> list[Coalition]:
    """Coalitions whose estimated dividend share |d| / sum |d| exceeds ``threshold``."""
    total = sum(abs(d) for d in dividends.values())
    if total == 0:
        return []
    return [g for g, d in dividends.items() if abs(d) / total > threshold]


def true_dividends(truth: GroundTruth, event: int | None = None) -> dict[Coalition, float]:
    """Non-zero ground-truth dividends as {frozenset: value}, in total or for one event."""
    d = truth.dividends if event is None else truth.event_dividends[event]
    return {frozenset(agents_of(s)): float(d[s]) for s in range(1, len(d)) if abs(d[s]) > 1e-9}


def dividend_error(
    estimated: dict[Coalition, float],
    truth: dict[Coalition, float],
    max_order: int | None = None,
    normalise: bool = True,
) -> float | None:
    """sum_{|S| <= k} |est(S) - true(S)| over the union of coalitions in either dict.

    With ``normalise`` (default), each side is first divided by its own sum of
    |dividends|, so methods with arbitrarily scaled coalition terms are comparable;
    the error then lies in [0, 2]. Returns None if either side is all zero after
    the order filter.
    """
    def keep(d):
        return {frozenset(g): v for g, v in d.items() if max_order is None or len(g) <= max_order}

    est, tru = keep(estimated), keep(truth)
    if normalise:
        se, st = sum(abs(v) for v in est.values()), sum(abs(v) for v in tru.values())
        if se == 0 or st == 0:
            return None
        est = {g: v / se for g, v in est.items()}
        tru = {g: v / st for g, v in tru.items()}
    keys = est.keys() | tru.keys()
    return float(sum(abs(est.get(g, 0.0) - tru.get(g, 0.0)) for g in keys))


# --- across runs -------------------------------------------------------------------------


def return_faithfulness(
    returns: Sequence[float], credit_errors: Sequence[float], groups: Sequence[str] | None = None
) -> dict[str, float | None]:
    """Spearman(return, 1 - L1), pooled and within each group (e.g. method). H4 uses the within-group values."""
    returns = np.asarray(returns, dtype=np.float64)
    faith = 1.0 - np.asarray(credit_errors, dtype=np.float64)
    out: dict[str, float | None] = {"pooled": spearman(returns, faith)}
    if groups is not None:
        groups = np.asarray(groups)
        for g in dict.fromkeys(groups.tolist()):
            sel = groups == g
            out[g] = spearman(returns[sel], faith[sel]) if sel.sum() >= 3 else None
    return out


# --- one-call evaluation of an episode -------------------------------------------------


@dataclass
class CreditReport:
    valid: bool                  # False if credit or target had nothing to normalise
    l1: float | None = None
    spearman: float | None = None
    leakage: float | None = None
    missed: float | None = None

    def as_dict(self) -> dict:
        return asdict(self)


def evaluate_credit(
    credit: np.ndarray,
    truth: GroundTruth,
    config: GridConfig,
    target: str = "shapley",
    event: int | None = None,
    bystanders: Iterable[int] | None = None,
) -> CreditReport:
    """All individual-level metrics for one episode's credit read-out.

    ``target`` is "shapley" or "loo". With ``event=k``, the credit is compared with
    event k's own ground truth; "bystanders" then means agents outside event k's
    coalition and "enablers" means event k's enablers. Without ``event``, the
    planted bystanders and all events' enablers are used. Pass ``bystanders`` to
    override (e.g. with realised bystanders).
    """
    if target not in ("shapley", "loo"):
        raise ValueError("target must be 'shapley' or 'loo'")
    if event is None:
        g = truth.shapley if target == "shapley" else truth.leave_one_out
        enablers = sorted({e for ev in config.events for e in ev.enablers})
        default_bystanders = config.bystanders
    else:
        g = truth.event_shapley[event] if target == "shapley" else truth.event_leave_one_out[event]
        ev = config.events[event]
        enablers = list(ev.enablers)
        default_bystanders = [i for i in range(config.n_agents) if i not in ev.coalition]
    bystanders = default_bystanders if bystanders is None else list(bystanders)
    return evaluate_against(credit, g, enablers, bystanders)


def evaluate_against(
    credit: np.ndarray, target: np.ndarray, enablers: Iterable[int], bystanders: Iterable[int]
) -> CreditReport:
    """Individual-level metrics of ``credit`` against any target vector (both normalised to shares)."""
    c_shares, g_shares = normalise_shares(credit), normalise_shares(target)
    if c_shares is None or g_shares is None:
        return CreditReport(valid=False)
    return CreditReport(
        valid=True,
        l1=credit_error(c_shares, g_shares),
        spearman=spearman(c_shares, g_shares),
        leakage=leakage(c_shares, list(bystanders)),
        missed=missed_credit(c_shares, g_shares, list(enablers)),
    )


def top_k_selection(credit: Sequence[float], coalition: Iterable[int], tol: float = 1e-9) -> float:
    """Probability that keeping the |coalition| highest-credited agents keeps exactly the coalition.

    Ties at the cut-off are broken uniformly at random, so an equal split scores its chance level
    (1 / C(n, k)) instead of an artefact of index order (v3 Phase 0.5; plan Test D).
    """
    credit = np.asarray(credit, dtype=np.float64)
    coalition = set(int(i) for i in coalition)
    k = len(coalition)
    if k == 0 or k > len(credit):
        return 0.0
    cut = np.sort(credit)[::-1][k - 1]
    scale = max(1.0, float(np.abs(credit).max()))
    above = {i for i in range(len(credit)) if credit[i] > cut + tol * scale}
    tied = {i for i in range(len(credit)) if abs(credit[i] - cut) <= tol * scale}
    need = coalition - above
    if not above <= coalition or not need <= tied:
        return 0.0
    return 1.0 / comb(len(tied), len(need))


def pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    """Pearson correlation of flattened arrays; None if either is constant."""
    x, y = np.ravel(x).astype(np.float64), np.ravel(y).astype(np.float64)
    if x.std() == 0 or y.std() == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def realised_bystanders(truth: GroundTruth, tol: float = 1e-9) -> list[int]:
    """Agents that belong to no coalition with a non-zero dividend in this episode.

    Stricter than "Shapley value ~ 0", which can also happen when an agent's
    positive and negative dividends cancel.
    """
    n = len(truth.shapley)
    involved = 0
    for s, d in enumerate(truth.dividends):
        if abs(d) > tol:
            involved |= s
    return [i for i in range(n) if not involved >> i & 1]
