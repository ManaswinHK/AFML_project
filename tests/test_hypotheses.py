import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts"))
from analyse_hypotheses import holm, paired  # noqa: E402


def test_holm_matches_hand_computation():
    # Sorted p: 0.01, 0.02, 0.04 -> 3*0.01, max(.03, 2*0.02), max(.04, 1*0.04)
    np.testing.assert_allclose(holm([0.04, 0.01, 0.02]), [0.04, 0.03, 0.04])
    assert holm([0.5, 0.9])[1] == 1.0 or holm([0.5, 0.9])[1] == pytest.approx(0.9)


def test_paired_detects_a_clear_five_seed_effect():
    x = {s: 1.0 + 0.1 * s for s in range(5)}
    y = {s: 0.5 + 0.1 * s + 0.01 * (s % 2) for s in range(5)}
    res = paired(x, y)
    assert res["n"] == 5 and res["diff"] == pytest.approx(0.496)  # diffs 0.5, 0.49, 0.5, 0.49, 0.5
    assert res["t_p"] < 0.001
    assert res["w_p"] == pytest.approx(1 / 32)  # the exact Wilcoxon floor at n = 5


def test_paired_needs_three_seeds_and_ignores_missing():
    assert paired({0: 1.0, 1: 2.0}, {0: 0.0, 1: 1.0}) is None
    res = paired({0: 1.0, 1: 2.0, 2: 3.0, 3: np.nan}, {0: 0.0, 1: 1.0, 2: 2.0, 3: 0.0})
    assert res["n"] == 3


def test_paired_handles_constant_difference():
    res = paired({s: 2.0 for s in range(5)}, {s: 1.0 for s in range(5)})
    assert res["t_p"] == 0.0 and res["diff"] == 1.0
