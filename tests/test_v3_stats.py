import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "v3"))
import stats as v3stats  # noqa: E402


def test_holm_matches_definition():
    assert np.allclose(v3stats.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


def test_exact_wilcoxon_minimum_with_five_positive_differences():
    assert v3stats.wilcoxon_exact(np.array([1, 2, 3, 4, 5.0])) == 1 / 32


def test_threshold_and_paired_families():
    below = {"c1": [0.01, 0.02, 0.015, 0.012, 0.018], "c2": [0.03, 0.02, 0.025, 0.028, 0.03]}
    assert v3stats.threshold_family(below, 0.15)["verdict"] == "SUPPORTED"
    assert v3stats.threshold_family(below, 0.02)["verdict"] != "SUPPORTED"
    a = {"c": [0.1, 0.12, 0.11, 0.09, 0.1]}
    b = {"c": [0.8, 0.82, 0.79, 0.81, 0.8]}
    assert v3stats.paired_family(a, b, delta=0.3)["verdict"] == "SUPPORTED"
    assert v3stats.paired_family(a, b, delta=0.75)["verdict"] == "NOT SUPPORTED"
    const = {"c": [0.0] * 5}
    assert v3stats.threshold_family(const, 0.05)["verdict"] == "SUPPORTED"
