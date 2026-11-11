"""Every v3 method runs end to end through run_data_cell on a tiny cell and writes the fields the analysis reads."""

import json

import numpy as np
import pytest

from whoearned.experiments import DataCell, TrainSpec, run_data_cell

ALGS = ("uniform", "floor_expected", "atd", "atd_struct", "atd_noint", "atd_remove", "atd_set", "atd_set_struct",
        "atd_noroles", "atd_set_noroles", "table3_mix", "mappo_int", "mappo_int_masked", "qmix_int", "qmix_int_masked")


@pytest.mark.parametrize("layout,shuffle", [("coalition", False), ("overlap", False), ("coalition", True)])
def test_v3_methods_run_and_report(tmp_path, layout, shuffle):
    cell = DataCell(delay=5, n_bystanders=1 if layout == "overlap" else 2, seed=1, n_train=12, n_test=3,
                    layout=layout, shuffle_roles=shuffle)
    algs = tuple(a for a in ALGS if not (shuffle and a == "floor_expected"))
    path = run_data_cell(cell, TrainSpec(updates=4, batch_size=4, algs=algs), str(tmp_path))
    r = json.load(open(path))
    assert set(r["algs"]) == set(algs)
    for alg, rec in r["algs"].items():
        assert "summary_loo_closed_noop" in rec and "top_k" in rec, alg
    for key in ("loo_closed", "shapley_closed", "v_closed_noop", "loo_ablation_noop", "roles"):
        assert len(r["truth"][key]) == 3
    # the exact ablation truth at t = 0 equals the closed-loop no-op game's leave-one-out credit
    for loo, v in zip(r["truth"]["loo_ablation_noop"], r["truth"]["v_closed_noop"]):
        v = np.array(v)
        full = len(v) - 1
        np.testing.assert_allclose(loo, [v[full] - v[full ^ (1 << i)] for i in range(len(loo))], atol=1e-9)
