"""Interactive credit visualisation: data export and page build."""

import json
import os
import sys

import numpy as np
import pytest

from whoearned.experiments import DataCell, TrainSpec, run_data_cell
from whoearned.viz_export import _r, export_cell, layout_dict, trajectory

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_trajectory_reproduces_episode():
    cell = DataCell(delay=25, n_bystanders=2, seed=0)
    cfg = cell.config()
    ep = cell.episodes("test")[0]
    tr = trajectory(cfg, ep.actions)
    assert np.allclose(tr["rewards"], ep.rewards)
    assert len(tr["positions"]) == cfg.horizon + 1
    assert tr["positions"][0] == [list(c) for c in cfg.start_cells]
    rec, got = ep.event_log[0], tr["events"][0]
    assert got["completion_time"] == rec.completion_time
    assert got["payout_time"] == rec.completion_time + 25
    # the lever shows as pulled from the step after the pull onwards
    pull = rec.pull_times[0]
    assert tr["pulled"][pull][0] == [] and tr["pulled"][pull + 1][0] == [0]


def test_layout_roles():
    lay = layout_dict(DataCell(delay=0, n_bystanders=2, seed=0).config())
    assert lay["roles"] == ["enabler", "finisher", "finisher", "bystander", "bystander"]
    assert lay["events"][0]["levers"] == {"0": [0, 4]}


def test_rounding():
    assert _r([1.23456789, np.float32(0.000123456), 3, None]) == [1.235, 0.0001235, 3, None]


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    cell = DataCell(delay=0, n_bystanders=2, seed=0, n_train=8, n_test=2)
    train = TrainSpec(updates=3, batch_size=4, log_every=1, algs=("vdn", "coma"))
    path = run_data_cell(cell, train, str(tmp_path_factory.mktemp("cell")))
    with open(path) as f:
        return cell, train, json.load(f)


def test_export_reproduces_stored_cell(tiny):
    cell, train, stored = tiny
    out = export_cell(cell, stored, train)
    assert len(out["episodes"]) == 2
    e = out["episodes"][0]
    assert np.asarray(e["methods"]["vdn"]["per_step"]).shape == (cell.horizon, 5)
    assert np.asarray(e["truth"]["step_noop"]).shape == (cell.horizon, 5)
    assert "dividends" in e["methods"]["vdn"] and "dividends" not in e["methods"]["coma"]
    assert e["methods"]["coma"]["step_target"] == "baseline"
    # episode shares are the per-step credit summed over time
    shares = np.asarray(e["methods"]["vdn"]["per_step"]).sum(0)
    assert np.allclose(shares / np.abs(shares).sum(), e["methods"]["vdn"]["shares"], atol=1e-3)


def test_export_rejects_mismatch(tiny):
    cell, train, stored = tiny
    bad = json.loads(json.dumps(stored))
    bad["algs"]["vdn"]["episodes"][1]["shares"][0] += 0.1
    with pytest.raises(RuntimeError, match="vdn episode 1"):
        export_cell(cell, bad, train)


def test_tolerant_mode_reports_drift_but_checks_losses(tiny):
    cell, train, stored = tiny
    bad = json.loads(json.dumps(stored))
    bad["algs"]["vdn"]["episodes"][1]["shares"][0] += 0.1
    out = export_cell(cell, bad, train, l1_tol=0.5)
    rep = out["reproduction"]["algs"]["vdn"]
    assert out["reproduction"]["mode"] == "tolerant"
    assert rep["episode_share_l1_max"] == pytest.approx(0.1, abs=1e-3)
    bad["algs"]["vdn"]["loss_trace"][0][1] += 1.0
    with pytest.raises(RuntimeError, match="early losses"):
        export_cell(cell, bad, train, l1_tol=0.5)


def test_build_injects_data(tmp_path, monkeypatch):
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import export_credit_viz as ecv

    monkeypatch.setattr(ecv, "CACHE", str(tmp_path))
    monkeypatch.setattr(ecv, "OUT", str(tmp_path / "out.html"))
    (tmp_path / "k.json").write_text(json.dumps({"key": "k", "note": "</script>"}))
    page = open(ecv.build(["k"]), encoding="utf-8").read()
    assert ecv.PLACEHOLDER not in page
    assert '{"cells":[{"key":"k","note":"<\\/script>"}]}' in page
