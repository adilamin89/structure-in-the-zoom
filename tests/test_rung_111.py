"""Tests for rung 1.11.0: threads return bit-identical values, the class-block ladders (fast) equal the row-by-row ones,
the row-by-row arithmetic is rung 1.10's bit for bit, and the paper's replay conventions keep it."""
import json
import subprocess
import sys
import warnings

import numpy as np
import pytest

import rung as r


def _toy(seed=0, tuned=1.0, n_per=(30, 41, 25, 37, 33, 29, 40, 35), d=60):
    rng = np.random.default_rng(seed)
    y = np.concatenate([np.full(m, c) for c, m in enumerate(n_per)])
    rng.shuffle(y)
    X = rng.standard_normal((len(n_per), d))[y] * tuned + rng.standard_normal((len(y), d)) * (1 + 0.3 * y[:, None] % 3)
    return X, y


def _numbers(d, pre=""):
    out = {}
    for k, v in d.items():
        k = str(k)
        if isinstance(v, dict):
            out.update(_numbers(v, pre + k + "."))
        elif isinstance(v, (list, tuple)):
            out.update({f"{pre}{k}[{i}]": float(x) for i, x in enumerate(v) if isinstance(x, (int, float, np.number))})
        elif isinstance(v, (int, float, np.number)) and not isinstance(v, bool):
            out[pre + k] = float(v)
    return out


def _identical(a, b):
    na, nb = _numbers(a), _numbers(b)
    assert na.keys() == nb.keys()
    bad = [k for k in na if not (na[k] == nb[k] or (np.isnan(na[k]) and np.isnan(nb[k])))]
    assert not bad, bad[:5]


def _designs():
    X, y = _toy()
    n = len(y)
    pairs = (y // 2) * 100 + np.random.default_rng(6).integers(0, 4, n)     # strata shared by class pairs: pair orders
    return [(X, y, dict(n_perm=12, k_orders=6, keep_null=True)),
            (X, y, dict(n_perm=10, k_orders=5, strata=pairs, shift_null=True, time_blocks=r.block_ids(n, 20))),
            (X, y, dict(n_perm=8, k_orders=4, n_floor_draws=10, floor_seed=42, floor_by_rung=True))]


@pytest.mark.parametrize("fast", [False, True])
def test_threads_are_bit_identical(fast):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for X, y, kw in _designs():
            _identical(r.zoom(X, y, fast=fast, workers=1, **kw), r.zoom(X, y, fast=fast, workers=4, **kw))
        X, y = _toy()
        _identical(r.decompose(X, y, n_null=6, n_shuffle=2, fast=fast, workers=1),
                   r.decompose(X, y, n_null=6, n_shuffle=2, fast=fast, workers=4))


def test_fast_equals_rows():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for X, y, kw in _designs():
            a, b = _numbers(r.zoom(X, y, fast=False, **kw)), _numbers(r.zoom(X, y, fast=True, **kw))
            for k in a:
                tol = 0.0 if "p_two" in k else 1e-10 * max(1.0, abs(a[k]))  # p counts exceedances: the same draws
                assert abs(a[k] - b[k]) <= tol or (np.isnan(a[k]) and np.isnan(b[k])), (k, a[k], b[k])
        X, y = _toy()
        a = r.decompose(X, y, n_null=6, fast=False)["delta_split_free"]
        b = r.decompose(X, y, n_null=6, fast=True)["delta_split_free"]
        assert all(abs(a[k] - b[k]) < 1e-10 for k in a)


def test_candidate_screen_threads_and_fast():
    rng = np.random.default_rng(9)
    X, a = rng.standard_normal((300, 40)), rng.integers(0, 6, 300)
    X[:, :5] += a[:, None] * 0.4
    L = {"a": a, "b": rng.integers(0, 4, 300), "c": rng.integers(0, 5, 300)}
    for fast in (False, True):
        _identical(r.candidates(X, L, n_perm=20, strata="others", fast=fast),
                   r.candidates(X, L, n_perm=20, strata="others", fast=fast, workers=4))
    rows, fast = r.candidates(X, L, n_perm=20, fast=False)["screen"], r.candidates(X, L, n_perm=20, fast=True)["screen"]
    assert all(abs(rows[k]["z"] - fast[k]["z"]) < 1e-10 and rows[k]["p_one"] == fast[k]["p_one"] for k in L)


def test_ill_conditioned_ladders_fall_back_to_rows():
    # class means 50 times the fluctuations: a one-class rung keeps ~1e-7 of its uncentred sum, so those ladders are
    # recomputed row by row and agree with rows to rounding of the slope alone
    X, y = _toy(3, tuned=50.0)
    a = r.zoom(X, y, n_perm=6, k_orders=4, fast=False)
    b = r.zoom(X, y, n_perm=6, k_orders=4, fast=True)
    assert abs(a["delta_orderavg"] - b["delta_orderavg"]) < 1e-14 and abs(a["null_mean"] - b["null_mean"]) < 1e-14


def test_auto_keeps_the_replay_conventions_row_by_row():
    X, y = _toy()
    auto = lambda **kw: r.zoom(X, y, n_perm=0, k_orders=0, **kw)["arithmetic"]
    assert auto() == "fast" and auto(floor_by_rung=True) == "rows" and auto(perm_seed=3700) == "rows"
    assert auto(fast=False) == "rows" and auto(floor_by_rung=True, fast=True) == "fast"
    assert r.decompose(X, y, n_null=4)["arithmetic"] == "fast"
    assert r.decompose(X, y, n_null=4, floor_by_rung=True)["arithmetic"] == "rows"


def test_rows_are_rung_1_10_bit_for_bit():
    # values rung 1.10.0 returned on this design (float.hex), every null family included
    X, y = _toy()
    res = r.zoom(X, y, fast=False, workers=3, **_designs()[1][2])
    pinned = {"delta": "-0x1.4fe69a0b541e0p-5", "delta_orderavg": "-0x1.4a47dc5df6290p-5", "null_mean": "0x1.baa18c67264edp-8",
              "strat_null_mean_orderavg": "0x1.8992208e1d7fbp-4", "shift_null_mean": "0x1.3855b1ead9363p-7",
              "tb_null_mean_orderavg": "0x1.83fae568c2fbep-9", "delta_pairavg": "-0x1.24d5c68d1ff2fp-4",
              "pairavg_null_mean": "0x1.9d44f2dc0a5a3p-8"}
    for k, v in pinned.items():
        assert res[k].hex() == v, (k, res[k].hex(), v)


def test_command_line_workers_and_arithmetic(tmp_path):
    X, y = _toy()
    np.save(tmp_path / "X.npy", X); np.save(tmp_path / "y.npy", y)
    run = lambda out, *a: subprocess.run([sys.executable, "-m", "rung", "data", "X.npy", "y.npy", "--n-perm", "10", "--k-orders",
                                          "4", "--floor-draws", "4", "--decompose", "--out", out, *a],
                                         cwd=tmp_path, capture_output=True, text=True, check=True)
    run("w1.json", "--workers", "1"); run("w3.json", "--workers", "3"); run("rows.json", "--arithmetic", "rows")
    w1, w3, rows = (json.load(open(tmp_path / f)) for f in ("w1.json", "w3.json", "rows.json"))
    _identical(w1, w3)
    assert w1["arithmetic"] == "fast" and rows["arithmetic"] == "rows" and rows["decompose"]["arithmetic"] == "rows"
    assert abs(w1["delta_orderavg"] - rows["delta_orderavg"]) < 1e-10 and w1["p_two"] == rows["p_two"]
    pf = subprocess.run([sys.executable, "-m", "rung", "data", "X.npy", "y.npy", "--n-perm", "0", "--k-orders", "0", "--paper-floor",
                         "--out", "pf.json"], cwd=tmp_path, capture_output=True, text=True, check=True)
    assert pf.returncode == 0 and json.load(open(tmp_path / "pf.json"))["arithmetic"] == "rows"
