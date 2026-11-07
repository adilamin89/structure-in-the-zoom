"""Tests for rung 1.12.0: the paper's two arms. The between arm reads the class means alone and must read zero when the
means are identical; the within arm reads the shift with the class means removed and must read zero when the classes
differ only in their means."""
import json
import subprocess
import sys

import numpy as np

import rung as r


def _population(means=1.5, fluct=2.0, n=480, d=48, K=8, seed=0, y=None):
    rng = np.random.default_rng(seed)
    if y is None:
        y = np.repeat(np.arange(K), n // K)
        rng.shuffle(y)
    X = rng.standard_normal((K, d))[y] * means + rng.standard_normal((n, d)) * (1 + fluct * (y[:, None] == np.arange(d)[None, :] // (d // K)))
    return X, y


def test_between_arm_reads_the_class_means_alone():
    # identical class means, fluctuations that differ: the excess is zero in expectation; one seed's z is one draw from a
    # distribution that classes with different fluctuations widen beyond SD 1 (labels are not exchangeable), so test the mean
    zs = [r.classmean_share(*_population(means=0.0, seed=s), n_perm=100)["z"] for s in range(10)]
    assert abs(np.mean(zs)) < 1.5
    X, y = _population(means=1.5, fluct=0.0)            # class means alone
    b = r.classmean_share(X, y, n_perm=100)
    assert b["z"] > 20 and b["p_one"] == 1 / 101
    share = (np.bincount(y) @ (np.stack([X[y == c].mean(0) for c in range(8)]) - X.mean(0)) ** 2).sum() / ((X - X.mean(0)) ** 2).sum()
    assert abs(b["share"] - share) < 1e-12


def test_within_arm_reads_zero_without_fluctuation_structure():
    X, y = _population(means=1.5, fluct=0.0)            # classes differ only in their means
    w = r.within_arm(X, y, n_perm=40, n_null=5)
    assert w["p_one_unrestricted_both_halves"] > 0.05
    assert all(abs(v) < 0.1 for v in w["meanfree_shift"])
    X, y = _population(means=1.5, fluct=2.0)            # and in their fluctuations
    w = r.within_arm(X, y, n_perm=40, n_null=5)
    assert w["p_one_unrestricted_both_halves"] == 1 / 41 and all(v > 0.1 for v in w["meanfree_shift"])


def test_within_arm_removes_each_halfs_class_means_and_threads_agree():
    X, y = _population()
    w1 = r.within_arm(X, y, n_perm=12, n_null=4, time_blocks=r.block_ids(len(y), 40), workers=1)
    w4 = r.within_arm(X, y, n_perm=12, n_null=4, time_blocks=r.block_ids(len(y), 40), workers=4)
    for h in ("half0", "half1"):
        assert w1[h]["null_unrestricted"]["draws"] == w4[h]["null_unrestricted"]["draws"]
        assert w1[h]["null_blocks"]["draws"] == w4[h]["null_blocks"]["draws"]
        # the Gram-based shift and decompose's total on the projected rows are one quantity, the same floor draws
        assert abs(w1[h]["meanfree_shift"] - w1[h]["meanfree"]["delta"]) < 1e-10
    h = (np.arange(len(y)) % 2) == 0
    Xh = X[h] - X[h].mean(0)
    Q = r._mean_span(Xh, y[h], 8)
    Z = Xh - (Xh @ Q) @ Q.T
    assert np.abs(np.stack([Z[y[h] == c].mean(0) for c in range(8)])).max() < 1e-10   # every class mean at the grand mean


def test_meanfree_surrogate_and_head_rankings():
    X, y = _population()
    Y, ly = r.bias_null(X, y, "meanfree")
    assert np.allclose(np.stack([Y[ly == c].mean(0) for c in range(8)]), Y.mean(0))
    assert np.allclose(Y - Y.mean(0) - (X - np.stack([X[y == c].mean(0) for c in range(8)])[y]), 0)
    # two units: one organized by its fluctuations alone, one by its class means alone
    Xf, yf = _population(means=0.0, n=128, d=16)
    Xm, _ = _population(means=1.5, fluct=0.0, n=128, d=16, seed=1, y=yf)
    rows = r.head_scores(np.stack([Xf, Xm])[None], yf, [0], n_perm=40, k_orders=10, split=False)
    f, m = rows
    assert m["classmean_frac_z"] > 10 and abs(f["classmean_frac_z"]) < 3 and f["z"] > 3   # z reads the fluctuation unit, share z does not


def test_command_line_arms(tmp_path):
    X, y = _population(n=240)
    np.save(tmp_path / "X.npy", X); np.save(tmp_path / "y.npy", y)
    p = subprocess.run([sys.executable, "-m", "rung", "data", "X.npy", "y.npy", "--n-perm", "20", "--k-orders", "4", "--floor-draws", "4",
                        "--arm", "both", "--out", "r.json"], cwd=tmp_path, capture_output=True, text=True, check=True)
    res = json.load(open(tmp_path / "r.json"))
    assert {"share", "z", "p_one"} <= set(res["between"]) and len(res["within"]["meanfree_shift"]) == 2
    assert "between arm" in p.stdout and "within arm" in p.stdout
    s = subprocess.run([sys.executable, "-m", "rung", "summarize", "r.json"], cwd=tmp_path, capture_output=True, text=True, check=True)
    assert "between arm (the class means alone)" in s.stdout and "within arm (class means removed)" in s.stdout
