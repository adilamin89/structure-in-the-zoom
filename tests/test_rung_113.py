"""Tests for rung 1.13.0: each arm's null keeps what its claim leaves free. The between arm's wild bootstrap holds its level
when the classes' fluctuations differ, where label permutations do not; the within arm's Freedman-Lane null is the observed
procedure applied to fitted class means plus permuted residuals, computed from Gram blocks; the paper's permutation nulls stay
available and are what --paper-floor replays."""
import json
import subprocess
import sys

import numpy as np

import rung as r


def _unequal(seed, K=6, n_per=30, d=60, r_dim=10):
    """Equal class means, each class fluctuating in its own random r_dim-dim subspace: the between arm's H0."""
    g = np.random.default_rng(seed)
    y = np.repeat(np.arange(K), n_per); g.shuffle(y)
    B = [np.linalg.qr(g.standard_normal((d, r_dim)))[0] for _ in range(K)]
    X = np.stack([B[c] @ g.standard_normal(r_dim) for c in y]) + 0.05 * g.standard_normal((len(y), d))
    return X, y


def test_wild_null_holds_its_level_where_permutation_does_not():
    zw, zp = [], []
    for s in range(40):
        X, y = _unequal(s)
        zw.append(r.classmean_share(X, y, n_perm=100, seed=s)["z"])
        zp.append(r.classmean_share(X, y, n_perm=100, seed=s, null="permutation")["z"])
    assert 0.7 < np.std(zw, ddof=1) < 1.35 and abs(np.mean(zw)) < 0.5      # calibrated: z about N(0, 1) under H0
    assert np.std(zp, ddof=1) > 1.4 * np.std(zw, ddof=1)                   # permutations mix the fluctuations H0 leaves free


def test_wild_null_keeps_each_class_covariance_and_detects_means():
    X, y = _unequal(1)
    X = X + 0.6 * np.random.default_rng(9).standard_normal((6, X.shape[1]))[y]        # class means that differ
    b = r.classmean_share(X, y, n_perm=100, seed=0)
    assert b["null"] == "wild" and b["z"] > 5 and b["p_one"] == 1 / 101
    with_strata = r.classmean_share(X, y, n_perm=50, seed=0, strata=np.arange(len(y)) % 3)
    assert with_strata["stratified"] and with_strata["p_one"] == 1 / 51


def test_freedman_lane_gram_blocks_equal_the_materialized_null():
    g = np.random.default_rng(4)
    y = np.repeat(np.arange(8), 24); g.shuffle(y)
    X = g.standard_normal((8, 40))[y] * 1.2 + g.standard_normal((len(y), 40)) * (1 + (y[:, None] == np.arange(40)[None, :] // 5))
    w = r.within_arm(X, y, n_perm=3, n_null=4, seed=42, perm_seed=500, null="freedman-lane")
    assert w["null"] == "freedman-lane"
    lab = np.searchsorted(np.unique(y), y); nc = 8; bc = r._rung_classes(nc, None)
    for hv in (0, 1):
        sel = np.where(np.arange(len(y)) % 2 == hv)[0]
        Xh, yh = X[sel] - X[sel].mean(0), lab[sel]
        M = np.stack([Xh[yh == c].mean(0) for c in range(nc)]); R = Xh - M[yh]
        rng = np.random.default_rng(500 + 10 * hv)
        for k in range(3):
            pi = r._relabel_within(np.arange(len(yh)), None, rng)
            Xs = M[yh] + R[pi]                                               # the null data, materialized
            P = Xs @ r._mean_span(Xs, yh, nc)
            ref = r._gram_shift(Xs @ Xs.T - P @ P.T, yh, nc, bc, 4, 42)
            assert abs(ref - w[f"half{hv}"]["null_unrestricted"]["draws"][k]) < 1e-9


def test_command_line_names_the_nulls_and_paper_floor_replays_permutation(tmp_path):
    g = np.random.default_rng(0)
    y = np.repeat(np.arange(8), 30); g.shuffle(y)
    X = g.standard_normal((8, 48))[y] + g.standard_normal((len(y), 48))
    np.save(tmp_path / "X.npy", X); np.save(tmp_path / "y.npy", y)
    base = [sys.executable, "-m", "rung", "data", "X.npy", "y.npy", "--n-perm", "12", "--k-orders", "4", "--floor-draws", "4", "--arm", "both"]
    for extra, nb, nw in (([], "wild", "freedman-lane"), (["--arm-null", "permutation"], "permutation", "permutation"),
                          (["--paper-floor"], "permutation", "permutation")):
        p = subprocess.run(base + extra + ["--out", "r.json"], cwd=tmp_path, capture_output=True, text=True, check=True)
        res = json.load(open(tmp_path / "r.json"))
        assert (res["between"]["null"], res["within"]["null"]) == (nb, nw), extra
        assert ("Freedman-Lane" in p.stdout) == (nw == "freedman-lane") and ("random signs" in p.stdout) == (nb == "wild")


def test_permutation_null_is_rung_112():
    """null='permutation' draws the 1.12 labellings in the 1.12 order: its share null equals a direct recomputation."""
    X, y = _unequal(3)
    b = r.classmean_share(X, y, n_perm=30, seed=7, null="permutation")
    rng = np.random.default_rng(7)
    lab = np.searchsorted(np.unique(y), y)
    vals = r._classmean_shares(X - X.mean(0), [lab[rng.permutation(len(lab))] for _ in range(30)], 6)
    assert abs(np.mean(vals) - b["null_mean"]) < 1e-14 and abs(np.std(vals) - b["null_sd"]) < 1e-14


def test_calibration_artifact_supports_the_docstrings():
    """The ranges classmean_share's and within_arm's docstrings quote, read back from scripts_canonical/null_calibration.py's
    artifact (200 seeds per regime)."""
    import os
    d = json.load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_canonical", "null_calibration.json")))
    cal = d["calibration"]
    assert d["design"]["seeds"] == 200 and len(cal) == 5
    rng = lambda k, f: (min(v[k][f] for v in cal.values()), max(v[k][f] for v in cal.values()))
    gauss = [v for name, v in cal.items() if "heavy" not in name]
    doc_b, doc_w = r.classmean_share.__doc__, r.within_arm.__doc__
    lo, hi = rng("between_wild", "z_sd"); a, b = rng("between_wild", "typeI_p05")
    assert f"z SD {lo:.2f} to {hi:.2f}, one-sided type-I {100 * a:.1f} to {100 * b:.1f}%" in " ".join(doc_b.split())
    a, b = rng("between_permutation", "typeI_p05")
    assert f"type-I {100 * a:.1f} to {100 * b:.0f}%" in " ".join(doc_b.split())
    lo, hi = rng("within_freedman_lane", "z_sd"); a, b = rng("within_freedman_lane", "typeI_p05")
    assert f"z SD {lo:.2f} to {hi:.2f}, one-sided type-I {100 * a:.1f} to {100 * b:.1f}%" in " ".join(doc_w.split())
    a, b = min(v["within_permutation"]["typeI_p05"] for v in gauss), max(v["within_permutation"]["typeI_p05"] for v in gauss)
    heavy = cal["n/PR 4, heavy gains"]["within_permutation"]["typeI_p05"]
    assert f"type-I {100 * a:.0f} to {100 * b:.1f}%), liberal under heavy-tailed gains ({100 * heavy:.0f}%)" in " ".join(doc_w.split())


def test_within_arm_under_the_calibrated_null():
    """REFERENCE.md and README.md's sentences on the 32 label-blind halves under Freedman-Lane, read back from
    scripts_canonical/meanfree_freedman_lane.py's artifact; the observed shifts are the paper's to the bit."""
    import os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    s = json.load(open(os.path.join(root, "data_canonical", "meanfree_freedman_lane.json")))
    nd, cells = s["summary"]["n_draws"], s["cells"]
    assert len(cells) == 32 and s["summary"]["max_abs_obs_vs_paper"] == 0.0
    ok = [c["fl_p_unr"] <= 0.05 and c["fl_p_epoch"] <= 0.05 for c in cells]
    miss = [c for c, k in zip(cells, ok) if not k]
    assert sum(ok) == 31 and len(miss) == 1 and min(c["obs"] for c in cells) == miss[0]["obs"]
    m = miss[0]
    ref = " ".join(open(os.path.join(root, "REFERENCE.md")).read().split())
    assert (f"clears both nulls at p <= 0.05 in 31 of the 32 halves ({nd} draws each" in ref
            and f"the smallest shift of the 32, {m['obs']:+.3f}, at p = {round(m['fl_p_unr'] * (nd + 1))}/{nd + 1} unrestricted and "
                f"{round(m['fl_p_epoch'] * (nd + 1))}/{nd + 1} inside epochs" in ref)
    assert ("a null that keeps each half's class means and permutes its residuals (Freedman-Lane) in 31 of the 32 halves"
            in " ".join(open(os.path.join(root, "README.md")).read().split()))
