"""rung 1.15: the class means and the fluctuations by symmetry sector (sector_split, rung data --sectors), on known answers."""
import os

import numpy as np
import pytest

import rung

HERE = os.path.dirname(os.path.abspath(__file__))


def _random(K=8, D=30, n=40, seed=0):
    rng = np.random.default_rng(seed)
    lab = np.repeat(np.arange(K), n)
    X = rng.standard_normal((K * n, D)) @ rng.standard_normal((D, D)) + 0.7 * rng.standard_normal((K, D))[lab]
    return X, lab


def test_exact_identities():
    """Any data, equal class weights: sum beta_l = beta, a = sum E_l a_l, the closed form of c equals c, and with balanced classes
    beta and c equal sign_law's (which weights classes by size)."""
    X, lab = _random()
    s = rung.sector_split(X, lab, n_jack=0, n_null=5, n_perm=0)
    sec = s["sectors"]
    assert abs(sum(v["beta_raw"] for v in sec) - s["beta"]) < 1e-12
    assert abs(sum(v["E_raw"] * v["a_raw"] for v in sec) - s["alignment"]) < 1e-10
    assert abs(s["c_closed"] - s["c_exact"]) < 1e-12
    sl = rung.sign_law(X, lab)
    assert abs(sl["trB_trW"] - s["beta"]) < 1e-10 and abs(sl["c_full"] - s["c_exact"]) < 1e-10


def test_kernel_selection_rule():
    """S_c = R^c S_0 R^-c with V = 1 + E_h (h = 3 of K = 16) embedded in 40 features, plus isotropic noise: the population kernel's
    harmonics vanish off {0, h, 2h} (the kernel selection rule, exact)."""
    K, h, D = 16, 3, 40
    rng = np.random.default_rng(1)
    Q = np.linalg.qr(rng.standard_normal((D, 3)))[0]
    A = rng.standard_normal((3, 3)); S0 = A @ A.T
    S = []
    for c in range(K):
        th = 2 * np.pi * h * c / K
        R = np.eye(3); R[1:, 1:] = [[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]
        S.append(Q @ (R @ S0 @ R.T) @ Q.T + 0.1 * np.eye(D))
    t = np.array([np.trace(s) for s in S]); T = np.array([[np.trace(a @ b) for b in S] for a in S])
    _, kappa, _ = rung._kernel_harmonics(T / np.outer(t, t))
    allowed = {0, h, 2 * h}
    assert max(abs(kappa[l]) for l in range(K // 2 + 1) if l not in allowed) < 1e-12
    assert min(abs(kappa[l]) for l in allowed) > 1e-6


def test_planted_alignment_by_sector():
    """Means at harmonic 1 along the two strongest fluctuation axes (a_1 > 1) and at harmonic 2 along two weak ones (a_2 < 1): over
    20 seeds at 96 rows per class the cross-fitted a_l sit within 3 SE of the truth (a planted design with known alignments)."""
    K, D, n = 8, 120, 96
    lam = 9.0 / np.arange(1, D + 1) ** 0.7
    ang = 2 * np.pi * np.arange(K) / K
    M = np.zeros((K, D)); M[:, 0] = 0.6 * np.cos(ang); M[:, 1] = 0.6 * np.sin(ang); M[:, 60] = 0.9 * np.cos(2 * ang); M[:, 61] = 0.9 * np.sin(2 * ang)
    trW, trW2 = lam.sum(), (lam ** 2).sum()
    truth = {1: ((lam[0] + lam[1]) / 2) / (trW2 / trW), 2: ((lam[60] + lam[61]) / 2) / (trW2 / trW)}
    got, shown = {1: [], 2: []}, {1: 0, 2: 0}
    for s in range(20):
        rng = np.random.default_rng(s); lab = np.repeat(np.arange(K), n)
        X = M[lab] + rng.standard_normal((K * n, D)) * np.sqrt(lam)
        r = rung.sector_split(X, lab, n_jack=0, n_null=20, n_perm=0, seed=s)
        for l in (1, 2):
            v = r["sectors"][l]
            got[l].append(v["x_x"] / v["beta_x"])                  # the estimator on every seed: no selection by the gate
            shown[l] += v["a_shown"]
        assert not r["sectors"][3]["a_shown"]                     # no harmonic-3 means: beta_3 inside its null
    assert shown[1] >= 18 and shown[2] == 20                      # beta_1 sits about 6 null SD up: the 3 SD gate drops a rare draw
    for l in (1, 2):
        v = np.array(got[l]); se = v.std(ddof=1) / np.sqrt(len(v))
        assert abs(v.mean() - truth[l]) < 3 * se, (l, v.mean(), truth[l], se)
    assert np.mean(got[1]) > 1 > np.mean(got[2])


@pytest.mark.parametrize("name", ["xy_L16_J10.42_T0.60.npz", "xy_L16_J10.42_T1.00.npz"])
def test_xy_order_parameters(name):
    """XY configurations (L = 16, polar coupling 0.42): classes = bins of each configuration's polar angle (16) or director (8); the
    cross-fitted amplitude net of the bins gives the polar order <|z1|> and the nematic order <|z2|> within 1% (the order parameters' normalisation)."""
    cfg = np.load(os.path.join(HERE, "data", name))["configs"]
    z1, z2 = np.exp(1j * cfg).mean(1), np.exp(2j * cfg).mean(1)
    feat = np.concatenate([np.cos(cfg), np.sin(cfg), np.cos(2 * cfg), np.sin(2 * cfg)], axis=1)
    l1 = np.minimum((np.angle(z1) % (2 * np.pi) / (2 * np.pi) * 16).astype(int), 15)
    l2 = np.minimum(((np.angle(z2) / 2) % np.pi / np.pi * 8).astype(int), 7)
    for lab, K, ref in ((l1, 16, np.abs(z1).mean()), (l2, 8, np.abs(z2).mean())):
        s = rung.sector_split(feat, lab, binned=True, n_jack=0, n_null=10, n_perm=0)
        assert abs(np.sqrt(2) * s["sectors"][1]["phi_x_binfree"] / ref - 1) < 0.01, (K, ref)


def test_kernel_jackknife_and_equivariance():
    """Classes whose fluctuations are one shape: kappa_l (l >= 1) within 3 jackknife SE of zero, the jackknife SE moves little with
    the split seed (50 splits; the replicates keep the splits) and kappa's split noise is a small part of it, and a kernel that
    rotates with the class shows a circulant share above its class-position permutations."""
    K, D, n = 8, 40, 60
    rng = np.random.default_rng(3); lab = np.repeat(np.arange(K), n)
    X = rng.standard_normal((K * n, D)) * np.linspace(2, 0.5, D)
    runs = [rung.sector_split(X, lab, n_null=5, n_perm=0, seed=0, split_seed=sd)["kernel"] for sd in range(6)]   # same groups
    a = runs[0]
    for l in range(1, K // 2 + 1):
        assert abs(a["kappa"][l]) < 3 * a["kappa_se"][l]
    se = np.array([r["kappa_se"][1:] for r in runs]); kap = np.array([r["kappa"][1:] for r in runs])
    assert np.all(np.abs(se / se.mean(0) - 1) < 0.15)              # the split draws move the jackknife SE by about 10%
    assert np.all(kap.std(0) < 0.25 * se.mean(0))                  # and kappa itself by a small part of its SE
    # fluctuations that rotate with the class: a strong axis turning by 2 pi c / K in the first two features
    Y = X.copy()
    for c in range(K):
        th = np.pi * c / K
        u = np.zeros(D); u[0], u[1] = np.cos(th), np.sin(th)
        Y[lab == c] += 4 * rng.standard_normal((n, 1)) * u
    k = rung.sector_split(Y, lab, n_null=5, n_perm=500, n_jack=0)["kernel"]
    assert k["circ_p"] < 0.05 and k["circ_share"] > k["circ_null_mean"]


def test_pairing_and_thin_classes():
    """Rows shared across classes (pair ids spanning classes) refuse the cross-fit; so do classes under 8 rows; a crossed design
    (every class the same carriers row for row) puts the kernel's sigma above 1 by more than two jackknife SE and warns."""
    X, lab = _random(n=40)
    strata = np.tile(np.arange(40), 8)                               # carrier i appears in every class
    s = rung.sector_split(X, lab, pairs=strata, n_jack=0, n_null=5, n_perm=0)
    assert s["pairing"]["paired"] and not s["crossfit"]["used"] and "shared" in s["crossfit"]["refused"]
    Xt, lt = _random(n=6)
    t = rung.sector_split(Xt, lt, n_jack=0, n_null=5, n_perm=0)
    assert not t["crossfit"]["used"] and "fewer than 8" in t["crossfit"]["refused"]
    rng = np.random.default_rng(4)
    carriers = rng.standard_normal((40, 30)) * 3
    Xc = np.vstack([carriers + 0.3 * rng.standard_normal((40, 30)) + 0.5 * rng.standard_normal(30) for _ in range(8)])
    c = rung.sector_split(Xc, np.repeat(np.arange(8), 40), n_null=5, n_perm=0)
    assert c["pairing"]["warn"] and c["pairing"]["sigma"] - 2 * c["pairing"]["sigma_se"] > 1
    lines = rung._sector_split_lines(c)
    assert any("crossed or paired design" in l for l in lines)


def test_cli_sectors_keeps_the_kernel_harmonics(tmp_path):
    """rung data --sectors writes sector_split beside the class-mean kernel harmonics, which keep their 1.14 values (a fixture
    written by rung 1.14.0's sectors() on the same data, tests/data/sectors_114_fixture.json)."""
    import json
    import subprocess
    import sys
    X, lab = _random(n=24)
    np.save(tmp_path / "X.npy", X); np.save(tmp_path / "y.npy", lab)
    env = {**os.environ, "PYTHONPATH": os.path.dirname(HERE)}
    subprocess.run([sys.executable, "-m", "rung", "data", "X.npy", "y.npy", "--sectors", "--n-perm", "0", "--k-orders", "0",
                    "--sector-null", "5", "--sector-jackknife", "0", "--out", "r.json"], cwd=tmp_path, check=True, env=env,
                   capture_output=True)
    r = json.load(open(tmp_path / "r.json"))
    assert r["sectors"] == json.loads(json.dumps(rung.sectors(X, lab)))
    assert r["sectors"] == json.load(open(os.path.join(HERE, "data", "sectors_114_fixture.json")))["sectors"]
    assert r["sector_split"]["frame"] == "features as given"
    assert r["sector_split"]["n_classes"] == 8 and len(r["sector_split"]["sectors"]) == 5


@pytest.mark.skipif(os.environ.get("RUNG_DATA_TESTS") != "1", reason="raw recordings: set RUNG_DATA_TESTS=1")
def test_gt3_by_sector():
    """GT3 (raw/gratings_drifting_GT3_2019_04_05_1.npy, eight 45-degree direction classes): the class-mean term c and its raw
    first-order sum by sector equal the extended version's (-0.3193 and -0.1625, equal class weights)."""
    root = os.path.dirname(HERE)
    d = np.load(os.path.join(root, "raw", "gratings_drifting_GT3_2019_04_05_1.npy"), allow_pickle=True).item()
    X = np.asarray(d["sresp"], np.float32).T.astype(np.float64)
    phi = np.asarray(d["istim"], float) % (2 * np.pi)
    lab = np.clip(np.digitize(phi, np.linspace(0, 2 * np.pi, 9)) - 1, 0, 7)
    s = rung.sector_split(X, lab, n_jack=0, n_null=5, n_perm=0)
    assert round(s["c_exact"], 4) == -0.3193 and round(s["c1_raw_sum"], 4) == -0.1625
    assert s["sectors"][2]["c1_raw"] < -0.2 and s["sectors"][2]["a_x"] > 1 > s["sectors"][1]["a_x"]


def _equal_weight_reference(X, lab):
    """beta and c from their definitions, independently of rung: equal class weights, W the mean of the classes' covariances
    (divisor n_c), B the covariance of the class means about their unweighted mean."""
    K = lab.max() + 1
    M = np.stack([X[lab == c].mean(0) for c in range(K)]); M -= M.mean(0)
    W = sum(np.cov(X[lab == c].T, bias=True) for c in range(K)) / K
    B = M.T @ M / K
    pr = lambda C: np.trace(C) ** 2 / np.trace(C @ C)
    return np.trace(B) / np.trace(W), np.log(pr(W + B)) - np.log(pr(W))


def test_unbalanced_classes_against_the_definitions():
    """Unequal class sizes and odd K: beta and c equal their equal-weight definitions computed directly (not from rung's sector
    sums), and the sectors add up to them."""
    rng = np.random.default_rng(11)
    for counts in ([9, 12, 20, 15, 11], [8, 30, 10, 14, 9, 21, 12]):
        lab = np.repeat(np.arange(len(counts)), counts)
        X = rng.standard_normal((len(lab), 6)) * rng.gamma(1.0, size=6) + 2 * np.cos(2 * np.pi * lab / len(counts))[:, None]
        s = rung.sector_split(X, lab, n_jack=0, n_null=0, n_perm=0)
        b, c = _equal_weight_reference(X, lab)
        assert abs(s["beta"] - b) < 1e-12 and abs(s["c_exact"] - c) < 1e-12 and abs(s["c_closed"] - c) < 1e-12


def test_kernel_zero_in_expectation():
    """Identical classes: every kernel entry from split halves, so kappa_l (l >= 1) is zero in expectation. One feature: every
    entry is 1, kappa_l = 0 exactly (the full-class diagonal t_c^2 instead gives -2/(n + 1) on every kappa_l). Five features, 16
    rows per class: the mean over 40 data sets within 3 SE of zero, where the t_c^2 diagonal sits more than 3 SE below it."""
    K, n = 8, 16
    lab = np.repeat(np.arange(K), n)
    X1 = np.random.default_rng(0).standard_normal((K * n, 1))
    k1 = rung.sector_split(X1, lab, n_jack=0, n_null=0, n_perm=0)["kernel"]["kappa"]
    assert np.all(np.abs(np.array(k1[1:])) < 1e-12)
    new, old = [], []
    for s in range(40):
        X = np.random.default_rng(100 + s).standard_normal((K * n, 5)) * np.array([2.0, 1.5, 1.0, 0.7, 0.4])
        new.append(rung.sector_split(X, lab, n_jack=0, n_null=0, n_perm=0, seed=s)["kernel"]["kappa"][1:])
        mom = rung.class_moments(X, lab, seed=s)
        old.append(rung._kernel_harmonics(mom["T"] / np.outer(mom["t"], mom["t"]))[1][1:])
    for a, ok in ((np.array(new), True), (np.array(old), False)):
        z = a.mean(0) / (a.std(0, ddof=1) / np.sqrt(len(a)))
        assert np.all(np.abs(z) < 3) == ok, z


def test_pairing_warning_on_independent_classes():
    """Identical independent classes, the hardest case for the pairing check (sigma near 1; a ratio, slightly above 1 on average at
    these counts): no warning in 40 data sets at 16 rows per class, 30 features."""
    K, n = 8, 16
    lab = np.repeat(np.arange(K), n)
    w = [rung.sector_split(np.random.default_rng(s).standard_normal((K * n, 30)), lab, n_null=0, n_perm=0, seed=s)["pairing"]["warn"]
         for s in range(40)]
    assert sum(w) == 0


def test_null_is_the_reported_statistics():
    """The null of beta_l^x is the same n_splits-split mean on permuted labels, so averaging more splits narrows it."""
    K, n = 8, 16
    lab = np.repeat(np.arange(K), n)
    X = np.random.default_rng(5).normal(size=(K * n, 10))
    sd = {m: np.array([v["beta_x_null_sd"] for v in rung.sector_split(X, lab, n_splits=m, n_null=200, n_jack=0, n_perm=0)["sectors"][1:]])
          for m in (1, 10)}
    assert np.all(sd[10] < sd[1])


def test_undefined_and_degenerate_inputs():
    """A split with no shared fluctuation direction refuses the cross-fit (no NaN); overlaps that do not vary leave the circulant
    share undefined with no p; a constant class leaves the kernel undefined; malformed inputs raise; one split prints no SE."""
    lab = np.repeat(np.arange(3), 8)
    X = np.zeros((24, 3)); X[7, 0] = X[15, 1] = X[23, 2] = 1
    s = rung.sector_split(X, lab, n_null=2, n_jack=0, n_perm=0)
    assert not s["crossfit"]["used"] and "share no direction" in s["crossfit"]["refused"] and "c1_x_sum" not in s
    lab4 = np.repeat(np.arange(4), 16)
    Xs = (np.tile([-1.0, 1.0], 32) + np.repeat(2 * np.arange(4), 16))[:, None]
    k = rung.sector_split(Xs, lab4, n_null=5, n_jack=0, n_perm=19)["kernel"]
    assert "circ_undefined" in k and "circ_p" not in k
    Xz = np.random.default_rng(1).normal(size=(64, 3)); Xz[lab4 == 0] = 0
    assert "undefined" in rung.sector_split(Xz, lab4, n_null=2, n_jack=0, n_perm=5)["kernel"]
    Xg, lg = _random(K=4, n=16, seed=2)
    for kw in ({"n_splits": 0}, {"n_jack": 1}, {"n_jack": -1}, {"kernel_splits": 0}, {"n_perm": -1}, {"pairs": np.zeros(5)},
               {"pairs": np.full(64, np.nan)}, {"pairs": np.full(64, np.nan, dtype=object)}, {"pairs": np.array([None] * 64)},
               {"n_splits": 1.0}, {"n_jack": 2.0}, {"kernel_splits": 1.0},
               {"pairs": np.array([np.float32(np.nan)] * 64, dtype=object)}, {"pairs": np.full(64, np.datetime64("NaT", "ns"))},
               {"pairs": np.array([complex(np.nan, 0)] * 64, dtype=object)}):
        with pytest.raises(ValueError):
            rung.sector_split(Xg, lg, **kw)
    with pytest.raises(ValueError):
        rung.sector_split(Xg[:, 0], lg)
    with pytest.raises(ValueError):
        rung.sector_split(Xg, np.array([None] + list(lg[1:]), dtype=object))
    Xg[:, 0] += 5 * np.cos(2 * np.pi * lg / 4)
    one = rung.sector_split(Xg, lg, n_splits=1, n_null=3, n_jack=0, n_perm=0)
    assert one["sectors"][1]["a_shown"] and any("one split: no SE" in l for l in rung._sector_split_lines(one))
    shown = [v for v in rung.sector_split(Xg, lg, n_null=20, n_jack=0, n_perm=0)["sectors"] if v["a_shown"]]
    assert shown and all(v["a_x_se"] > 0 and np.isfinite(v["a_x_se"]) for v in shown)


def test_cli_frame_and_paired_account(tmp_path):
    """--standardize: the sector split reads the z-scored frame decompose reads (balanced classes: the same c); --strata declared
    --paired and spanning classes: --account reports the identity undefined, while nuisance strata (the same ids, not declared
    pairs) leave it as it was; bad --sector-* values stop before any analysis."""
    import json
    import subprocess
    import sys
    rng = np.random.default_rng(3)
    lab = np.repeat(np.arange(4), 16)
    X = rng.normal(size=(64, 5)) * np.array([1, 3, 0.5, 2, 1]); X[:, 0] += 5 * np.cos(2 * np.pi * lab / 4)
    np.save(tmp_path / "X.npy", X); np.save(tmp_path / "y.npy", lab); np.save(tmp_path / "st.npy", np.tile(np.arange(16), 4))
    env = {**os.environ, "PYTHONPATH": os.path.dirname(HERE)}
    base = [sys.executable, "-m", "rung", "data", "X.npy", "y.npy", "--n-perm", "0", "--k-orders", "0", "--workers", "1",
            "--floor-draws", "2"]
    run = lambda *x: subprocess.run(base + list(x), cwd=tmp_path, env=env, capture_output=True, text=True)
    assert run("--standardize", "--sectors", "--decompose", "--n-bias", "0", "--sector-null", "3", "--out", "z.json").returncode == 0
    r = json.load(open(tmp_path / "z.json"))
    assert abs(r["sector_split"]["c_exact"] - r["decompose"]["sign_law"]["c_full"]) < 1e-9
    assert r["sector_split"]["frame"].startswith("features z-scored")
    assert run("--account", "--no-heldout", "--strata", "st.npy", "--paired", "--out", "p.json").returncode == 0
    assert "independent classes" in json.load(open(tmp_path / "p.json"))["account"]["undefined"]
    assert run("--account", "--no-heldout", "--strata", "st.npy", "--out", "n.json").returncode == 0
    assert run("--account", "--no-heldout", "--out", "a.json").returncode == 0
    assert json.load(open(tmp_path / "n.json"))["account"] == json.load(open(tmp_path / "a.json"))["account"]
    assert run("--sectors", "--sector-null", "3", "--strata", "st.npy", "--out", "s.json").returncode == 0
    assert json.load(open(tmp_path / "s.json"))["sector_split"]["crossfit"]["used"]
    bad = run("--sectors", "--sector-splits", "0", "--out", "b.json")
    assert bad.returncode != 0 and "--sector-splits" in bad.stderr and not (tmp_path / "b.json").exists()


def test_kernel_zero_at_unequal_counts():
    """Identical Gaussian row laws, unequal counts (4, 64, 4, 64): halves of one size in every class keep kappa_2 zero in
    expectation (halves of unequal sizes put E kappa_2 = ||M_h - M_j||^2 > 0 on alternating counts)."""
    lab = np.repeat(np.arange(4), [4, 64, 4, 64])
    k2 = [rung.sector_split(np.random.default_rng(s).standard_normal((len(lab), 2)) * np.array([4.0, 1.0]), lab, n_jack=0, n_null=0,
                            n_perm=0, kernel_splits=3, seed=s)["kernel"]["kappa"][2] for s in range(400)]
    assert abs(np.mean(k2)) < 3 * np.std(k2, ddof=1) / np.sqrt(len(k2))


def test_sigma_undefined_and_legacy_kernel_label():
    """A kernel whose mean diagonal overlap is zero leaves sigma undefined (no Infinity in the JSON); a kernel written before the
    split-half estimator is not labelled as one."""
    import json
    lab = np.repeat(np.arange(3), 4)
    X = np.tile([[1.0, 0], [-1, 0], [0, 1], [0, -1]], (3, 1))
    s = rung.sector_split(X, lab, n_jack=0, n_null=0, n_perm=0, kernel_splits=1, split_seed=19)
    assert s["kernel"]["sigma"] is None and "sigma_undefined" in s["kernel"] and "Infinity" not in json.dumps(s)
    X2, l2 = _random(n=24)
    r = rung.sector_split(X2, l2, n_jack=0, n_null=3, n_perm=0)
    old = json.loads(json.dumps(r)); del old["kernel"]["estimator"]
    assert any("split halves" in x for x in rung._sector_split_lines(r))
    assert not any("split halves" in x for x in rung._sector_split_lines(old))


def test_circulant_p_at_unequal_counts():
    """Identical non-Gaussian row law in every class, counts 10 and 100 alternating (the population kernel is constant): the
    class-position p rejects at 0.05 within three binomial SD of 5% over 100 data sets (each class contributing the same fixed
    subset of rows makes the classes exchangeable; with each class's own rows it ran near 20%)."""
    lab = np.repeat(np.arange(8), [10, 100] * 4)
    rej = 0
    for s in range(100):
        rng = np.random.default_rng(8000 + s)
        ax = rng.random(len(lab)) < 0.95; z = rng.standard_normal(len(lab)); X = np.zeros((len(lab), 2))
        X[ax, 0] = z[ax] / np.sqrt(0.95); X[~ax, 1] = z[~ax] / np.sqrt(0.05)
        X += 0.05 * rng.standard_normal(X.shape)
        rej += rung.sector_split(X, lab, n_null=0, n_jack=0, n_perm=99, seed=s)["kernel"]["circ_p"] <= 0.05
    assert rej <= 100 * 0.05 + 3 * np.sqrt(100 * 0.05 * 0.95), rej


def test_circulant_p_counts_exact_ties():
    """Four classes have three distinct cyclic arrangements: arrangements equal in exact arithmetic tie, so a p at the largest share
    is at least 8/24 of the permutations (roundoff of 3e-16 once made it 1/24)."""
    y = np.repeat(np.arange(4), 16)
    X = np.random.default_rng(91087).normal(size=(64, 3))
    assert rung.sector_split(X, y, n_splits=1, n_null=0, n_jack=0, n_perm=19999, seed=87)["kernel"]["circ_p"] > 0.25
