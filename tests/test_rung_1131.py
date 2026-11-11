"""Tests for rung 1.13.1: the Python defaults draw the floor 20 times as the command line does; summarize states a p at its
floor and a sign law whose threshold is below zero in words that hold; a nuisance-keeping null's verdict says what it licenses;
labels may be words, and a non-integer label stops the run instead of merging classes."""
import json
import subprocess
import sys

import numpy as np
import pytest

import rung as r


def _toy(seed=0):
    g = np.random.default_rng(seed)
    y = np.repeat(np.arange(8), 40)
    X = g.standard_normal((len(y), 48))
    X *= 1 + 2 * (y[:, None] == np.arange(48) // 6)
    return X, y


def test_python_defaults_draw_the_floor_as_the_cli_does():
    X, y = _toy()
    a, b = r.decompose(X, y), r.decompose(X, y, n_null=20)
    assert a["delta_split_free"] == b["delta_split_free"] and a["delta"] == b["delta"]
    assert a["delta_split_free"] != r.decompose(X, y, n_null=10)["delta_split_free"]
    w0, w1 = r.within_arm(X, y, n_perm=0), r.within_arm(X, y, n_perm=0, n_null=20)
    assert w0["half0"]["meanfree_shift"] == w1["half0"]["meanfree_shift"]


def test_p_floor_note():
    assert r._p_floor_note(1 / 501, 500) == "(at its floor 1/(500 + 1): no permutation reached the observed value)"
    assert r._p_floor_note(18 / 501, 500) == "(floor 1/(500 + 1) = 0.002)"


def test_sign_law_clause_with_a_negative_threshold():
    sl = {"threshold_full": -5.4, "alignment": 1.51, "trB_trW": 0.158, "ratio": 14.6}
    txt = r._sign_law_clause(sl)
    assert txt.startswith("always negative here") and "1 + beta/2 = 1.08" in txt and "exactly when" not in txt
    assert 1 + 2 * (1 - 1.51) / 0.158 < 0                                  # the threshold is negative exactly when a > 1 + beta/2
    txt = r._sign_law_clause({"threshold_full": 22.0, "alignment": 0.86, "trB_trW": 0.013, "ratio": 8.4})
    assert txt.startswith("negative exactly when PR(W)/PR(B) = 8.4 exceeds")


def test_sign_law_threshold_is_negative_exactly_when_alignment_exceeds_one_plus_half_beta():
    g = np.random.default_rng(3)
    seen = set()
    for s in range(24):
        X, y = _toy(s)
        u = np.linalg.qr(g.standard_normal((X.shape[1], 2)))[0]
        X = X + 4 * g.standard_normal((len(y), 1)) * u[:, 0]          # one strong fluctuation shared by every class
        w = np.cos(s * np.pi / 46) * u[:, 0] + np.sin(s * np.pi / 46) * u[:, 1]   # class means from along it to off it
        X = X + 0.6 * (y[:, None] - 3.5) * w
        sl = r.sign_law(X, y)
        neg = sl["threshold_full"] < 0
        assert neg == (sl["alignment"] > 1 + sl["trB_trW"] / 2)
        if neg:
            assert sl["c_full"] < 0 and sl["law_says_negative"]
        seen.add(neg)
    assert seen == {True, False}


def test_absorbed_verdict_says_what_the_null_licenses():
    X, y = _toy()
    res = r.zoom(X, y, n_perm=19, k_orders=0, strata=np.arange(len(y)) % 2)
    res.update(p_two=0.01, strat_p_two=0.4, n_perm=200)
    txt = " ".join(r.summarize_data(res, verbose=False)["lines"])
    assert "compatible with the strata's composition" in txt and "reads nuisance composition, not the labels" not in txt


def _run(tmp_path, *args):
    return subprocess.run([sys.executable, r.__file__, *args], cwd=tmp_path, capture_output=True, text=True)


def test_word_labels_equal_integer_labels(tmp_path):
    X, y = _toy()
    np.save(tmp_path / "X.npy", X); np.save(tmp_path / "y.npy", y)
    names = np.array(["ant", "bee", "cat", "dog", "eel", "fox", "gnu", "hen"])
    np.savetxt(tmp_path / "w.csv", names[y], fmt="%s")
    np.save(tmp_path / "w.npy", names[y])
    a = _run(tmp_path, "data", "X.npy", "y.npy", "--n-perm", "0", "--out", "a.json")
    b = _run(tmp_path, "data", "X.npy", "w.csv", "--n-perm", "0", "--out", "b.json")
    c = _run(tmp_path, "data", "X.npy", "w.npy", "--n-perm", "0", "--out", "c.json")
    assert a.returncode == b.returncode == c.returncode == 0, (b.stderr, c.stderr)
    assert "classes coded in sorted order" in b.stdout
    ja, jb, jc = (json.load(open(tmp_path / f)) for f in ("a.json", "b.json", "c.json"))
    assert ja["delta"] == jb["delta"] == jc["delta"] and jb["label_values"] == list(names)
    d = _run(tmp_path, "data", "X.npy", "w.csv", "--n-perm", "0", "--order", ",".join(names[::-1]), "--out", "d.json")
    e = _run(tmp_path, "data", "X.npy", "y.npy", "--n-perm", "0", "--order", "7,6,5,4,3,2,1,0", "--out", "e.json")
    assert json.load(open(tmp_path / "d.json"))["delta"] == json.load(open(tmp_path / "e.json"))["delta"]
    f = _run(tmp_path, "data", "X.npy", "w.csv", "--n-perm", "0", "--order", "ant,yak")
    assert f.returncode != 0 and "yak" in f.stderr


def test_non_integer_labels_stop_the_run(tmp_path):
    X, y = _toy()
    np.save(tmp_path / "X.npy", X); np.save(tmp_path / "half.npy", y + 0.5); np.save(tmp_path / "whole.npy", y.astype(float))
    bad = _run(tmp_path, "data", "X.npy", "half.npy", "--n-perm", "0")
    assert bad.returncode == 2 and "non-integer values (e.g. 0.5)" in bad.stderr
    ok = _run(tmp_path, "data", "X.npy", "whole.npy", "--n-perm", "0")
    assert ok.returncode == 0


def test_version_is_one_string():
    import pathlib
    path = pathlib.Path(r.__file__).parent / "pyproject.toml"
    if not path.exists():
        pytest.skip("rung installed without its source tree")
    toml = path.read_text()
    assert f'version = "{r.__version__}"' in toml          # the release pins its own number in test_rung_114


def test_split_class_dimension_is_not_capped_by_the_sample_count():
    g = np.random.default_rng(0)
    d = 300; lam = 1 / np.arange(1, d + 1) ** 0.7; true = lam.sum() ** 2 / (lam ** 2).sum()     # population PR 86.3
    X = g.standard_normal((32, d)) * np.sqrt(lam); y = np.zeros(32, int)
    assert r.class_pr(X, y)[0] < 31 < 0.75 * true < r.class_pr(X, y, "split")[0] < 1.3 * true
    Xs = g.standard_normal((40, 30)); ys = np.zeros(40, int)                  # the feature path (n > d) equals the Gram path
    Xg = np.hstack([Xs, np.zeros((40, 20))])
    assert r.class_pr(Xs, ys, "split", seed=5)[0] == pytest.approx(r.class_pr(Xg, ys, "split", seed=5)[0], rel=1e-12)
    Xc, yc = _toy()
    m = r.class_moments(Xc, yc, seed=0)                                          # the same estimator as Appendix K's moments
    assert np.allclose(sorted(r.class_pr(Xc, yc, "split").values()), np.sort(m["t"] ** 2 / np.diag(m["T"])), rtol=1e-9)


def test_sampling_line_warns_when_classes_are_sampled_below_their_dimension():
    g = np.random.default_rng(1)
    d = 200; lam = 1 / np.arange(1, d + 1) ** 0.7
    y = np.repeat(np.arange(8), 24)
    X = g.standard_normal((len(y), d)) * np.sqrt(lam) + 3 * g.standard_normal((8, d))[y]
    res = r.zoom(X, y, n_perm=0, k_orders=0); res["decompose"] = r.decompose(X, y)
    txt = " ".join(r.summarize_data(res, verbose=False)["lines"])
    assert "from split halves" in txt and "reads low" in txt


def test_sampling_line_reads_each_class_against_its_own_dimension():
    """One large high-dimensional class and seven small low-dimensional ones, each sampled beyond its own dimension: no warning, though
    the average class size sits below the largest class's dimension (Codex round 2, the average-vs-largest comparison of 1.13.0)."""
    g = np.random.default_rng(2)
    sizes = [512] + [32] * 7
    y = np.repeat(np.arange(8), sizes)
    rows = []
    for c, n in enumerate(sizes):
        k = 200 if c == 0 else 4
        lam = np.zeros(200); lam[:k] = 1 / np.arange(1, k + 1) ** 0.3
        rows.append(g.standard_normal((n, 200)) * np.sqrt(lam) + 2 * g.standard_normal(200))
    X = np.vstack(rows)
    res = r.zoom(X, y, n_perm=0, k_orders=0); res["decompose"] = r.decompose(X, y)
    d = res["decompose"]
    n_pc = d["rungs"][-1]["size"] / 8
    assert n_pc < max(d["class_pr_split"].values())                       # the old rule would have warned
    assert all(d["class_n"][k] > v for k, v in d["class_pr_split"].items())
    txt = " ".join(r.summarize_data(res, verbose=False)["lines"])
    assert "from split halves" in txt and "reads low" not in txt
