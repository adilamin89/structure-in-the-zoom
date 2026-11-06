"""rung report: one self-contained HTML page per result (data, feature ladder, language model), read from the JSON alone."""
import json
import os
import subprocess
import sys

import numpy as np

import rung as r

ROOT = os.path.join(os.path.dirname(__file__), "..")


def _toy(seed=0):
    rng = np.random.default_rng(seed)
    y = np.repeat(np.arange(8), 32)
    X = rng.standard_normal((256, 32)) * (1 + 2 * (y[:, None] == np.arange(32)[None, :] // 4))
    return X, y


def _page(tmp_path, result, name="r.json", **kw):
    p = tmp_path / name
    p.write_text(json.dumps(result, default=float))
    return open(r.report(str(p), **kw)).read()


def test_data_report_holds_figure_reading_and_numbers(tmp_path):
    X, y = _toy()
    res = r.zoom(X, y, n_perm=30, k_orders=5, n_floor_draws=5)
    res["decompose"] = r.decompose(X, y, n_null=5)
    page = _page(tmp_path, res)
    assert page.count("data:image/png;base64,") == 2 and "http" not in page.split("Reading rules")[0]   # self-contained
    assert "the shift against its nulls, its split and harmonics" in page                         # the summary panel
    for line in r.summarize_data(res, verbose=False)["lines"]:
        assert line.strip().replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("'", "&#x27;") in page
    sign = "-" if res["delta"] < 0 else "+"
    assert f"{res['theta_floor']:+.3f} {sign} {abs(res['delta']):.3f}" in page and "+ +" not in page and "+ -" not in page
    assert "within-class part D" in page and "p cannot fall below" in page
    assert page.index("<h2>Reading</h2>") < page.index("<h2>Numbers</h2>")


def test_feature_ladder_and_title_are_escaped(tmp_path):
    X, _ = _toy()
    res = r.feature_ladder(X, order=np.arange(32), sizes=[4, 8, 16, 32], n_floor_draws=5)
    page = _page(tmp_path, res, title="<b>my run</b>")
    assert "&lt;b&gt;my run&lt;/b&gt;" in page and "<b>my run</b>" not in page
    assert "feature-ladder result" in page and "exchangeable control" in page
    assert page.count("data:image/png;base64,") == 1                                              # no summary panel


def test_language_model_report_from_a_shipped_battery(tmp_path):
    src = os.path.join(ROOT, "data_canonical", "heldout_qwen05_world_knowledge32.json")
    out = tmp_path / "lm.html"
    page = open(r.report(src, str(out))).read()
    assert "language-model result" in page and "Qwen/Qwen2.5-0.5B" in page and "<h3>world_knowledge32</h3>" in page
    assert page.count("data:image/png;base64,") == 1


def test_command_line_writes_the_page_beside_the_json(tmp_path):
    X, y = _toy(1)
    np.save(tmp_path / "X.npy", X); np.save(tmp_path / "y.npy", y)
    run = lambda *a: subprocess.run([sys.executable, "-m", "rung", *a], cwd=tmp_path, capture_output=True, text=True, check=True)
    run("data", "X.npy", "y.npy", "--n-perm", "20", "--k-orders", "3", "--floor-draws", "3", "--out", "r.json")
    assert json.load(open(tmp_path / "r.json"))["rung_version"] == r.__version__
    out = run("report", "r.json").stdout
    assert "wrote r.html" in out and (tmp_path / "r.html").exists()


def test_head_scores_report(tmp_path):
    rng = np.random.default_rng(2)
    labels = np.repeat(np.arange(6), 10)
    H = rng.standard_normal((2, 4, 60, 12))
    H[1, 2] *= 1 + 2 * (labels[:, None] == np.arange(12)[None, :] % 6)   # head 1.2 writes class-specific fluctuations (a lift)
    rows = r.head_scores(H, labels, [0, 1], n_perm=10, k_orders=4, n_floor_draws=3)
    res = {"model": "toy", "axis": "toy_axis", "classes": list(range(6)), "n_prompts": 60, "paper_seeds": False,
           "n_perm": 10, "k_orders": 4, "heads": rows, "known": ["1.2"],
           "AUC_z": r._auc(np.array([h["z"] for h in rows]), np.array([h["head"] == "1.2" for h in rows]))}
    page = _page(tmp_path, res)
    assert "attention-head result" in page and "ranked by z: 1.2" in page and "AUC of z against the known heads" in page
    assert page.count("data:image/png;base64,") == 1
