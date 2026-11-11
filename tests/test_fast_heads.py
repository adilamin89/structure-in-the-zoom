"""Class-block ladders (rung 1.9.0): head_scores computes every class-union ladder from class-block sums of the centred
Gram; it must replay the row-by-row ladder it replaced, seeds and draw order unchanged."""
import numpy as np
import pytest

import rung as r


def _reference_head_scores(H, labels, layers, paper_seeds=False, n_perm=100, k_orders=50, n_floor_draws=20, seed=0):
    """The 1.8.1 computation of the order-averaged shift, its null and z, row by row (kept here as the reference)."""
    labels = np.asarray(labels)
    lab = np.searchsorted(np.unique(labels), labels)
    nc, n = lab.max() + 1, len(lab)
    bc = r._rung_classes(nc, None)
    mem = [np.where(lab == c)[0] for c in range(nc)]
    orng = np.random.default_rng(3801 if paper_seeds else seed + 3801)
    orders = [orng.permutation(nc).tolist() for _ in range(k_orders)]
    nrng = np.random.default_rng(4800 if paper_seeds else seed + 4800)
    null_draws = []
    for _ in range(n_perm):
        pl = lab[nrng.permutation(n)]
        null_draws.append(([np.where(pl == c)[0] for c in range(nc)], [nrng.permutation(nc).tolist() for _ in range(k_orders)]))
    rows = []
    for j, l in enumerate(layers):
        for h in range(H.shape[1]):
            X = H[j, h].astype(np.float64)
            Xs = X / (X.std() + 1e-9)
            K = Xs @ Xs.T
            sizes = r._ladder(K, mem, list(range(nc)), bc)[1]
            frng = np.random.default_rng(l * 100 + h + 1 if paper_seeds else [seed, l, h])
            lg = np.array([[np.log(max(r._subset_pr(K, frng.choice(n, s_, replace=False)), 1e-9)) for s_ in sizes]
                           for _ in range(n_floor_draws)])
            th_f = r._slope(sizes, np.exp(lg.mean(axis=0)))
            dbar = float(np.mean([r._ladder(K, mem, o, bc)[0] for o in orders])) - th_f
            vals = np.array([np.mean([r._ladder(K, m_, o, bc)[0] for o in os_]) - th_f for m_, os_ in null_draws])
            rows.append({"delta_orderavg": dbar, "z": float((dbar - vals.mean()) / vals.std()), "p_two": r._null_stats(dbar, vals)[1]})
    return rows


def _heads(offset=0.0, seed=0):
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(6), 12)
    H = rng.standard_normal((2, 3, len(labels), 16))
    H[:, 0] += 0.8 * rng.standard_normal((6, 16))[labels]                    # head 0 writes the class means
    H[:, 1] *= 1 + 1.5 * (labels[:, None] == np.arange(16)[None, :] % 6)     # head 1 writes class-specific fluctuations
    return H + offset, labels


@pytest.mark.parametrize("paper_seeds", [False, True])
@pytest.mark.parametrize("offset", [0.0, 1e3])
def test_block_ladders_replay_the_row_by_row_ladders(paper_seeds, offset):
    H, labels = _heads(offset)
    kw = dict(paper_seeds=paper_seeds, n_perm=8, k_orders=6, n_floor_draws=4)
    fast = r.head_scores(H, labels, [3, 7], split=False, order_average="random", **kw)   # the reference reads random orders
    ref = _reference_head_scores(H, labels, [3, 7], **kw)
    for a, b in zip(fast, ref):
        assert a["delta_orderavg"] == pytest.approx(b["delta_orderavg"], abs=1e-9)
        assert a["z"] == pytest.approx(b["z"], abs=1e-7)
        assert a["p_two"] == b["p_two"]


def test_union_ladder_slopes_equal_ladder_slopes_on_unequal_classes():
    rng = np.random.default_rng(3)
    lab = rng.integers(0, 8, 200)
    X = rng.standard_normal((200, 30)) + 2.0 * rng.standard_normal((8, 30))[lab]
    Xg = X - X.mean(0)
    K = Xg @ Xg.T
    mem = [np.where(lab == c)[0] for c in range(8)]
    bc = r._rung_classes(8, None)
    orders = [rng.permutation(8).tolist() for _ in range(5)]
    fast = r._union_ladder_slopes(r._class_blocks(K, lab, 8), orders, bc)
    slow = [r._ladder(K, mem, o, bc)[0] for o in orders]
    assert np.allclose(fast, slow, atol=1e-10, rtol=0)


def test_block_ladders_match_an_svd_reference_under_a_large_common_offset():
    # features whose means are 50 times their SD: block sums of the UNcentred Gram would cancel catastrophically
    # (~1e-6); on the centred Gram the slopes match the SVD of each row-centred union to 1e-10
    rng = np.random.default_rng(5)
    lab = np.repeat(np.arange(6), 20)
    X = 50.0 + rng.standard_normal((120, 24)) * (1 + (lab[:, None] == np.arange(24)[None, :] % 6))
    Xg = X - X.mean(0)
    bc = r._rung_classes(6, None)
    orders = [rng.permutation(6).tolist() for _ in range(4)]
    fast = r._union_ladder_slopes(r._class_blocks(Xg @ Xg.T, lab, 6), orders, bc)

    def pr_svd(rows):
        s = np.linalg.svd(X[rows] - X[rows].mean(0), compute_uv=False) ** 2
        return s.sum() ** 2 / (s ** 2).sum()

    for o, f in zip(orders, fast):
        sets = [np.concatenate([np.where(lab == c)[0] for c in o[:k]]) for k in bc]
        assert f == pytest.approx(r._slope([len(s_) for s_ in sets], [pr_svd(s_) for s_ in sets]), abs=1e-10)
