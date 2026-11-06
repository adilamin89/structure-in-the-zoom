"""Days of the week and months of the year as cyclic classes.

Design (the planted-cycle design of run28 / arXiv App G): 16 carrier sentences per axis, the class word rotated through
every carrier, so classes differ only by the planted word. Two placements per axis: 'inner' (the class word sits inside the
sentence, the read-out last token is carrier text, so the embedding layer carries no class information) and 'final' (the
class word is the last word, read at its own token, as Engels et al. 2025 read the day/month circles). Prompts are
class-major: index = class * 16 + carrier.

Per layer (hidden states from llm_battery.py encode):
  floors: ordinary (random same-size prompt subsets) and carrier-matched (each draw takes, for every carrier, exactly the
    number of prompts the rung holds from it, class versions at random; run61's floor);
  shifts: cycle order (averaged over the n origins x 2 directions), a far-step order (step 3 for C7, the antipodal-paired
    order o, o+6, o+1, o+7, ... for C12; averaged over origins and directions), and the average over 200 random orders;
  null: within-carrier label permutations (each carrier's n versions get a random permutation of the n labels), each
    averaged over 50 random orders (order-averaged statistic) or over the same origins/directions (cycle and far-step);
  class-mean kernel: centred class means, cosine kernel, circulant profile f(D), harmonic shares on the n-cycle, against
    the within-carrier-permutation null; circularity: the angular order of the class means in their top-2 PC plane
    (number of cycle neighbours that are angular neighbours); within-class fluctuation overlap adjacent minus antipodal.
Out: data_canonical/cyclic_axes_order_<model>.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import llm_battery as hb  # noqa: E402

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December"]

DAYS_INNER = [
    "On {c} the office opens late because of the staff meeting",
    "Every {c} my grandmother bakes bread for the whole street",
    "The museum is free on {c} for students and teachers",
    "We usually go swimming on {c} after work",
    "Last {c} the train was delayed by almost an hour",
    "The farmers market comes to town every {c} morning",
    "On {c} evenings the library hosts a chess club",
    "My sister calls me every {c} to talk about her week",
    "The garbage truck comes on {c} before sunrise",
    "Our team plays its home games on {c} nights",
    "The new restaurant is closed on {c} for cleaning",
    "Each {c} the children practice piano for an hour",
    "The report is due on {c} at noon sharp",
    "He always wears a blue shirt on {c} to the office",
    "The ferry does not run on {c} during the winter",
    "On {c} the park fills with families and dogs",
]
DAYS_FINAL = [
    "The office opens late this week because of a meeting on",
    "My grandmother bakes bread for the whole street every",
    "The museum is free for students and teachers on",
    "After work we usually go swimming on",
    "The train was delayed by almost an hour last",
    "The farmers market comes to town every",
    "The library hosts a chess club in the evening on",
    "My sister calls me to talk about her week every",
    "The garbage truck comes before sunrise on",
    "Our team plays its home games at night on",
    "The new restaurant is closed for cleaning on",
    "The children practice piano for an hour each",
    "The report is due at noon sharp on",
    "He always wears a blue shirt to the office on",
    "During the winter the ferry does not run on",
    "The park fills with families and dogs on",
]
MONTHS_INNER = [
    "In {c} the town holds a festival by the river",
    "The school year here begins in {c} for most children",
    "Last {c} we drove to the coast to visit friends",
    "Every {c} the company publishes its sales report",
    "The bridge was closed in {c} for repairs to the deck",
    "My cousin was born in {c} and loves to celebrate",
    "The garden looks different in {c} than in other months",
    "In {c} the prices at the hotel go up sharply",
    "The orchestra tours the northern cities each {c}",
    "We signed the lease in {c} after a long search",
    "The magazine's {c} issue sold out in two days",
    "In early {c} the fishermen repair their nets",
    "The election was held in {c} after a short campaign",
    "Every {c} the family gathers at the old farmhouse",
    "The museum reopened in {c} with a new exhibit",
    "By the end of {c} the project was nearly finished",
]
MONTHS_FINAL = [
    "The town holds a festival by the river every year in",
    "For most children the school year here begins in",
    "We drove to the coast to visit friends last",
    "The company publishes its sales report every",
    "The bridge was closed for repairs to the deck in",
    "My cousin loves to celebrate because she was born in",
    "The garden looks completely different in",
    "The prices at the hotel go up sharply in",
    "The orchestra tours the northern cities each",
    "After a long search we signed the lease in",
    "The magazine sold out its whole issue in",
    "The fishermen repair their nets in early",
    "After a short campaign the election was held in",
    "The family gathers at the old farmhouse every",
    "With a new exhibit the museum reopened in",
    "The project was nearly finished by the end of",
]
SETS = {"days": (DAYS, DAYS_INNER, DAYS_FINAL), "months": (MONTHS, MONTHS_INNER, MONTHS_FINAL)}
COUNTS = {7: [1, 2, 3, 4, 5, 7], 12: [1, 2, 3, 4, 6, 8, 12]}
N_CARRIER = 16   # set to 15 for cyclic_months_inner15 (main)
N_NULL, K_ORD, N_ORD200 = 200, 50, 200


def battery(name):
    _, axis, place = name.split("_")
    classes, inner, final = SETS[axis]
    templates = inner if place.startswith("inner") else final
    if place == "inner15":       # amendment: MONTHS_INNER[8] ends on the class word; dropped
        templates = [t for i, t in enumerate(inner) if i != 8]
    prompts, labels = [], []
    for ci, c in enumerate(classes):
        for t in templates:
            prompts.append(t.format(c=c) if place.startswith("inner") else f"{t} {c}")
            labels.append(ci)
    return prompts, np.array(labels), list(classes), COUNTS[len(classes)]


BATTERIES = ["cyclic_days_inner", "cyclic_days_final", "cyclic_months_inner", "cyclic_months_final"]


def cycle_orders(n):
    out = []
    for o in range(n):
        for s in (1, -1):
            out.append([(o + s * k) % n for k in range(n)])
    return out


def far_orders(n):
    out = []
    for o in range(n):
        for s in (1, -1):
            if n % 2 == 0:   # antipodal pairs: o, o+n/2, o+1, o+1+n/2, ...
                seq = []
                for k in range(n // 2):
                    seq += [(o + s * k) % n, (o + s * k + n // 2) % n]
            else:            # step-3 traversal (a generator of C7 whose steps are maximal)
                seq = [(o + s * 3 * k) % n for k in range(n)]
            assert sorted(seq) == list(range(n)), seq
            out.append(seq)
    return out


def carrier_floor(K, labels, counts, seed, n_draw=20):
    """Each draw: for a rung of k classes every carrier contributes k prompts, class versions at random."""
    n = len(labels)
    nc = labels.max() + 1
    carrier = np.arange(n) % N_CARRIER
    by_carrier = [np.where(carrier == c)[0] for c in range(N_CARRIER)]
    rng = np.random.default_rng(seed)
    nl = np.zeros((n_draw, len(counts)))
    sizes = [k * N_CARRIER for k in counts]
    for d in range(n_draw):
        for j, k in enumerate(counts):
            idx = np.concatenate([rng.choice(bc, k, replace=False) for bc in by_carrier])
            nl[d, j] = np.log(max(hb.r37.subset_pr(K, idx), 1e-9))
    assert nc == len(by_carrier[0])
    return hb.r17.slope(sizes, np.exp(nl.mean(axis=0)))


def within_carrier_perm(labels, rng):
    n = len(labels)
    nc = labels.max() + 1
    carrier = np.arange(n) % N_CARRIER
    sl = labels.copy()
    for c in range(N_CARRIER):
        idx = np.where(carrier == c)[0]
        sl[idx] = rng.permutation(nc)
    return sl


def kernel_stats(X, labels, nc):
    means = np.stack([X[labels == c].mean(axis=0) for c in range(nc)])
    cm = means - means.mean(axis=0)
    nrm = np.linalg.norm(cm, axis=1, keepdims=True) + 1e-12
    C = (cm / nrm) @ (cm / nrm).T
    f = np.array([np.mean([C[i, (i + d) % nc] for i in range(nc)]) for d in range(nc)])
    spec = np.abs(np.fft.rfft(f)) ** 2
    nonzero = spec[1:]
    h_share = (nonzero / nonzero.sum()).tolist() if nonzero.sum() > 0 else [0.0] * len(nonzero)
    # circularity in the top-2 PC plane of the class means
    U, S, Vt = np.linalg.svd(cm, full_matrices=False)
    ang = np.angle(U[:, 0] * S[0] + 1j * U[:, 1] * S[1])
    order = np.argsort(ang)
    pos = np.empty(nc, int)
    pos[order] = np.arange(nc)
    neigh = sum(1 for c in range(nc) if (pos[(c + 1) % nc] - pos[c]) % nc in (1, nc - 1))
    pc2_share = float((S[:2] ** 2).sum() / (S ** 2).sum())
    anti = nc // 2
    return {"f": f.tolist(), "harmonic_share": h_share, "h1_share": h_share[0],
            "f1_minus_fanti": float(f[1] - f[anti]), "cycle_neighbours_in_pc_plane": int(neigh),
            "pc2_share": pc2_share}


def fluct_overlap(X, labels, nc):
    """Within-class covariance overlap Tr(Sa Sb)/sqrt(Tr Sa^2 Tr Sb^2), adjacent minus antipodal, via Gram traces."""
    R = [X[labels == c] - X[labels == c].mean(axis=0) for c in range(nc)]
    G = {}
    for a in range(nc):
        for b in range(a, nc):
            M = R[a] @ R[b].T
            G[(a, b)] = float((M * M).sum())
    def ov(a, b):
        a, b = min(a, b), max(a, b)
        return G[(a, b)] / np.sqrt(G[(a, a)] * G[(b, b)])
    adj = np.mean([ov(c, (c + 1) % nc) for c in range(nc)])
    anti = np.mean([ov(c, (c + nc // 2) % nc) for c in range(nc)])
    return float(adj), float(anti)


def analyse_battery(H, labels, names, counts):
    n_layers, n = H.shape[0], H.shape[1]
    nc = len(names)
    mem0 = hb.members_of(labels, nc)
    cyc, far = cycle_orders(nc), far_orders(nc)
    orng = np.random.default_rng(3801)
    rand200 = [orng.permutation(nc).tolist() for _ in range(N_ORD200)]
    nrng = np.random.default_rng(4900)
    draws = [(within_carrier_perm(labels, nrng), [nrng.permutation(nc).tolist() for _ in range(K_ORD)])
             for _ in range(N_NULL)]
    for lab, _ in draws[:5]:
        assert np.array_equal(np.bincount(lab), np.bincount(labels))
    rows = []
    for l in range(n_layers):
        t0 = time.time()
        X = H[l].astype(np.float64)
        K = hb.gram(X, "linear")
        sizes = [k * N_CARRIER for k in counts]
        floors = {"ordinary": hb.floor(K, n, sizes, l * 100 + 1), "carrier": carrier_floor(K, labels, counts, l * 100 + 2)}
        th = {"cycle": np.mean([hb.ladder(K, mem0, o, counts)[0] for o in cyc]),
              "far": np.mean([hb.ladder(K, mem0, o, counts)[0] for o in far])}
        per_rand = np.array([hb.ladder(K, mem0, o, counts)[0] for o in rand200])
        th["orderavg"] = float(per_rand.mean())
        nulls = {"cycle": [], "far": [], "orderavg": []}
        kstats_null = []
        for lab, ords in draws:
            mem = hb.members_of(lab, nc)
            nulls["cycle"].append(np.mean([hb.ladder(K, mem, o, counts)[0] for o in cyc]))
            nulls["far"].append(np.mean([hb.ladder(K, mem, o, counts)[0] for o in far]))
            nulls["orderavg"].append(np.mean([hb.ladder(K, mem, o, counts)[0] for o in ords]))
            ks = kernel_stats(X, lab, nc)
            kstats_null.append((ks["h1_share"], ks["f1_minus_fanti"]))
        ks = kernel_stats(X, labels, nc)
        h1n = np.array([a for a, _ in kstats_null])
        adj, anti = fluct_overlap(X, labels, nc)
        row = {"layer": l, "floors": floors, "kernel": ks,
               "h1_null_mean": float(h1n.mean()), "h1_null_sd": float(h1n.std()),
               "h1_p_one": float((1 + (h1n >= ks["h1_share"]).sum()) / (N_NULL + 1)),
               "fluct_overlap_adjacent": adj, "fluct_overlap_antipodal": anti,
               "cycle_pct_among_orders": float((per_rand < th["cycle"]).mean())}
        for fl, thf in floors.items():
            for st in ("cycle", "far", "orderavg"):
                d = float(th[st] - thf)
                nv = np.array(nulls[st]) - thf
                nm, ns = float(nv.mean()), float(nv.std())
                row[f"{st}_{fl}"] = {"delta": d, "null_mean": nm, "null_sd": ns,
                                     "p_one": float((1 + (nv >= d).sum()) / (N_NULL + 1)),
                                     "p_two": float((1 + (np.abs(nv - nm) >= abs(d - nm)).sum()) / (N_NULL + 1))}
        dn = np.array(nulls["cycle"]) - np.array(nulls["far"])
        D = float(th["cycle"] - th["far"])
        row["D_cycle_minus_far"] = {"D": D, "null_mean": float(dn.mean()), "null_sd": float(dn.std()),
                                    "p_two": float((1 + (np.abs(dn - dn.mean()) >= abs(D - dn.mean())).sum())
                                                   / (N_NULL + 1))}
        rows.append(row)
        print(f"  L{l:02d} oa(carrier)={row['orderavg_carrier']['delta']:+.3f} p1={row['orderavg_carrier']['p_one']:.3f}"
              f" | cycle={row['cycle_carrier']['delta']:+.3f} far={row['far_carrier']['delta']:+.3f}"
              f" D={D:+.3f}(p2 {row['D_cycle_minus_far']['p_two']:.3f}) | oa(ord)={row['orderavg_ordinary']['delta']:+.3f}"
              f" | h1={ks['h1_share']:.2f} (null {h1n.mean():.2f}, p {row['h1_p_one']:.3f}) f1-fa={ks['f1_minus_fanti']:+.2f}"
              f" circ={ks['cycle_neighbours_in_pc_plane']}/{nc} | ovl adj-anti={adj - anti:+.3f}"
              f" ({time.time() - t0:.0f}s)", flush=True)
    return {"class_names": names, "rung_counts": counts, "n_prompts": int(n), "layers": rows}


def main():
    mkey = sys.argv[1]
    bats = sys.argv[2:] or BATTERIES
    out_path = hb.OUTDIR / f"cyclic_axes_order_{mkey}.json"
    out = json.load(open(out_path)) if out_path.exists() else {"model_key": mkey, "batteries": {}}
    global N_CARRIER
    for b in bats:
        N_CARRIER = 15 if b.endswith("inner15") else 16
        f = hb.STATES / f"{mkey}__{b}.npz"
        if b.endswith("inner15") and not f.exists():   # slice the 16-carrier states: each prompt is its own forward pass
            z16 = np.load(hb.STATES / f"{mkey}__cyclic_months_inner.npz")
            keep = np.array([i for i in range(len(z16["labels"])) if i % 16 != 8])
            np.savez(f, H=z16["H"][:, keep], labels=z16["labels"][keep], names=z16["names"], prompts=z16["prompts"][keep],
                     model=z16["model"], device=z16["device"], dtype=z16["dtype"])
        z = np.load(f)
        prompts, labels, names, counts = battery(b)
        assert np.array_equal(labels, z["labels"]) and list(z["prompts"]) == prompts
        print(f"[{mkey} / {b}] H {z['H'].shape} {z['dtype']}", flush=True)
        res = analyse_battery(z["H"], labels, names, counts)
        res["dtype"] = str(z["dtype"])
        out["batteries"][b] = res
        json.dump(out, open(out_path, "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
