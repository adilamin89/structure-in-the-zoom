"""Does the language-model topic shift survive its per-layer nulls, and which part of
it is the declared path? Pythia-160M, the paper's battery (topic_class_ladder.TOPICS, 8 topics x 8 prompts, last
non-pad token), states from data_canonical/llm_states_paper_battery.npz (reproduces topic_class_ladder.json at all 13
layers; video_vd1/scripts/00_llm_states.py asserts it). The arXiv draft (Sep 15, Sec 8.2) warned that at 128 prompts a
random relabelling read -0.31 and +0.17 at the first two layers, and that order-averaged profiles were flat; the
camera-ready quotes +0.62 at the embedding layer with a shuffled control at the last layer only.

Per layer: (1) the declared shift, recomputed with topic_class_ladder.ladder_delta (seed 42 + layer, as the artifact);
(2) the label-permutation null, 200 random reassignments of prompts to eight classes of eight at the same rung sizes
(the floor depends only on rung sizes, so it is shared), one-sided p = (1 + #{null >= observed}) / 201;
(3) the order-averaged shift, the mean over 200 random orders of the eight classes, which describes the partition
without the declared path; (4) at layer 0, the number of distinct last tokens per class; (5) the null of the order-averaged statistic, 200
label permutations each averaged over 50 random class orders.
Declared order: topic_class_ladder.ladder_delta takes sorted(set(labels)) of the topic NAMES, so the paper's artifact
accumulates the topics alphabetically (arithmetic, biology, cooking, geography, history, law, sports, weather); the
first run of this script used the dictionary order and did not reproduce the artifact (+0.765 against +0.617 at
layer 0). Labels are mapped back to names here, and the run asserts the artifact is reproduced.

Expectations written before running:
R1 the permutation null is centred within 0.05 of zero at every layer;
R2 the declared shift exceeds every permuted value at every layer (p = 1/201);
R3 the order-averaged shift is positive at every layer;
R4 the embedding-layer declared shift exceeds its order average (the declared order starts on a class whose prompts
   share last tokens), i.e. part of the +0.62 is the path.
Out: data_canonical/llm_path_nulls.json
"""
import json
from pathlib import Path

import numpy as np

from topic_class_ladder import BIN_COUNTS, ladder_delta, pr_c, slope

D = Path(__file__).resolve().parent.parent / "data_canonical"
N_PERM, N_ORDER = 200, 200


def ladder_theta(H, members):
    sizes, prs = [], []
    for k in BIN_COUNTS:
        sel = np.concatenate(members[:k])
        sizes.append(len(sel)); prs.append(pr_c(H[sel]))
    return slope(sizes, prs)


def main():
    z = np.load(D / "llm_states_paper_battery.npz")
    H_all, prompts = z["H"].astype(np.float64), z["prompts"]
    labels = np.array([str(z["names"][i]) for i in z["labels"]])   # topic names: ladder_delta sorts them
    ref = json.load(open(D / "topic_class_ladder.json"))["per_layer"]
    classes = sorted(set(labels.tolist()))
    members = [np.where(labels == c)[0] for c in classes]
    last_tok = [p.split()[-1] for p in prompts]
    distinct = {str(c): len({last_tok[i] for i in m}) for c, m in zip(classes, members)}
    print("declared (alphabetical) order:", classes)
    print("distinct last words per class (declared order):", distinct, flush=True)
    rows = []
    for li in range(H_all.shape[0]):
        H = H_all[li]
        th_o, th_f, d, _ = ladder_delta(H, labels.tolist(), np.random.default_rng(42 + li))
        prng = np.random.default_rng(9000 + li)
        null = []
        for _ in range(N_PERM):
            perm = labels[prng.permutation(len(labels))]
            null.append(ladder_theta(H, [np.where(perm == c)[0] for c in classes]) - th_f)
        null = np.array(null)
        orng = np.random.default_rng(5000 + li)
        oav = np.array([ladder_theta(H, [members[j] for j in orng.permutation(len(classes))]) - th_f
                        for _ in range(N_ORDER)])
        assert abs(d - ref[li]["delta_topic"]) < 1e-3, ("artifact not reproduced", li, d, ref[li]["delta_topic"])
        p = (1 + int((null >= d).sum())) / (N_PERM + 1)
        # null of the order-averaged statistic: permute labels, then average over random class orders
        nrng = np.random.default_rng(13000 + li)
        oav_null = []
        for _ in range(N_PERM):
            perm = labels[nrng.permutation(len(labels))]
            pm = [np.where(perm == c)[0] for c in classes]
            oav_null.append(np.mean([ladder_theta(H, [pm[j] for j in nrng.permutation(len(classes))]) - th_f
                                     for _ in range(50)]))
        oav_null = np.array(oav_null)
        p_oav = (1 + int((oav_null >= oav.mean()).sum())) / (N_PERM + 1)
        rows.append({"layer": li, "delta": d, "delta_artifact": ref[li]["delta_topic"], "floor": th_f,
                     "null_mean": float(null.mean()), "null_sd": float(null.std()), "null_max": float(null.max()),
                     "p_one_sided": p, "order_avg": float(oav.mean()), "order_sd": float(oav.std()),
                     "order_min": float(oav.min()), "order_max": float(oav.max()),
                     "declared_pct_in_orders": float((oav < d).mean()),
                     "order_avg_null_mean": float(oav_null.mean()), "order_avg_null_sd": float(oav_null.std()),
                     "order_avg_null_max": float(oav_null.max()), "p_order_avg": p_oav})
        r = rows[-1]
        print(f"layer {li:2d}  delta {d:+.3f} (artifact {r['delta_artifact']:+.3f})  null {r['null_mean']:+.3f}"
              f" +- {r['null_sd']:.3f} (max {r['null_max']:+.3f}) p={p:.4f}  order-avg {r['order_avg']:+.3f}"
              f" +- {r['order_sd']:.3f} [{r['order_min']:+.3f}, {r['order_max']:+.3f}]"
              f"  declared at {100 * r['declared_pct_in_orders']:.0f}th pct  | order-avg null {r['order_avg_null_mean']:+.3f}"
              f" +- {r['order_avg_null_sd']:.3f} (max {r['order_avg_null_max']:+.3f}) p={p_oav:.4f}", flush=True)
    R1 = all(abs(r["null_mean"]) <= 0.05 for r in rows)
    R2 = all(r["p_one_sided"] <= 1 / (N_PERM + 1) + 1e-12 for r in rows)
    R3 = all(r["order_avg"] > 0 for r in rows)
    R4 = rows[0]["delta"] > rows[0]["order_avg"]
    print(f"R1 null centred within 0.05 at every layer: {R1} (max |mean| "
          f"{max(abs(r['null_mean']) for r in rows):.3f})")
    print(f"R2 declared above every permutation at every layer: {R2} "
          f"(layers failing: {[r['layer'] for r in rows if r['p_one_sided'] > 1 / (N_PERM + 1) + 1e-12]})")
    print(f"R3 order-averaged shift positive at every layer: {R3} "
          f"(range {min(r['order_avg'] for r in rows):+.3f} to {max(r['order_avg'] for r in rows):+.3f})")
    print(f"R4 embedding declared above its order average: {R4} ({rows[0]['delta']:+.3f} vs {rows[0]['order_avg']:+.3f})")
    json.dump({"distinct_last_words": distinct, "rows": rows, "R1": R1, "R2": R2, "R3": R3, "R4": R4,
               "n_perm": N_PERM, "n_order": N_ORDER}, open(D / "llm_path_nulls.json", "w"), indent=1)
    print("DONE")


if __name__ == "__main__":
    main()
