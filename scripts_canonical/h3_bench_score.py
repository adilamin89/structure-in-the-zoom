"""H3-R14 scorer: B1 (layers above null), B2 (path vs partition), B3 (lexical share from H2's last-token arms), each number
printed before its reading. Reads data_canonical/h3_bench/{h3_bench,h2_lasttoken,h2_lasttoken_power}_<key>.json.
Out: data_canonical/h3_bench/h3_bench_score.json"""
import json
from pathlib import Path

import numpy as np

D = Path(__file__).resolve().parent.parent / "data_canonical" / "h3_bench"
KEYS = ("pythia-160m", "pythia-2.8b-deduped")


def main():
    out = {}
    for key in KEYS:
        f = D / f"h3_bench_{key}.json"
        if not f.exists():
            continue
        A = json.load(open(f))["axes"]
        lt = json.load(open(D / f"h2_lasttoken_{key}.json"))["batteries"] if (D / f"h2_lasttoken_{key}.json").exists() else {}
        lp = json.load(open(D / f"h2_lasttoken_power_{key}.json")) if (D / f"h2_lasttoken_power_{key}.json").exists() else {}
        for a, r in A.items():
            R = r["layers"]
            n = len(R)
            above = [x["dbar50_run37"] > 0 and x["perm"]["p_one"] <= 0.05 for x in R]
            row = {"n_layers": n, "B1_layers_above_null": f"{sum(above)}/{n}",
                   "B1_past_embedding": f"{sum(above[1:])}/{n - 1}",
                   "B1_failing_layers": [x["layer"] for x, g in zip(R, above) if not g],
                   "dbar50_range_past_emb": [min(x["dbar50_run37"] for x in R[1:]), max(x["dbar50_run37"] for x in R[1:])],
                   "excess_range_past_emb": [min(x["dbar50_run37"] - x["perm"]["null_mean"] for x in R[1:]),
                                             max(x["dbar50_run37"] - x["perm"]["null_mean"] for x in R[1:])],
                   "embedding": {"dbar50": R[0]["dbar50_run37"], "p_one": R[0]["perm"]["p_one"],
                                 "delta_declared": R[0]["delta_declared"]},
                   "declared_pct_range": [min(x["declared_pct"] for x in R), max(x["declared_pct"] for x in R)],
                   "B2_orderavg_IE": r["profile"]["orderavg"]["IE"], "B2_orderavg_IE_p_two": r["profile"]["orderavg"]["p_two"],
                   "B2_reading": "path (order-averaged profile inside its null)" if r["profile"]["orderavg"]["p_two"] > 0.05
                   else "partition (order-averaged profile outside its null)",
                   "declared_IE": r["profile"]["declared"]["IE"], "declared_IE_p_two": r["profile"]["declared"]["p_two"]}
            if "strat_source" in R[0]:
                st = [x["dbar50_run37"] > 0 and x["strat_source"]["p_one"] <= 0.05 for x in R]
                row["within_source_null_above"] = f"{sum(st)}/{n}"
            if a in lt:
                t = lt[a]
                row["distinct_last_tokens"] = t["distinct_last_tokens"]
                row["last_tokens_examples"] = [x[:6] for x in t["last_tokens_per_class"]]
                row["spearman_distinct_vs_classPR"] = t["spearman_distinct_vs_classPR"]
                row["m_matched"] = t["m"]
                if isinstance(t["matched"], list) and a in lp:
                    M = t["matched"]
                    P = lp[a]["random_matched_size"]
                    half = n // 2
                    share = {}
                    for tag, rng in (("layers_1_to_half", range(1, half + 1)), ("layers_half_to_last", range(half, n))):
                        mm = np.mean([M[l]["dbar_median"] for l in rng])
                        rr = np.mean([P[l]["dbar_median"] for l in rng])
                        share[tag] = {"matched_mean": float(mm), "random_mean": float(rr),
                                      "lexical_share": float((rr - mm) / rr) if rr != 0 else None}
                    row["B3"] = share
                    row["B3_layers_above_null_matched"] = int(sum(x["p_median"] <= 0.05 for x in M[1:]))
                    row["B3_layers_above_null_random"] = int(sum(x["p_median"] <= 0.05 for x in P[1:]))
                    row["B3_embedding"] = {"matched": M[0]["dbar_median"], "random": P[0]["dbar_median"]}
                else:
                    row["B3"] = "matched arm impossible (a class has fewer than 6 distinct last tokens)"
                if "within_lasttoken_strata" in t:
                    S = t["within_lasttoken_strata"]
                    row["lasttoken_strata_null_above"] = f"{sum(x['dbar'] > 0 and x['p_one'] <= 0.05 for x in S)}/{len(S)}"
                if a in lp:
                    row["movable_fraction"] = lp[a]["movable_fraction"]
            out[f"{key}/{a}"] = row
    json.dump(out, open(D / "h3_bench_score.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
