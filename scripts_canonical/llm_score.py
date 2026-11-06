"""Score the language-model predictions from the llm_battery artifacts; every criterion's number is printed
before its verdict. Out: data_canonical/llm_score.json"""
import json
from pathlib import Path

import numpy as np

D = Path(__file__).resolve().parent.parent / "data_canonical"
SCALES = ["pythia-160m", "pythia-410m-deduped", "pythia-1b-deduped", "pythia-2.8b-deduped"]
AXES = ["world_knowledge", "language_type"]


def load(m):
    f = D / f"llm_battery_{m}.json"
    return json.load(open(f)) if f.exists() else None


def main():
    out = {}
    r37 = json.load(open(D / "run37_inferential_nulls.json"))["models"]
    # R1 reproduction
    rep = {}
    for m in ("pythia-160m", "pythia-2.8b-deduped", "pythia-410m-deduped", "pythia-1b-deduped"):
        a = load(m)
        if a is None or m not in r37:
            continue
        for ax in AXES:
            if ax not in a["batteries"]:
                continue
            new = a["batteries"][ax]["layers"]
            old = r37[m]["axes"][ax]["layers"]
            dd = max(abs(n["delta_declared"] - o["delta"]) for n, o in zip(new, old))
            da = max(abs(n["dbar50_run37"] - o["delta_orderavg"]) for n, o in zip(new, old))
            rep[f"{m}/{ax}"] = {"max_abs_diff_declared": dd, "max_abs_diff_orderavg50": da, "pass": dd <= 0.005 and da <= 0.005}
            print(f"R1 {m}/{ax}: max|diff| declared {dd:.2e}, order-avg {da:.2e} -> {'PASS' if rep[f'{m}/{ax}']['pass'] else 'MISS'}")
    out["R1"] = rep
    # L1, L2, L3, R5
    L1, L2, L3, R5 = {}, {}, {}, {}
    for m in SCALES:
        a = load(m)
        if a is None:
            continue
        for ax in AXES:
            if ax not in a["batteries"]:
                continue
            b = a["batteries"][ax]
            rows = b["layers"]
            n = len(rows)
            ok = [r["dbar50_run37"] > 0 and r["perm"]["p_one"] <= 0.05 for r in rows]
            L1[f"{m}/{ax}"] = {"n_layers": n, "n_pass": int(sum(ok)), "frac": sum(ok) / n,
                               "dbar50_range": [min(r["dbar50_run37"] for r in rows), max(r["dbar50_run37"] for r in rows)],
                               "dbar200_range": [min(r["dbar200"] for r in rows), max(r["dbar200"] for r in rows)],
                               "declared_pct_range": [min(r["declared_pct"] for r in rows), max(r["declared_pct"] for r in rows)],
                               "failing_layers": [r["layer"] for r, o in zip(rows, ok) if not o],
                               "pass": sum(ok) / n >= 0.9}
            pr = b["profile"]
            L2[f"{m}/{ax}"] = {"orderavg_IE": pr["orderavg"]["IE"], "orderavg_p_two": pr["orderavg"]["p_two"],
                               "declared_IE": pr["declared"]["IE"], "declared_p_two": pr["declared"]["p_two"],
                               "pass": pr["orderavg"]["p_two"] > 0.05}
            read = [r for r in rows if abs(r["dbar50_run37"]) > r["perm"]["null_sd"]]
            keep = {v: sum(np.sign(r[v]["dbar50"]) == np.sign(r["dbar50_run37"]) for r in read) for v in ("zscore", "pc4")}
            L3[f"{m}/{ax}"] = {"n_read": len(read), "n_unread": n - len(read), **{f"{v}_sign_kept": int(k) for v, k in keep.items()},
                               "d_eff_range": [min(r["spectrum"]["d_eff"] for r in rows), max(r["spectrum"]["d_eff"] for r in rows)],
                               "top1_range": [min(r["spectrum"]["top1_share"] for r in rows),
                                              max(r["spectrum"]["top1_share"] for r in rows)],
                               "pass": len(read) > 0 and all(k / len(read) >= 0.9 for k in keep.values())}
            if ax == "language_type":
                s = [r["strat"]["p_one"] <= 0.05 for r in rows]
                R5[m] = {"n_pass": int(sum(s)), "n_layers": n, "pass": sum(s) / n >= 2 / 3}
    for name, d in (("L1", L1), ("L2", L2), ("L3", L3), ("R5", R5)):
        out[name] = d
        for k, v in d.items():
            print(name, k, {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()})
    # L4 BLiMP, L5 Baroni
    L4, L5 = {}, {}
    for m in SCALES:
        a = load(m)
        if a is None:
            continue
        if "blimp_B" in a["batteries"]:
            rows = a["batteries"]["blimp_B"]["layers"]
            n = len(rows)
            absorbed = sum(r["swap"]["p_two"] > 0.05 for r in rows)
            beats_perm = sum(r["perm"]["p_two"] <= 0.05 for r in rows)
            L4[m] = {"n_layers": n, "swap_absorbs": int(absorbed), "perm_exceeded": int(beats_perm),
                     "dbar50_range": [min(r["dbar50_run37"] for r in rows), max(r["dbar50_run37"] for r in rows)],
                     "pass": absorbed / n >= 0.9 and beats_perm / n >= 0.9}
        if "baroni64" in a["batteries"]:
            rows = a["batteries"]["baroni64"]["layers"]
            n = len(rows)
            late = [r for r in rows if r["layer"] >= n / 2]
            beyond = sum(r["swap"]["p_two"] < 0.05 for r in late)
            L5[m] = {"n_late": len(late), "beyond_swap": int(beyond),
                     "late_dbar_minus_swapnull": [round(r["dbar50_run37"] - r["swap"]["null_mean"], 4) for r in late],
                     "pass": beyond / len(late) >= 1 / 3}
    for name, d in (("L4", L4), ("L5", L5)):
        out[name] = d
        for k, v in d.items():
            print(name, k, v)
    # R4 OLMo-1B reproduction of run55
    r55 = json.load(open(D / "run55_blind_probe_physics.json"))["models"]
    R4o = {}
    for m in ("OLMo-1B-hf", "OLMo-2-0425-1B"):
        a = load(m)
        if a is None or m not in r55:
            continue
        for ax in AXES:
            if ax not in a["batteries"]:
                continue
            new, old = a["batteries"][ax]["layers"], r55[m]["axes"][ax]["layers"]
            dmax = max(max(abs(n["dbar20_run55"] - o["linear"]["delta_orderavg"]),
                           abs(n["zscore"]["dbar20_run55"] - o["zscore"]["delta_orderavg"]),
                           abs(n["pc4"]["dbar20_run55"] - o["pc4"]["delta_orderavg"])) for n, o in zip(new, old))
            R4o[f"{m}/{ax}"] = {"max_abs_diff_vs_run55": dmax, "pass": dmax <= 0.01,
                                "linear_dbar50": [round(r["dbar50_run37"], 3) for r in new],
                                "zscore_dbar50": [round(r["zscore"]["dbar50"], 3) for r in new],
                                "pc4_dbar50": [round(r["pc4"]["dbar50"], 3) for r in new],
                                "top1_share": [round(r["spectrum"]["top1_share"], 2) for r in new],
                                "n_linear_above_null": sum(r["perm"]["p_one"] <= 0.05 for r in new), "n_layers": len(new)}
            print("R4", m, ax, R4o[f"{m}/{ax}"])
    out["R4_olmo"] = R4o
    # R8 precision
    a, b = load("pythia-410m-deduped"), load("pythia-410m-deduped_fp32cpu")
    if a and b and "world_knowledge" in b["batteries"]:
        x = a["batteries"]["world_knowledge"]["layers"]
        y = b["batteries"]["world_knowledge"]["layers"]
        agree = sum(np.sign(p["dbar50_run37"]) == np.sign(q["dbar50_run37"]) for p, q in zip(x, y))
        md = max(abs(p["dbar50_run37"] - q["dbar50_run37"]) for p, q in zip(x, y))
        out["R8"] = {"sign_agree": int(agree), "n": len(x), "max_abs_diff": md, "pass": agree / len(x) >= 0.9}
        print("R8", out["R8"])
    json.dump(out, open(D / "llm_score.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
