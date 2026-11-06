"""Allen's own invalid
intervals and a declared silence rule, mapped to the paper's 167 populations. Reads each session NWB in raw/allen_nwb
(fetched by allen_fetch_nwb.sh) and the stored counts (allen_counts, start times asserted equal).
  Variant A: presentation invalid for a population when it overlaps an invalid_times interval tagged with the probe of any of
             its units, or 'all_probes', or 'stimulus'; units on electrodes with valid_data False dropped.
  Variant B: A + silence rule: summed count 0 while the expected count >= 10, expected = the population's spikes outside its
             invalid intervals / that valid session time x presentation duration (session time: first to last spike in the
             session).
Out: data_canonical/allen_invalid_map.json; masks data_canonical/allen_masks/session_<sid>.npz (per area: units_keep,
validA, validB over the stored presentation order)."""
import json
import sys
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import allen_engine as E  # noqa: E402

NWB = HERE.parent / "raw" / "allen_nwb"
MASKS = HERE.parent / "data_canonical" / "allen_masks"


def dec(x):
    return x.decode() if isinstance(x, bytes) else str(x)


def one(sid):
    f = h5py.File(NWB / f"session_{sid}.nwb", "r")
    u = f["units"]
    st, sti, peak = u["spike_times"][:], u["spike_times_index"][:], u["peak_channel_id"][:]
    el = f["general/extracellular_ephys/electrodes"]
    eid = el["id"][:]
    probe_of = dict(zip(eid, [dec(x) for x in el["group_name"][:]]))
    valid_of = dict(zip(eid, el["valid_data"][:].astype(bool)))
    uprobe = np.array([probe_of[c] for c in peak])
    uvalid = np.array([valid_of[c] for c in peak])
    if "invalid_times" in f["intervals"]:
        it = f["intervals/invalid_times"]
        tg, ti = it["tags"][:], it["tags_index"][:]
        b = np.r_[0, ti]
        iv = [(float(it["start_time"][i]), float(it["stop_time"][i]), [dec(x) for x in tg[b[i]:b[i + 1]]]) for i in range(len(ti))]
    else:
        iv = []
    c = np.load(E.COUNTS / f"session_{sid}.npz")
    dg = [k for k in f["intervals"].keys() if "drifting_gratings" in k and "contrast" not in k][0]
    d = f["intervals"][dg]
    ori = np.asarray(d["orientation"][:], float)
    keep = np.isfinite(ori)
    start, stop = d["start_time"][:][keep], d["stop_time"][:][keep]
    assert np.allclose(start, c["start"]), sid
    bounds = np.r_[0, sti]
    t_first = min(st[bounds[j]] for j in range(len(sti)) if bounds[j + 1] > bounds[j])
    t_last = max(st[bounds[j + 1] - 1] for j in range(len(sti)) if bounds[j + 1] > bounds[j])
    S = E.load_session(sid)
    out = {"session": sid, "invalid_intervals": [{"start": a, "stop": z, "tags": t} for a, z, t in iv], "areas": {}}
    masks = {}
    for area, idx in E.populations(S).items():
        probes = sorted(set(uprobe[idx]))
        rel = [(a, z) for a, z, t in iv if any(p in t for p in probes) or "all_probes" in t or "stimulus" in t]
        validA = np.ones(len(start), bool)
        for a, z in rel:
            validA &= ~((start < z) & (stop > a))
        ukeep = idx[uvalid[idx]]
        # expected count: spikes of the kept units outside the population's invalid intervals, over that valid time
        nsp = 0
        for j in ukeep:
            sp = st[bounds[j]:bounds[j + 1]]
            ok = np.ones(len(sp), bool)
            for a, z in rel:
                ok &= ~((sp >= a) & (sp <= z))
            nsp += ok.sum()
        tvalid = (t_last - t_first) - sum(max(0.0, min(z, t_last) - max(a, t_first)) for a, z in rel)
        rate = nsp / tvalid
        summed = S["X"][:, ukeep].sum(1)
        expected = rate * (stop - start)
        validB = validA & ~((summed == 0) & (expected >= 10))
        masks[str(area)] = {"units_keep": ukeep, "validA": validA, "validB": validB}
        out["areas"][str(area)] = {"N": int(len(idx)), "probes": probes, "units_invalid_electrode": int((~uvalid[idx]).sum()),
                                   "n_pres": int(len(start)), "invalid_A": int((~validA).sum()), "invalid_B": int((~validB).sum()),
                                   "silent_zero": int((summed == 0).sum()), "expected_median": float(np.median(expected)),
                                   "survive_B_frac": float(validB.mean())}
    MASKS.mkdir(parents=True, exist_ok=True)
    np.savez(MASKS / f"session_{sid}.npz", **{f"{a}__{k}": v for a, m in masks.items() for k, v in m.items()})
    return out


def main():
    rows = []
    for s in json.load(open(HERE.parent / "data_canonical" / "allen_expansion_all_sessions.json"))["results"]:
        if not (NWB / f"session_{s['session']}.nwb").exists():
            print("missing NWB", s["session"], flush=True)
            continue
        r = one(s["session"])
        rows.append(r)
        bad = {a: (v["invalid_A"], v["invalid_B"], v["units_invalid_electrode"]) for a, v in r["areas"].items() if v["invalid_A"] or v["invalid_B"] or v["units_invalid_electrode"]}
        print(s["session"], "intervals", len(r["invalid_intervals"]), "| affected (A, B, bad-electrode units):", bad, flush=True)
    pops = [(r["session"], a, v) for r in rows for a, v in r["areas"].items()]
    summ = {"sessions": len(rows), "populations": len(pops),
            "pops_with_invalid_A": sum(v["invalid_A"] > 0 for _, _, v in pops), "pops_with_invalid_B": sum(v["invalid_B"] > 0 for _, _, v in pops),
            "pops_with_bad_electrode_units": sum(v["units_invalid_electrode"] > 0 for _, _, v in pops),
            "pops_below_half_B": [(s, a) for s, a, v in pops if v["survive_B_frac"] < 0.5],
            "sessions_affected": sorted({s for s, _, v in pops if v["invalid_A"] or v["invalid_B"] or v["units_invalid_electrode"]}),
            "interval_tags": sorted({" ".join(t["tags"][:1] + t["tags"][2:]) for r in rows for t in r["invalid_intervals"]})}
    print("SUMMARY", json.dumps(summ, indent=1), flush=True)
    json.dump({"rows": rows, "summary": summ},
              open(HERE.parent / "data_canonical" / "allen_invalid_map.json", "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
