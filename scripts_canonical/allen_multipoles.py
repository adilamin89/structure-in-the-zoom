"""Allen multipole map: harmonic content of the code per visual area.

Reads the session NWBs already persisted on the Volume (no re-download) and
computes, per session x VIS area (>=50 units), the class-mean signal
correlation C(dphi) over the 8 drifting-grating directions and its harmonic
fit C = a + c cos(dphi) + b cos(2 dphi) + b4 cos(4 dphi):
  - quadrupole b, dipole c, b4, the inter-sector ratio J1/J2 = c/b
  - cardinal fraction of tuning power (broken-rotation anisotropy)
This is the spin-orbit (l=1/l=2) coupling measured across the cortical
hierarchy; feeds the 9pp discussion and Paper C.

Run: modal run modal_allen_multipoles.py
"""
import modal

image = modal.Image.debian_slim(python_version="3.11").pip_install("numpy", "h5py")
app = modal.App("allen-multipoles")
vol = modal.Volume.from_name("allen-neuropixels-data")

MIN_UNITS = 50


@app.function(image=image, volumes={"/data": vol}, timeout=3600, cpu=4, memory=32768)
def session_multipoles(sid: int):
    import json
    import os

    import h5py
    import numpy as np

    nwb = f"/data/nwb/session_{sid}.nwb"
    if not os.path.exists(nwb):
        return {"session": sid, "status": "no_nwb"}
    f = h5py.File(nwb, "r")
    units = f["units"]
    st = units["spike_times"][:]
    sti = units["spike_times_index"][:]
    peak_ch = units["peak_channel_id"][:]
    el = f["general/extracellular_ephys/electrodes"]
    ch_to_loc = dict(zip(el["id"][:],
                         [x.decode() if isinstance(x, bytes) else str(x)
                          for x in el["location"][:]]))
    unit_loc = np.array([ch_to_loc.get(c, "") for c in peak_ch])

    dgn = [k for k in f["intervals"].keys() if "drifting_gratings" in k
           and "contrast" not in k][0]
    dg = f["intervals"][dgn]
    start, stop = dg["start_time"][:], dg["stop_time"][:]
    ori = np.array([float(x) if not isinstance(x, bytes) else
                    (np.nan if x in (b"null", b"") else float(x))
                    for x in dg["orientation"][:]])
    keep = np.isfinite(ori)
    start, stop, ori = start[keep], stop[keep], ori[keep]
    dirs = np.sort(np.unique(ori))

    bounds = np.concatenate([[0], sti])
    counts = np.zeros((len(ori), len(sti)), dtype=np.float32)
    for j in range(len(sti)):
        sp = np.sort(st[bounds[j]:bounds[j + 1]])
        counts[:, j] = np.searchsorted(sp, stop) - np.searchsorted(sp, start)

    def fit_multipoles(C, dphi):
        A = np.vstack([np.ones_like(dphi), np.cos(dphi), np.cos(2 * dphi),
                       np.cos(4 * dphi)]).T
        coef, *_ = np.linalg.lstsq(A, C, rcond=None)
        resid = C - A @ coef
        r2 = 1 - resid.var() / C.var() if C.var() > 0 else None
        return coef, r2

    res = {"session": sid, "status": "ok", "areas": {}}
    for area in sorted(set(unit_loc)):
        if not area.startswith("VIS"):
            continue
        idx = np.where(unit_loc == area)[0]
        if len(idx) < MIN_UNITS:
            continue
        X = counts[:, idx]
        M = np.stack([X[ori == d].mean(axis=0) for d in dirs])  # 8 x units
        Mc = M - M.mean(axis=1, keepdims=True)
        Mn = Mc / (np.linalg.norm(Mc, axis=1, keepdims=True) + 1e-9)
        Cfull = Mn @ Mn.T
        # C as a function of direction difference (average over pairs)
        dmat = np.abs(np.radians(dirs)[:, None] - np.radians(dirs)[None, :])
        dmat = np.minimum(dmat, 2 * np.pi - dmat)
        pairs = {}
        for i in range(8):
            for j in range(8):
                key = round(float(dmat[i, j]), 6)
                pairs.setdefault(key, []).append(float(Cfull[i, j]))
        dphi = np.array(sorted(pairs))
        Cbar = np.array([np.mean(pairs[k]) for k in sorted(pairs)])
        (a0, c1, b2, b4), r2 = fit_multipoles(Cbar, dphi)
        # cardinal anisotropy: tuning power at 0/90/180/270 vs oblique
        power = (Mc ** 2).sum(axis=1)
        card = float(power[np.isin(dirs, [0, 90, 180, 270])].sum() / power.sum())
        res["areas"][area] = {"n_units": int(len(idx)),
                              "a": float(a0), "c_dipole": float(c1),
                              "b_quadrupole": float(b2), "b4": float(b4),
                              "J1_over_J2": float(c1 / b2) if abs(b2) > 1e-6 else None,
                              "fit_r2": r2 if r2 is None else float(r2),
                              "cardinal_fraction": card,
                              "C_dphi": {str(round(float(k), 3)): float(np.mean(v))
                                         for k, v in pairs.items()}}
        print(f"[{sid}] {area}: b={b2:+.3f} c={c1:+.3f} J1/J2="
              f"{res['areas'][area]['J1_over_J2']} card={card:.2f}", flush=True)
    f.close()
    os.makedirs("/data/multipoles", exist_ok=True)
    with open(f"/data/multipoles/session_{sid}.json", "w") as fo:
        json.dump(res, fo, indent=1)
    vol.commit()
    return res


@app.local_entrypoint()
def main():
    import csv
    import io
    import json
    import urllib.request

    url = ("https://allen-brain-observatory.s3.us-west-2.amazonaws.com/"
           "visual-coding-neuropixels/ecephys-cache/sessions.csv")
    rows = list(csv.DictReader(io.StringIO(
        urllib.request.urlopen(url).read().decode())))
    sids = [int(r["id"]) for r in rows if r["session_type"] == "brain_observatory_1.1"]
    results = list(session_multipoles.map(sids, return_exceptions=True))
    ok = [r for r in results if isinstance(r, dict) and r.get("status") == "ok"]
    with open("data/allen_multipoles_all_sessions.json", "w") as f:
        json.dump({"results": [r if isinstance(r, dict) else {"error": str(r)}
                               for r in results]}, f, indent=1)
    import statistics as stt
    from collections import defaultdict
    per = defaultdict(lambda: defaultdict(list))
    for r in ok:
        for area, a in r["areas"].items():
            for k in ("b_quadrupole", "c_dipole", "J1_over_J2", "cardinal_fraction"):
                if a.get(k) is not None:
                    per[area][k].append(a[k])
    print(f"ok sessions: {len(ok)}")
    for area in sorted(per, key=lambda a: -len(per[a]["b_quadrupole"])):
        d = per[area]
        if len(d["b_quadrupole"]) < 10:
            continue
        print(f"{area}: n={len(d['b_quadrupole'])} b={stt.mean(d['b_quadrupole']):+.3f} "
              f"c={stt.mean(d['c_dipole']):+.3f} J1/J2={stt.mean(d['J1_over_J2']):+.3f} "
              f"card={stt.mean(d['cardinal_fraction']):.3f}")
    print("Saved data/allen_multipoles_all_sessions.json")
