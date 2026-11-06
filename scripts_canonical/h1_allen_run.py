"""H1 Allen driver (registrations H1-R1..R20): every session, every population of the paper (VIS areas >= 50 units, all
units), plus the quality-filtered VISp arm (isi_violations < 0.5, amplitude_cutoff < 0.1, presence_ratio > 0.9; >= 30
units). One JSON per session in data_canonical/h1_allen/ (kill-safe, skipped when present). Process pool (spawn), 4 workers.
Run: VECLIB_MAXIMUM_THREADS=1 python -X faulthandler h1_allen_run.py [sid ...]"""
import json
import sys
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "data_canonical" / "h1_allen"
STORED = HERE.parent / "data_canonical" / "allen_expansion_all_sessions.json"
QUALITY_MIN = 30
N_PERM = 200


def run_session(sid):
    import numpy as np
    import h1_allen_engine as E
    fp = OUT / f"session_{sid}.json"
    if fp.exists():
        return sid, "skip"
    t0 = time.time()
    S = E.load_session(sid)
    stored = {s["session"]: s for s in json.load(open(STORED))["results"]}[sid]["areas"]
    rep = E.reproduce_stored(S)
    assert set(rep) == set(stored), (set(rep), set(stored))
    out = {"session": sid, "reproduce_stored": {a: {"recomputed": rep[a], "stored": stored[a]["delta_dir8"]} for a in rep},
           "areas": {}, "quality_VISp": None}
    for a, idx in E.populations(S).items():
        assert abs(rep[a] - stored[a]["delta_dir8"]) < 1e-6, (sid, a, rep[a], stored[a]["delta_dir8"])
        P = E.pop_view(S, a, idx)   # H5-R1 masking path
        out["areas"][a] = E.analyze_population(P["X"], P["lab"], P["tf"], full=True, n_perm=N_PERM)
        print(f"[{sid}] {a:6s} N {len(idx):3d} pooled {out['areas'][a]['pooled']['delta']:+.3f} tfres "
              f"{out['areas'][a]['tfres']['delta']:+.3f} ({time.time() - t0:.0f}s)", flush=True)
    q = E.populations(S, quality=True, min_units=QUALITY_MIN).get("VISp")
    if q is not None:
        P = E.pop_view(S, "VISp", q)
        out["quality_VISp"] = E.analyze_population(P["X"], P["lab"], P["tf"], full=True, n_perm=N_PERM)
    fp.write_text(json.dumps(out, default=float))
    print(f"[{sid}] DONE {len(out['areas'])} areas ({time.time() - t0:.0f}s)", flush=True)
    return sid, "ok"


def main():
    OUT.mkdir(exist_ok=True)
    sids = [int(x) for x in sys.argv[1:]] or [s["session"] for s in json.load(open(STORED))["results"]]
    sys.path.insert(0, str(HERE))
    with ProcessPoolExecutor(4, mp_context=mp.get_context("spawn")) as ex:
        for sid, st in ex.map(run_session, sids):
            print("session", sid, st, flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    sys.path.insert(0, str(HERE))
    main()
