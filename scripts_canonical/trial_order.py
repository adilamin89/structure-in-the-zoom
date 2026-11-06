"""Trial order (checks of App H and App K): consecutive same-class rates against the rate expected at random.
Allen: the 32 sessions' drifting-grating presentations in start-time order (allen_counts npz, finite orientation), 8 direction
classes. Imaging: the 8 grating recordings in stimtimes order (asserted increasing), 8 classes of 45 degrees (the Table 9 bins).
Random expectation = sum_c p_c^2 of the class frequencies. Out: data_canonical/trial_order.json"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
COUNTS = HERE.parent / "raw" / "allen_counts"
RAW = HERE.parent / "raw"


def rates(lab):
    p = np.bincount(lab) / len(lab)
    return float(np.mean(lab[1:] == lab[:-1])), float((p ** 2).sum())


out = {"allen": [], "imaging": []}
for f in sorted(COUNTS.glob("session_*.npz")):
    d = np.load(f); o = np.argsort(d["start"]); ori = d["orientation"][o]
    r, e = rates(np.searchsorted(np.sort(np.unique(ori)), ori))
    out["allen"].append({"session": int(f.stem.split("_")[1]), "rate": r, "random": e})
for fp in sorted(RAW.glob("gratings*.npy")):
    d = np.load(fp, allow_pickle=True).item(); t = np.asarray(d["stimtimes"], float).ravel()
    assert np.all(np.diff(t) > 0)
    phi = np.asarray(d["istim"], float).ravel()
    lab = np.clip(np.digitize(phi % (2 * np.pi), np.linspace(0, 2 * np.pi, 9)) - 1, 0, 7)
    r, e = rates(lab)
    out["imaging"].append({"name": fp.stem, "rate": r, "random": e, "n": int(len(lab))})
# The same rate over each VISp population's masked presentation sequence (the rows its statistics use)
import sys  # noqa: E402
sys.path.insert(0, str(HERE))
import allen_engine as E  # noqa: E402
out["allen_visp_masked"] = []
for f in sorted(COUNTS.glob("session_*.npz")):
    S = E.load_session(int(f.stem.split("_")[1])); idx = E.populations(S).get("VISp")
    if idx is None:
        continue
    P = E.pop_view(S, "VISp", idx); st = np.load(f)["start"][P["rows"]]
    r, e = rates(P["lab"][np.argsort(st)])
    out["allen_visp_masked"].append({"session": S["sid"], "rate": r, "random": e, "n": int(len(P["rows"]))})
am = [x["rate"] for x in out["allen_visp_masked"]]
a = [x["rate"] for x in out["allen"]]; im = [x["rate"] for x in out["imaging"]]
out["summary"] = {"allen_median": float(np.median(a)), "allen_range": [float(min(a)), float(max(a))],
                  "allen_random_median": float(np.median([x["random"] for x in out["allen"]])),
                  "allen_visp_masked_median": float(np.median(am)), "allen_visp_masked_range": [float(min(am)), float(max(am))],
                  "imaging_range": [float(min(im)), float(max(im))], "imaging_random_range": [float(min(x["random"] for x in out["imaging"])), float(max(x["random"] for x in out["imaging"]))]}
print(json.dumps(out["summary"], indent=1))
for x in out["imaging"]:
    print(x["name"][9:30], "rate %.4f random %.4f" % (x["rate"], x["random"]))
json.dump(out, open(HERE.parent / "data_canonical" / "trial_order.json", "w"), indent=1)
print("ALL DONE")
