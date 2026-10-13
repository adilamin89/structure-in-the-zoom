"""Camera-ready Table 1 (datasets): every count read from the released artifacts, none typed by hand.
Inputs: basin_memory/data/stringer_theta_decomposition.json (all 18 Stringer recordings: neurons, samples),
data_canonical/multipole_harmonics_8dir.json (per-class presentation counts on the 8-class direction binning),
data_canonical/allen_expansion_all_sessions.json (32 Allen sessions: presentations, units per area).
Out: data_canonical/data_table_camera_ready.json and the LaTeX rows on stdout."""
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data_canonical"
BM = HERE.parent.parent.parent / "basin_memory" / "data"


def bm(name, local):
    """Raw-analysis file from basin_memory/data in the working tree, else its copy in data_canonical (public release)."""
    p = BM / name
    return p if p.exists() else DATA / local


def rng(v):
    return [min(v), max(v)]


def mouse(name):
    m = re.search(r"(GT\d|TX\d+|MP\d+)", name)
    return m.group(1)


st = json.load(open(bm("stringer_theta_decomposition.json", "theta_decomposition.json")))["rows"]
mp = {r["name"]: r for r in json.load(open(DATA / "multipole_harmonics_8dir.json"))["rows"]}
groups = {"gratings": [r for r in st if r["name"].startswith("gratings")],
          "static": [r for r in st if r["name"].startswith("static")],
          "spont": [r for r in st if r["name"].startswith("spont")]}
out = {"stringer": {}}
for g, rows in groups.items():
    rec = {"n_recordings": len(rows), "mice": sorted({mouse(r["name"]) for r in rows}),
           "neurons": rng([r["n_neurons"] for r in rows]), "samples": rng([r["n_stimuli"] for r in rows])}
    if g == "gratings":
        counts = [c for r in rows for c in mp[r["name"]]["bin_counts"]]
        rec["per_class"] = rng(counts)
        rec["by_type"] = {t: sum(r["name"].startswith(f"gratings_{t}") for r in rows)
                          for t in ("drifting", "local", "low_contrast")}
    out["stringer"][g] = rec
out["stringer"]["all"] = {"n_recordings": len(st), "neurons": rng([r["n_neurons"] for r in st]),
                          "mice": len({mouse(r["name"]) for r in st})}

al = json.load(open(DATA / "allen_expansion_all_sessions.json"))["results"]
ok = [s for s in al if s.get("status") == "ok"]
units = [a["n_units"] for s in ok for a in s["areas"].values()]
visp = [s["areas"]["VISp"]["n_units"] for s in ok if "VISp" in s["areas"]]
out["allen"] = {"n_sessions": len(ok), "n_populations": len(units),
                "areas": sorted({k for s in ok for k in s["areas"]}),
                "presentations": rng([s["n_presentations"] for s in ok]),
                "units_per_population": rng(units), "visp_units": rng(visp), "n_visp": len(visp)}
json.dump(out, open(DATA / "data_table_camera_ready.json", "w"), indent=1)
print(json.dumps(out, indent=1))
