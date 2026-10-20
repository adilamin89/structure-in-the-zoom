"""Camera-ready App G, second table: which sector survives graining, eight grating recordings, four block types.

Reads data_canonical/run51_spatial_blocking.json (run51_spatial_blocking.py: K = 1, 4, 16, 32; blocks sorted by preferred
orientation (2nd-harmonic phase), by preferred direction (1st-harmonic phase), at random, or anatomical = k-means on
imaging-plane centroids; direction-aligned shift with the paper's estimator, two floor seeds averaged; class-mean balance
b2/c1 from the exact 8-point profile). Prints the LaTeX rows and every range the App G text quotes (CHECK lines).
No new computation: it formats the registered run's output."""
import json
from pathlib import Path

D = json.load(open(Path(__file__).resolve().parent.parent / "data_canonical" / "run51_spatial_blocking.json"))
TAG = {"drifting": "Drifting", "local": "Local", "low_contrast": "Low-contrast"}
ARMS = ["ori_sorted", "dir_sorted", "random", "spatial"]
rows = [r for r in D["rows"] if r["status"] == "ok"]


def label(name):
    kind = next(k for k in ("low_contrast", "drifting", "local") if k in name)
    return f"{TAG[kind]} GT{name.split('_GT')[1][0]}"


for r in rows:
    ret, flow = r["retention_K32"], r["balance_flow_K32"]
    cells = " & ".join(f"${ret[a]:.2f}$" for a in ARMS)
    flows = " & ".join(f"${flow[a]:.2f}$" for a in ("ori_sorted", "dir_sorted", "spatial"))
    print(f"{label(r['name'])} & ${r['graining']['random'][0]:+.3f}$ & {cells} & {flows} \\\\")

allv = [v for r in rows for a in ARMS for v in r["graining"][a]]
print("CHECK cells", len(allv), "all positive", all(v > 0 for v in allv), "min %.3f" % min(allv))
even = [r for r in rows if any(t in r["name"] for t in ("drifting_GT1", "drifting_GT3", "low_contrast"))]
odd = [r for r in rows if r not in even]
rng = lambda rs, a, key="retention_K32": (min(x[key][a] for x in rs), max(x[key][a] for x in rs))
print("CHECK ori retention even-led %.2f-%.2f, others %.2f-%.2f" % (*rng(even, "ori_sorted"), *rng(odd, "ori_sorted")))
print("CHECK dir retention highest of four in", sum(r["retention_K32"]["dir_sorted"] >= max(r["retention_K32"][a] for a in ARMS) for r in rows), "of", len(rows))
print("CHECK balance flow ori %.2f-%.2f dir %.2f-%.2f spatial %.2f-%.2f random %.2f-%.2f" % (
    *rng(rows, "ori_sorted", "balance_flow_K32"), *rng(rows, "dir_sorted", "balance_flow_K32"),
    *rng(rows, "spatial", "balance_flow_K32"), *rng(rows, "random", "balance_flow_K32")))
gap = max(abs(r["graining"]["spatial"][k] - r["graining"]["random"][k]) for r in rows for k in range(4))
gap32 = max(abs(r["graining"]["spatial"][-1] - r["graining"]["random"][-1]) for r in rows)
print("CHECK |spatial - random| shift: max over all K %.3f, at K = 32 %.3f" % (gap, gap32))
gap16 = max(abs(r["graining"]["spatial"][k] - r["graining"]["random"][k]) for r in rows for k in (2, 3))
signs4 = [r["graining"]["spatial"][1] - r["graining"]["random"][1] for r in rows]
print("CHECK |spatial - random| at K = 16 and 32: max %.3f; at K = 4 spatial-random from %+.3f to %+.3f" % (gap16, min(signs4), max(signs4)))
