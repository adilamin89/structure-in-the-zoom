"""C13 (Oct 24): what decoding accuracy and the class-mean part share across the eight grating recordings.

The Discussion reports that direction decoding (256 neurons, five-fold logistic; h3_v1_decode.json) ranks the recordings as the
class-mean part of the shift does (Table 9; Spearman -0.98). Eq. 3 says the class-mean part is set by the class means' strength
against the fluctuations, beta = Tr B / Tr W, their dimensions and their alignment a. This script asks whether beta is the shared
variable: it reads beta at the decoder's size (N = 256, mean over the ten random neuron draws of cr_h1_b1_neuron_count.json, raw
frame) and at all neurons, and prints its rank correlation with decoding accuracy and with the class-mean part.

Inputs (data_canonical/): h3_v1_decode.json (decode['256']['mean']), cr_h1_b1_neuron_count.json (rows[N].raw.trB_trW),
cr_withinclass_controls.json (orig.class_means = Table 9). Output: data_canonical/c13_decoding_beta.json.
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

D = Path(__file__).resolve().parent.parent / "data_canonical"
dec = {r["name"]: r["decode"]["256"]["mean"] for r in json.loads((D / "h3_v1_decode.json").read_text())["rows"]}
cm = {r["name"]: r["orig"]["class_means"] for r in json.loads((D / "cr_withinclass_controls.json").read_text())["rows"]}
recs = json.loads((D / "cr_h1_b1_neuron_count.json").read_text())["recordings"]

names = [r["name"] for r in recs]
assert set(names) == set(dec) == set(cm), "the three artifacts must cover the same eight recordings"


def beta_at(rec, n):
    rows = [x for x in rec["rows"] if x["N"] == n]
    assert rows, (rec["name"], n)
    return float(np.mean([x["raw"]["trB_trW"] for x in rows])), len(rows)


acc = np.array([dec[n] for n in names])
cmp_ = np.array([cm[n] for n in names])
out = {"recordings": names, "decode_acc_256": acc.tolist(), "class_mean_part": cmp_.tolist(), "by_size": {}}
for label, size_of in (("256", lambda r: 256), ("all", lambda r: r["n_neurons"])):
    vals = [beta_at(r, size_of(r)) for r in recs]
    beta = np.array([v for v, _ in vals])
    res = {
        "beta": beta.tolist(), "draws": [k for _, k in vals],
        "spearman_acc_beta": float(spearmanr(acc, beta)[0]), "pearson_acc_beta": float(pearsonr(acc, beta)[0]),
        "spearman_classmean_beta": float(spearmanr(cmp_, beta)[0]),
    }
    out["by_size"][label] = res
    print(f"N = {label}: beta {np.round(beta, 3).tolist()}  draws {res['draws']}")
    print(f"  Spearman(acc, beta) {res['spearman_acc_beta']:+.3f}  Pearson {res['pearson_acc_beta']:+.3f}  "
          f"Spearman(class-mean part, beta) {res['spearman_classmean_beta']:+.3f}")
out["spearman_acc_classmean"] = float(spearmanr(acc, cmp_)[0])
print(f"Spearman(acc, class-mean part) {out['spearman_acc_classmean']:+.3f}")
(D / "c13_decoding_beta.json").write_text(json.dumps(out, indent=1))
print("wrote", D / "c13_decoding_beta.json")
