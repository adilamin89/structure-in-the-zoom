"""Camera-ready App E, "The class means alone" (C7, Oct 21): the numbers of that paragraph from the stored origin sweeps.
Reads only artifacts: cr_order_effect_origins.json (within and class-mean parts of the order effect at each of the sixteen
origins), cr_failure_interrogation.json (oe_B_alone_mean: the order effect of the class-mean covariance B by itself,
origin-averaged) and multipole_harmonics_8dir.json (b2/c1). Order effect = paired minus sequential (negative: the
sequential order climbs faster).
Prints (CHECK lines): the largest |B-alone| effect; its sign against the sector (b/c above or below one); the regression of
the class-mean part minus the B-alone effect (the overlap of the class means with the fluctuations) on the within-class
part, origin means across the eight recordings. Writes nothing."""
import json
from pathlib import Path
import numpy as np

D = Path(__file__).resolve().parent.parent / "data_canonical"
oo = {r["name"]: r for r in json.load(open(D / "cr_order_effect_origins.json"))["rows"]}
fi = {r["name"]: r for r in json.load(open(D / "cr_failure_interrogation.json"))["rows"]}
bc = {r["name"]: r["b2_over_c1"] for r in json.load(open(D / "multipole_harmonics_8dir.json"))["rows"] if "b2_over_c1" in r}
names = sorted(fi)
w = np.array([np.mean([o["within"] for o in oo[n]["origins"]]) for n in names])
m = np.array([np.mean([o["class_means"] for o in oo[n]["origins"]]) for n in names])
b = np.array([fi[n]["oe_B_alone_mean"] for n in names])
even = np.array([bc[n] > 1 for n in names])
for n, wi, mi, bi, e in zip(names, w, m, b, even):
    print(f"{n[:30]:30s} within {wi:+.4f} class-mean {mi:+.4f} B alone {bi:+.4f} b/c {'>' if e else '<'} 1")
print("CHECK max |B alone| = %.4f" % np.abs(b).max())
print("CHECK B alone favours sequential in %d of %d even-led, paired in %d of %d dipole-led"
      % ((b[even] < 0).sum(), even.sum(), (b[~even] > 0).sum(), (~even).sum()))
s, _ = np.polyfit(w, m - b, 1)
print("CHECK (class-mean part - B alone) on within part: slope %.3f, r %.4f" % (s, np.corrcoef(w, m - b)[0, 1]))
