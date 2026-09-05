"""Figure (appendix): the calibrated-model bracket of GT3 and the cross-recording
refit (S77; panel (b) of the S70 fig_mechanism.png, with its inset drawn as a
second panel). Sources: run2b_corotating_seeds.json, run11b_fresh_draw_prediction.json,
run48_overshoot_across_recordings.json.
Out: ../figures_canonical/fig_model_bracket.png + ../../arxiv/figures/
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 6.8, "savefig.dpi": 300})
HERE = Path(__file__).resolve().parent; DATA = HERE.parent / "data_canonical"
OUTS = [HERE.parent / "figures_canonical" / "fig_model_bracket.png", HERE.parent.parent / "arxiv" / "figures" / "fig_model_bracket.png"]
j = lambda f: json.load(open(DATA / f))
run2b = j("run2b_corotating_seeds.json"); run11b = j("run11b_fresh_draw_prediction.json"); run48 = j("run48_overshoot_across_recordings.json")
fig, axes = plt.subplots(1, 2, figsize=(5.0, 2.5), gridspec_kw={"width_ratios": [1.1, 1.0]})
ax = axes[0]
bars = [("shared\nmodes", run2b["shared"]["mean"], run2b["shared"]["sd"], "#9ecae1"),
        ("class-dep.\nmodes", run2b["corotating"]["mean"], run2b["corotating"]["sd"], "#4292c6"),
        ("alignment-\ncalibrated", run11b["boot_mean"], (run11b["boot_ci95"][1] - run11b["boot_ci95"][0]) / 2, "#08519c")]
for i, (lab, mu, err, c) in enumerate(bars):
    ax.bar(i, mu, 0.62, color=c, yerr=err, capsize=3, error_kw={"lw": 1.0})
ax.axhline(0.237, color="#c23b3b", lw=1.6); ax.text(-0.42, 0.226, "GT3 observed $+0.237$", color="#c23b3b", fontsize=6.8, ha="left", va="top")
ax.set_xticks(range(3)); ax.set_xticklabels([b[0] for b in bars], fontsize=6.5); ax.set_ylabel(r"model $\delta$"); ax.set_ylim(0, 0.56)
ax.set_title("(a) two variants bracket GT3"); ax.spines[["top", "right"]].set_visible(False)
ax = axes[1]
for key, mk in (("GT1", "s"), ("GT2", "^"), ("GT3", "o")):
    r = run48["rows"][key]; ax.plot(r["observed_delta"], r["delta_pred"], mk, ms=6, color="#08519c")
    ax.text(r["observed_delta"] + 0.008, r["delta_pred"], key, fontsize=7, va="center")
ax.plot([0.15, 0.42], [0.15, 0.42], "-", color="0.5", lw=0.7)
ax.set_xlim(0.15, 0.44); ax.set_ylim(0.15, 0.44); ax.set_xlabel("observed $\\delta$"); ax.set_ylabel("predicted $\\delta$")
ax.set_title("(b) refit to three recordings"); ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(w_pad=1.5)
for out in OUTS:
    if out is not OUTS[0] and not out.parent.exists(): continue
    out.parent.mkdir(parents=True, exist_ok=True); fig.savefig(out, dpi=300); print("wrote", out)
