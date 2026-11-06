"""H1-R22b (Oct 22): B2's two-term prediction of delta for every Allen population (tfres arm), from the counts alone, written
and committed BEFORE comparison with the measured tfres delta of h1_allen_run.py. Within part: mean field with the split-half
noise sigma (h1_allen_engine.sigma_split, cells = direction x TF); class-mean part: cr_h1_b245.classmean_prediction on the
TF-residualized input. Nothing is fitted. Out: data_canonical/h1_allen_b2_predictions.json"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import h1_allen_engine as E  # noqa: E402

spec = importlib.util.spec_from_file_location("b245", HERE / "cr_h1_b245.py"); B = importlib.util.module_from_spec(spec); spec.loader.exec_module(B)
sids = [s["session"] for s in json.load(open(HERE.parent / "data_canonical" / "allen_expansion_all_sessions.json"))["results"]]
out = []
for sid in sids:
    S = E.load_session(sid)
    for a, idx in E.populations(S).items():
        P = E.pop_view(S, a, idx)   # H5-R1 masking path
        X, lab = P["X"], P["lab"]
        Y = E.tf_residualize(X, lab, P["cell"])
        sig, _ = E.sigma_split(X, lab, P["cell"], "noise")
        sizes = [int((lab < k).sum()) for k in B.RUNGS]
        cmp_, c_full, us = B.classmean_prediction(B.full_terms(Y, lab), sizes)
        wp = E.mean_field(sig)
        out.append({"session": sid, "area": a, "N": int(X.shape[1]), "sigma_noise": sig, "within_pred": wp, "classmean_pred": cmp_,
                    "delta_pred": wp + cmp_, "c_full": c_full})
    print(sid, len(out), flush=True)
json.dump(out, open(HERE.parent / "data_canonical" / "h1_allen_b2_predictions.json", "w"), indent=1)
print("wrote h1_allen_b2_predictions.json DONE", flush=True)
