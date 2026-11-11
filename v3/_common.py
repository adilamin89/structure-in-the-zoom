"""Shared paths for the extended version's generators (v3/): the repository root, its rung, its raw/ data and the topic axis."""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RAW = ROOT / "raw"
AXIS = ROOT / "axes" / "world_knowledge32.json"
STATES = RAW / "llm_states" / "pythia-2.8b-deduped__main__topics.npz"
sys.path.insert(0, str(ROOT))
import rung  # noqa: E402,F401

GRATINGS = {"Drifting GT1": "gratings_drifting_GT1_2019_04_12_1", "Drifting GT2": "gratings_drifting_GT2_2019_04_05_1",
            "Drifting GT3": "gratings_drifting_GT3_2019_04_05_1", "Local GT1": "gratings_local_GT1_2019_04_27_2",
            "Local GT2": "gratings_local_GT2_2019_04_23_2", "Local GT3": "gratings_local_GT3_2019_04_24_2",
            "Low-contrast GT1": "gratings_low_contrast_GT1_2019_04_09_1", "Low-contrast GT2": "gratings_low_contrast_GT2_2019_04_12_2"}


def states(path=None):
    """Pythia-2.8B-deduped's last-token states on the topic axis (33 layers x 256 prompts x 2560 units), as
    scripts_canonical/lm_shape_surrogate.py --save-states writes them; asserted to hold the released axis's prompts."""
    import json
    p = Path(path) if path else STATES
    if not p.exists():
        raise SystemExit(f"{p} not found: write it with python scripts_canonical/lm_shape_surrogate.py --device mps "
                         f"--save-states {STATES.relative_to(ROOT)} (needs the model extras and the 5.3 GB model)")
    z = np.load(p, allow_pickle=True)
    H, lab, prompts, names = z["H"], z["labels"], z["prompts"], list(map(str, z["names"]))
    axis = json.load(open(AXIS))
    assert list(map(str, prompts)) == [q for c in names for q in axis[c]], "states must hold the released axis's prompts"
    assert np.bincount(lab).tolist() == [32] * 8 and H.shape[:2] == (33, 256)
    return H, lab, p.name, (str(z["dtype"]) if "dtype" in z else None), (str(z["device"]) if "device" in z else None)


def zframe(X):
    """Units scaled to unit variance; units with (near) zero variance dropped."""
    sd = X.std(0)
    keep = sd > 1e-6 * np.median(sd)
    return (X[:, keep] - X[:, keep].mean(0)) / sd[keep]
