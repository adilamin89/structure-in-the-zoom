"""LLM analogue of the V1 sector-balance results (S74, 2026-09-02): additivity of
the class-mean kernel's harmonic content over hidden units, its flow under unit
blocking, and a context-length knob. Pythia-160m, last-token hidden states,
the paper's own prompts (planted compass / clock C8 axes from run28; the
world-knowledge and construction-type axes from axes/).

MEASURES (per layer):
  Kernel profile: correlation matrix of class means centred over classes
  (as in run28), circular profile f(D) over cyclic class distance D, DFT
  harmonics h1..h4 (h4 = Nyquist). "Parity power" = h4^2, "cycle power" =
  h1^2 (the compass kernel was h4-dominated = parity quotient; the clock was
  not circulant).
  A. ADDITIVITY: h_k computed from the profile vs from the sum over units of
     per-unit tuning harmonic power (unit-standardised) - must agree to
     numerical precision, as in V1.
  B. UNIT BLOCKING: average K hidden units per block, K in 1..64, with blocks
     (i) random, (ii) sorted by each unit's preferred phase on the dominant
     harmonic (h4 phase for compass, h1 phase for clock); recompute the
     profile harmonics. Expectation from V1: random invariant; sorted blocking
     amplifies the sorted harmonic's share.
  C. CONTEXT LENGTH: prepend a fixed 38-word neutral preamble to every prompt
     ("long") vs the paper's prompts ("short"); compare kernel effective rank
     and harmonic shares on compass/clock, and the declared-path delta profile
     (theta_zoom.zoom, n_perm=100, declared order, k_orders=20) on the
     world-knowledge and construction axes: does the crossover depth move?
EXPECTATIONS: A is an identity (exact) once the per-unit curves are formed from the
class-centred, unit-standardised means that define the kernel (first two runs compared a squared
coefficient ratio with a power ratio and a raw-row standardisation; both corrected 2026-09-02:
the identity is coefficient ratio = unit power ratio). B and C are exploratory; the V1
pattern (random-invariant, sorted-amplified) is the registered expectation for
B; no direction is registered for C.

Out: ../data_canonical/llm_sector_blocking.json
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data_canonical"
AXES = HERE.parent.parent / "arxiv_supplement" / "axes"
sys.path.insert(0, str(HERE.parent.parent / "arxiv_supplement"))
from theta_zoom import zoom  # noqa: E402

spec = importlib.util.spec_from_file_location("r17", HERE / "run17_multiclass_battery.py")
r17 = importlib.util.module_from_spec(spec); spec.loader.exec_module(r17)
spec28 = importlib.util.spec_from_file_location("r28", HERE / "run28_cyclic_axis_llm.py")
r28 = importlib.util.module_from_spec(spec28); spec28.loader.exec_module(r28)

MODEL = "EleutherAI/pythia-160m"
NB = 8
KS = [1, 2, 4, 8, 16, 32, 64]
PREAMBLE = ("The following note was written on an ordinary weekday afternoon, after a "
            "quiet morning of routine errands in town, with the weather mild and the "
            "streets calm, and it records one plain observation without comment: ")


def prompts_for(axis):
    if axis == "compass":
        return {d: [t.format(d=d) for t in r28.COMPASS_TEMPLATES] for d in r28.COMPASS}
    if axis == "clock":
        return {t_: [tpl.format(t=t_) for tpl in r28.CLOCK_TEMPLATES] for t_ in r28.CLOCK}
    d = json.load(open(AXES / f"{axis}.json"))
    return {k: v[:16] for k, v in d.items()}


def circ_profile_harmonics(M):
    """M: classes x units (centred over classes). Kernel = corr of class means; profile over cyclic distance."""
    Cm = np.corrcoef(M)
    prof = np.array([np.mean([Cm[i, (i + k) % NB] for i in range(NB)]) for k in range(NB)])
    ang = 2 * np.pi * np.arange(NB) / NB
    h = {k: float(2 / NB * np.sum(prof * np.cos(k * ang))) for k in (1, 2, 3, 4)}   # uniform 2/NB so ratios match the unit power ratios
    return prof, h


def unit_harmonics(M):
    """Per-unit harmonic power summed over units = DFT of the kernel's circular profile (identity).
    The kernel is the correlation of CLASS-CENTRED class means (run28 definition), so the rows are
    centred over classes first and then standardised over units; the k >= 1 harmonics of the summed
    unit autocorrelations equal the profile harmonics exactly."""
    Mc = M - M.mean(0)
    Ms = (Mc - Mc.mean(1, keepdims=True)) / (Mc.std(1, keepdims=True) + 1e-9)
    ang = 2 * np.pi * np.arange(NB) / NB
    P = {k: float(np.sum(np.abs((Ms * np.exp(1j * k * ang[:, None])).sum(0)) ** 2)) for k in (1, 2, 3, 4)}
    return P


def eff_rank(M):
    Mc = M - M.mean(0)
    K = np.corrcoef(Mc); w = np.linalg.eigvalsh(K); w = np.clip(w, 0, None)
    return float(w.sum() ** 2 / (w ** 2).sum())


def blocking_flow(M, sort_h):
    """M: classes x units. Sort units by preferred phase of harmonic sort_h; block-average K units."""
    ang = 2 * np.pi * np.arange(NB) / NB
    Mc = M - M.mean(0)
    z = (Mc * np.exp(1j * sort_h * ang[:, None])).sum(0)
    phase = np.angle(z) % (2 * np.pi)
    rng = np.random.default_rng(0)
    out = {"K": KS, "sorted": [], "random": []}
    for K in KS:
        nblk = M.shape[1] // K
        for key, order in (("sorted", np.argsort(phase)), ("random", rng.permutation(M.shape[1]))):
            Mb = np.stack([M[:, order[i * K:(i + 1) * K]].mean(1) for i in range(nblk)], 1)
            _, h = circ_profile_harmonics(Mb)
            tot = sum(v ** 2 for v in h.values()) + 1e-12
            out[key].append({"share_h%d" % k: h[k] ** 2 / tot for k in (1, 2, 3, 4)})
    return out


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).to(device)
    results = {}
    for axis in ("compass", "clock", "world_knowledge", "language_type"):
        pr = prompts_for(axis)
        classes = list(pr.keys())
        labels = np.array([i for i, c in enumerate(classes) for _ in pr[c]])
        for cond in ("short", "long"):
            prompts = [p for c in classes for p in pr[c]]
            if cond == "long":
                prompts = [PREAMBLE + p[0].lower() + p[1:] for p in prompts]
            H = r17.get_hidden_states(model, tok, prompts, device=device)   # layers x prompts x d
            H = np.asarray(H)
            per_layer = []
            for L in range(H.shape[0]):
                X = H[L]
                M = np.stack([X[labels == i].mean(0) for i in range(NB)])
                row = {"layer": L, "eff_rank": eff_rank(M)}
                if axis in ("compass", "clock"):
                    prof, h = circ_profile_harmonics(M - M.mean(0))
                    P = unit_harmonics(M)
                    tot = sum(v ** 2 for v in h.values()) + 1e-12
                    row.update({"profile": prof[:5].tolist(), "h": h,
                                "share": {"h%d" % k: h[k] ** 2 / tot for k in (1, 2, 3, 4)},
                                # identity: profile coefficient ratio h_k/h_1 == unit power ratio P_k/P_1
                                "additivity_h4_over_h1_profile": (h[4] / (h[1] + 1e-12)),
                                "additivity_h4_over_h1_units": (P[4] / (P[1] + 1e-12)),
                                "additivity_h2_over_h1_profile": (h[2] / (h[1] + 1e-12)),
                                "additivity_h2_over_h1_units": (P[2] / (P[1] + 1e-12))})
                    if cond == "short" and L in (3, 6, 9, 12):
                        row["blocking"] = blocking_flow(M, 4 if axis == "compass" else 1)
                else:
                    z = zoom(X, labels, n_perm=100, k_orders=20, k_null_orders=5)
                    row.update({"delta": float(z["delta"]), "delta_orderavg": float(z["delta_orderavg"]),
                                "p_two": float(z["p_two"])})
                per_layer.append(row)
            results[f"{axis}/{cond}"] = per_layer
            if axis in ("compass", "clock"):
                mid = per_layer[6]
                print(f"  {axis:15s} {cond:5s} L6: eff_rank {mid['eff_rank']:.2f} shares h1 {mid['share']['h1']:.2f} h2 {mid['share']['h2']:.2f} "
                      f"h4 {mid['share']['h4']:.2f} | additivity h4/h1 profile {mid['additivity_h4_over_h1_profile']:.3f} units {mid['additivity_h4_over_h1_units']:.3f}")
            else:
                d = [r["delta"] for r in per_layer]
                cross = next((i for i in range(1, len(d)) if d[i - 1] < 0 <= d[i]), None)
                print(f"  {axis:15s} {cond:5s} delta emb {d[0]:+.3f} final {d[-1]:+.3f} min {min(d):+.3f} first zero-crossing layer {cross} (of {len(d)-1})")
    # verdicts
    def maxdev(axis):
        rows = results[f"{axis}/short"]
        return max(abs(r["additivity_h4_over_h1_profile"] - r["additivity_h4_over_h1_units"]) / (r["additivity_h4_over_h1_units"] + 1e-9) for r in rows[1:])
    verdict = {"A_additivity_max_rel_dev_compass": maxdev("compass"), "A_additivity_max_rel_dev_clock": maxdev("clock")}
    for axis, hk in (("compass", "h4"), ("clock", "h1")):
        rows = [r for r in results[f"{axis}/short"] if "blocking" in r]
        verdict[f"B_{axis}_sorted_{hk}_share_K1_to_K64"] = [(r["layer"], round(r["blocking"]["sorted"][0]["share_" + hk], 3), round(r["blocking"]["sorted"][-1]["share_" + hk], 3)) for r in rows]
        verdict[f"B_{axis}_random_{hk}_share_K1_to_K64"] = [(r["layer"], round(r["blocking"]["random"][0]["share_" + hk], 3), round(r["blocking"]["random"][-1]["share_" + hk], 3)) for r in rows]
    for axis in ("world_knowledge", "language_type"):
        for cond in ("short", "long"):
            d = [r["delta"] for r in results[f"{axis}/{cond}"]]
            verdict[f"C_{axis}_{cond}_delta_emb_final_mincross"] = [round(d[0], 3), round(d[-1], 3), next((i for i in range(1, len(d)) if d[i - 1] < 0 <= d[i]), None)]
    for axis in ("compass", "clock"):
        for cond in ("short", "long"):
            rows = results[f"{axis}/{cond}"]
            verdict[f"C_{axis}_{cond}_meanL3plus_effrank_h4share_h1share"] = [round(float(np.mean([r["eff_rank"] for r in rows[3:]])), 2),
                                                                             round(float(np.mean([r["share"]["h4"] for r in rows[3:]])), 3),
                                                                             round(float(np.mean([r["share"]["h1"] for r in rows[3:]])), 3)]
    print("VERDICT", json.dumps(verdict, indent=1, default=str))
    json.dump({"model": MODEL, "preamble": PREAMBLE, "results": results, "verdict": verdict},
              open(DATA / "llm_sector_blocking.json", "w"), indent=1, default=float)
    print("wrote", DATA / "llm_sector_blocking.json")


if __name__ == "__main__":
    main()
