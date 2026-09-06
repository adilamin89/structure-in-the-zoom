"""Run 61 (S78b, 2026-09-06) - The rung-four stall in a language model: a code that
reads the parity quotient of a planted cycle should stall at four classes, and
the units that distinguish a class from its antipode are the language-model
analogue of the direction-selective fraction.

DESIGN: Pythia-160m (fp16 on MPS), last-token hidden states at every layer
(the embedding and twelve blocks), three axes from the released battery:
compass (eight directions on C8, sixteen shared carrier sentences; Section 8.6
finds its kernel dominated by the alternating harmonic, the parity quotient),
clock (eight three-hour times on C8, sixteen carriers; organized by
period-of-day phrases, which do differ between a time and its antipode), and
the construction axis (eight sentence types, no cycle; the antipode map
c -> c + 4 is an arbitrary pairing). For each axis and layer:
(i) rung 1.3's zoom() with antipode = {c: c + 4 mod 8} on the declared order
    (one member of each pair first): the per-rung deficit and the late fraction
    against the ordinary floor, n_perm = 200;
(ii) the same ladder against a carrier-stratified floor (compass and clock):
    each floor draw takes, for every carrier, exactly the number of prompts
    the observed rung holds from that carrier, chosen at random, so the
    carrier-composition artifact of Section 8.6 is removed from the deficit;
(iii) per unit, the class-mean tuning curve over the eight classes, its DFT,
    and the odd share (harmonics 1 and 3 over harmonics 1 to 4); the
    population odd share is the same ratio summed over units (the kernel's odd
    share); the fraction of units with odd share above 0.5 is the analogue of
    the direction-selective fraction.

REGISTERED EXPECTATIONS (written before the run; layers 3-12, where class
information has reached the last token):
P1: with the carrier-stratified floor the compass axis's late fraction is
    below the clock axis's at every layer from 3 on (10 of 10).
P2: the compass axis's population odd share is below the clock's at every
    layer from 3 on (10 of 10).
P3: across the twenty (axis, layer) cells of compass and clock from layer 3 on,
    the stratified late fraction tracks the population odd share (Spearman
    >= 0.7).
The construction axis's late fraction (ordinary floor) and odd share are
reported for scale. A miss is reported at full volume.

Out: ../data_canonical/run61_llm_antipode_stall.json (+ .log)
"""
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data_canonical"
SUPP = HERE.parent.parent / "arxiv_supplement"
OUT = DATA / "run61_llm_antipode_stall.json"
spec = importlib.util.spec_from_file_location("rung", SUPP / "rung.py")
rung = importlib.util.module_from_spec(spec); spec.loader.exec_module(rung)

MODEL = "EleutherAI/pythia-160m"
AXES = ["compass", "clock", "language_type"]
NC = 8
ANTIPODE = {c: (c + 4) % NC for c in range(NC)}
N_PERM = 200
N_FLOOR = 20
LAYER_FROM = 3


def encode(model_name, prompts, device="mps", max_len=128):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype = torch.float16 if device in ("mps", "cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device); model.eval()
    tok = AutoTokenizer.from_pretrained(model_name)
    states = []
    with torch.no_grad():
        for p in prompts:
            ids = tok(p, return_tensors="pt", truncation=True, max_length=max_len).input_ids.to(device)
            hs = model(ids, output_hidden_states=True).hidden_states
            states.append([h[0, -1, :].float().cpu().numpy() for h in hs])
    return [np.stack([s[l] for s in states]) for l in range(len(states[0]))]


def stratified_deficit(X, labels, strata, rng):
    """The declared ladder against a floor that keeps the carrier composition
    of every rung: for each carrier, as many class versions as the rung holds,
    class versions at random."""
    X = np.asarray(X, float); X = X / (X.std() + 1e-9); K = X @ X.T
    members = [np.where(labels == c)[0] for c in range(NC)]
    bc = [c for c in rung.BIN_COUNTS_DEFAULT if c <= NC]
    carriers = {s: np.where(strata == s)[0] for s in np.unique(strata)}
    sizes, obs, flo = [], [], []
    for c in bc:
        sel = np.concatenate(members[:c]); sizes.append(len(sel))
        obs.append(np.log(max(rung._subset_pr(K, sel), 1e-9)))
        # the observed rung's carrier composition, carrier by carrier (the
        # sidecars are not a clean classes-by-carriers grid)
        want = {s: int((strata[sel] == s).sum()) for s in carriers}
        draws = []
        for _ in range(N_FLOOR):
            idx = np.concatenate([rng.choice(v, want[s], replace=False) for s, v in carriers.items() if want[s] > 0])
            draws.append(np.log(max(rung._subset_pr(K, idx), 1e-9)))
        flo.append(float(np.mean(draws)))
    deficit = [o - f for o, f in zip(obs, flo)]
    delta = rung._slope(sizes, np.exp(obs)) - rung._slope(sizes, np.exp(flo))
    return {"sizes": sizes, "deficit": deficit, "delta": float(delta),
            "late_fraction": float(deficit[bc.index(4)] / deficit[0]) if deficit[0] != 0 else None}


def odd_share(X, labels):
    mu = np.stack([X[labels == c].mean(0) for c in range(NC)])
    F = np.fft.fft(mu - mu.mean(0, keepdims=True), axis=0)
    p = np.abs(F[1:5]) ** 2                       # harmonics 1..4 per unit (4 is the Nyquist term)
    odd_j = (p[0] + p[2]) / (p.sum(0) + 1e-12)
    return {"odd_share_pop": float((p[0] + p[2]).sum() / (p.sum() + 1e-12)),
            "frac_units_odd": float((odd_j > 0.5).mean()), "median_unit_odd_share": float(np.median(odd_j))}


def main():
    t0 = time.time()
    out = {"design": {"model": MODEL, "axes": AXES, "n_perm": N_PERM, "n_floor_strat": N_FLOOR, "antipode": ANTIPODE, "layer_from": LAYER_FROM}, "axes": {}}
    for ax in AXES:
        name, classes, prompts, labels, strata = rung._load_axis(str(SUPP / "axes" / f"{ax}.json"), use_strata=True)
        print(f"[{ax}] {len(classes)} classes, {len(prompts)} prompts, strata {'yes' if strata is not None else 'no'} | {time.time()-t0:.0f}s", flush=True)
        layers = encode(MODEL, prompts)
        rows = []
        for l, X in enumerate(layers):
            r = rung.zoom(X, labels, n_perm=N_PERM, k_orders=20, k_null_orders=10, n_floor_draws=N_FLOOR, seed=l, antipode=ANTIPODE)
            row = {"layer": l, "delta": r["delta"], "p_two": r.get("p_two"), "deficit": r["deficit"], "late_fraction": r.get("late_fraction"),
                   "first_half_has_no_pair": r.get("antipode_first_half_has_no_pair")}
            if strata is not None:
                row["strat"] = stratified_deficit(X, labels, strata, np.random.default_rng(1000 + l))
            row.update(odd_share(X, labels))
            rows.append(row)
            lf = row["late_fraction"]; ls = row["strat"]["late_fraction"] if "strat" in row else None
            print(f"  L{l:2d}: delta {row['delta']:+.3f} (p {row['p_two']:.3f}) | late {lf if lf is None else round(lf, 2)} | strat late {ls if ls is None else round(ls, 2)} "
                  f"| odd share {row['odd_share_pop']:.2f} | units odd>0.5 {row['frac_units_odd']:.2f} | {time.time()-t0:.0f}s", flush=True)
        out["axes"][ax] = {"classes": classes, "layers": rows}
        json.dump(out, open(OUT, "w"), indent=1)
    comp, clk = out["axes"]["compass"]["layers"], out["axes"]["clock"]["layers"]
    L = range(LAYER_FROM, len(comp))
    p1 = [comp[l]["strat"]["late_fraction"] < clk[l]["strat"]["late_fraction"] for l in L]
    p2 = [comp[l]["odd_share_pop"] < clk[l]["odd_share_pop"] for l in L]
    cells = [(a[l]["odd_share_pop"], a[l]["strat"]["late_fraction"]) for a in (comp, clk) for l in L]
    rho = float(spearmanr([c[0] for c in cells], [c[1] for c in cells])[0])
    verdict = {"P1_count": int(sum(p1)), "P1": bool(all(p1)), "P2_count": int(sum(p2)), "P2": bool(all(p2)), "P3_rho": rho, "P3": bool(rho >= 0.7),
               "compass_strat_late": [comp[l]["strat"]["late_fraction"] for l in L], "clock_strat_late": [clk[l]["strat"]["late_fraction"] for l in L],
               "compass_odd_share": [comp[l]["odd_share_pop"] for l in L], "clock_odd_share": [clk[l]["odd_share_pop"] for l in L],
               "construction_late_ordinary": [out["axes"]["language_type"]["layers"][l]["late_fraction"] for l in L],
               "construction_odd_share": [out["axes"]["language_type"]["layers"][l]["odd_share_pop"] for l in L]}
    out["verdict"] = verdict
    print("VERDICT", json.dumps(verdict, indent=1), flush=True)
    json.dump(out, open(OUT, "w"), indent=1); print(f"wrote {OUT} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
