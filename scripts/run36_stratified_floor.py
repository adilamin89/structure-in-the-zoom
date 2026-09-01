"""Run 36 — Carrier-stratified matched floor for the planted axes
(round-4 referee item 1: the deepest catch of the campaign).

THE ISSUE: at the planted axes' embedding layer the class label carries no
information (each class contains the same sixteen carrier vectors), yet
delta = -0.2. The structured rung is perfectly stratified over carriers
while random floor subsets draw duplicates and omissions, so delta can
register nuisance-composition structure rather than label representation.

THE FIX (two null levels): the ordinary matched floor controls sample
count and global geometry; a CARRIER-STRATIFIED floor additionally
preserves the observed rung's carrier composition (at rung size 16k it
draws exactly k instances of each carrier, class versions chosen at
random). A within-carrier label permutation gives the matching shuffle:
each carrier's eight class versions receive a permutation of the eight
labels, preserving every class's carrier composition exactly.

REGISTERED EXPECTATIONS:
T1: at the embedding layer delta_stratified ~ 0 for both planted axes
    (the degenerate-layer signal is entirely carrier composition, and the
    right null removes it).
T2: at later layers, EITHER delta_stratified stays ~ -0.2 (the negative
    sign law is genuinely class-linked) OR it collapses (the planted-axis
    negative delta was carrier composition throughout) — either outcome
    is decisive and reportable, and rewrites the generic claim
    accordingly.
T3: within-carrier label shuffles ~ 0 under the stratified floor.

Model: Pythia-160m (MPS). Out: ../data_canonical/run36_stratified_floor.json
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "data_canonical" / "run36_stratified_floor.json"

spec = importlib.util.spec_from_file_location(
    "r28", HERE / "run28_cyclic_axis_llm.py")
r28 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r28)
r17 = r28.r17

BIN_COUNTS = [1, 2, 3, 4, 6, 8]
N_FLOOR = 10
N_SHUF = 20


def stratified_delta(X, labels, carriers, rng):
    """Ladder delta against a carrier-stratified floor, with
    within-carrier label-permutation shuffles."""
    n_classes, n_carriers = 8, 16
    members = [np.where(labels == c)[0] for c in range(n_classes)]
    car_members = [np.where(carriers == t)[0] for t in range(n_carriers)]

    sizes, prs = [], []
    for k in BIN_COUNTS:
        sel = np.concatenate(members[:k])
        sizes.append(len(sel))
        prs.append(r17.pr_c(X[sel]))
    th_o = r17.slope(sizes, np.asarray(prs))

    # stratified floor: at size 16k draw exactly k versions of each carrier
    nl = np.zeros((N_FLOOR, len(sizes)))
    for d in range(N_FLOOR):
        for ki, k in enumerate(BIN_COUNTS):
            sel = np.concatenate([rng.choice(cm, k, replace=False)
                                  for cm in car_members])
            nl[d, ki] = np.log(max(r17.pr_c(X[sel]), 1e-9))
    th_f = r17.slope(sizes, np.exp(nl.mean(axis=0)))

    # within-carrier label permutation shuffles
    shufs = []
    for s in range(N_SHUF):
        srng = np.random.default_rng(700 + s)
        plab = labels.copy()
        for cm in car_members:
            plab[cm] = labels[cm][srng.permutation(len(cm))]
        m2 = [np.where(plab == c)[0] for c in range(n_classes)]
        sz2, pr2 = [], []
        for k in BIN_COUNTS:
            sel = np.concatenate(m2[:k])
            sz2.append(len(sel))
            pr2.append(r17.pr_c(X[sel]))
        shufs.append(r17.slope(sz2, np.asarray(pr2)) - th_f)
    return th_o - th_f, float(np.mean(shufs)), float(np.std(shufs))


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    axes = {
        "compass": {d: [t.format(d=d) for t in r28.COMPASS_TEMPLATES]
                    for d in r28.COMPASS},
        "clock": {t_: [tpl.format(t=t_) for tpl in r28.CLOCK_TEMPLATES]
                  for t_ in r28.CLOCK},
    }
    model = AutoModelForCausalLM.from_pretrained(
        "EleutherAI/pythia-160m").to("mps")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    out = {"model": "EleutherAI/pythia-160m", "design":
           "carrier-stratified floor + within-carrier label shuffles",
           "axes": {}}
    for axis_name, classes in axes.items():
        prompts, labels, carriers = [], [], []
        for ci, cn in enumerate(classes):
            for ti, p in enumerate(classes[cn]):
                prompts.append(p)
                labels.append(ci)
                carriers.append(ti)
        labels = np.array(labels)
        carriers = np.array(carriers)
        per_layer = r17.get_hidden_states(model, tokenizer, prompts,
                                          device="mps")
        rows = []
        for l in range(len(per_layer)):
            X = per_layer[l].astype(np.float32)
            X = X / (X.std() + 1e-9)
            d_plain, _ = r17.ladder_delta(X, labels, 8,
                                          np.random.default_rng(l * 100 + 1))
            d_strat, sh_m, sh_sd = stratified_delta(
                X, labels, carriers, np.random.default_rng(l * 100 + 1))
            rows.append({"layer": l, "delta_plain": d_plain,
                         "delta_stratified": d_strat,
                         "strat_shuffle_mean": sh_m,
                         "strat_shuffle_sd": sh_sd})
        out["axes"][axis_name] = rows
        e, f = rows[0], rows[-1]
        print(f"[{axis_name}] L0 plain {e['delta_plain']:+.3f} -> strat "
              f"{e['delta_stratified']:+.3f} | final plain "
              f"{f['delta_plain']:+.3f} -> strat "
              f"{f['delta_stratified']:+.3f} | strat-shuf "
              f"{f['strat_shuffle_mean']:+.3f}±{f['strat_shuffle_sd']:.3f}",
              flush=True)
        json.dump(out, open(OUT, "w"), indent=1)
    print("DONE run36", flush=True)


if __name__ == "__main__":
    main()
