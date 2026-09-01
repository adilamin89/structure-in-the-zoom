"""Run 43b — Baroni contrasts at 64 pairs per contrast (resolution upgrade).

WHY: run43 (16 pairs) found NO signal at either null level — a registered
miss for BA1 whose diagnosis is that all three Baroni datasets share one
closed lexicon across classes (no cross-class vocabulary difference, hence
no composition signal; bracketing BLiMP's |z|~20 from the other side). This
upgrade quadruples the sample to separate "below resolution at 16 pairs"
from "absent": 64 pairs per contrast -> 6 classes x 64 = 384 prompts.

REGISTERED EXPECTATIONS (written before the run):
BB1: label-free perm null still matched at embedding (shared lexicon).
BB2: DECISION at n=64: strat null exceeded at mid/late layers on 2.8B if
     the arrangement contrasts organize accumulation at all; a second null
     scopes the instrument cleanly (contrasts live in within-set geometry,
     per Baroni's ID measurements, not between-class accumulation).

Out: ../data_canonical/run43b_baroni_64pairs.json
"""
import importlib.util
from pathlib import Path
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("r43", HERE / "run43_baroni_complexity.py")
r43 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r43)
r43.N_PAIRS = 64
r43.OUT = HERE.parent / "data_canonical" / "run43b_baroni_64pairs.json"
r43.main()
