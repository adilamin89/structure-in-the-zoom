"""Run 53 — the fifth corpus-by-architecture cell: Mamba-2.8B on the Pile.

WHY: Section 8.4 attributes the construction-axis rise to "architecture lineage or
OLMo's idiosyncrasy" because the four cells (Pythia + GPT-Neo on the Pile;
RedPajama-INCITE on RedPajama; OLMo-1B on Dolma) never vary the architecture
off the GPT-Neo/NeoX lineage at fixed corpus. state-spaces/mamba-2.8b-hf is a
state-space model (no attention) trained on the Pile (300B tokens), i.e. the
same corpus as Pythia-2.8B and GPT-Neo-1.3B with a non-transformer architecture.
Same machinery as run37 (500 label permutations per layer, order-averaged
deltabar over K=50 orders, K_NULL_ORDERS=20, twenty-draw floors, paper seeds),
same prompts (run17.build_axes()), three axes: world_knowledge, language_type,
ethical. Hidden states = the residual stream after each of the 64 mixer blocks
plus the embedding (65 states), last token, fp16 on MPS (smoke test: finite,
|h| <= 141).

REGISTERED EXPECTATIONS (written before the run, 2026-09-04 23:05 CDT):
M1 (the one to beat): the construction-axis rise is a NeoX-lineage property, so
   on Mamba the language_type profile does NOT rise: integrated excess IE and
   peak excess PE fail p <= 0.01 (one-sided, positive). If both pass at
   p <= 0.01, the rise is not lineage-specific and the Section 8.4 attribution
   closes toward "OLMo-1B is the idiosyncratic cell".
M2: world_knowledge positive at the embedding and declining (IE < 0 in the
   registered direction), as on every dense model so far.
M3: ethical declared-order per-layer counts at chance (about 1.7 of 33 expected;
   here 65 layers, about 3); order-averaged linkage weak, as on Pythia-2.8B.
M4: order-averaged label linkage certified at most depths on world_knowledge
   and language_type (the partition-level statistic), whatever the shape.
Out: ../data_canonical/run53_mamba_fifth_cell.json (log alongside).
"""
import importlib.util
from pathlib import Path
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("r37", HERE / "run37_inferential_nulls.py")
r37 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r37)
r37.OUT = HERE.parent / "data_canonical" / "run53_mamba_fifth_cell.json"
r37.MODELS = [("state-spaces/mamba-2.8b-hf", "mps")]
r37.AXES = ["world_knowledge", "language_type", "ethical"]
r37.DECLARED_DIRECTION["ethical"] = {"IE": +1, "PE": +1}
r37.main()
