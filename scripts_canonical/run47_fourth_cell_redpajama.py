"""Run 47 — the fourth cell: one architecture, two corpora.

WHY: Sections 8.3 and 8.5 stop at a three-cell corpus-by-architecture
association (Pile x {Pythia, GPT-Neo} rise; Dolma x OLMo no rise) and state
that "a fourth cell, one architecture across two corpora, is required before
existence can be attributed to corpus." RedPajama-INCITE-Base-3B-v1 is the
GPT-NeoX architecture at Pythia-2.8B's exact configuration (32 layers,
d=2560), trained on RedPajama-1T instead of the Pile. Same battery, same
run37 machinery (500 label permutations per layer, order-averaged deltabar).

REGISTERED EXPECTATIONS (written before the run):
F1: world_knowledge positive and per-layer certified at most layers
    (content axes have been universal across all six models so far).
F2: DECISION CELL, pre-declared direction from the three-cell pattern
    (rise present on both Pile models, absent on the Dolma model): the
    corpus reading predicts the structural rise is ABSENT or weak on
    RedPajama (integrated excess NOT certified, no negative-to-positive
    crossover); the architecture reading predicts it is PRESENT (IE p<0.05,
    crossover like Pythia-2.8B). We register the corpus prediction as the
    one to beat. Either outcome closes the paper's stated limitation.

Out: ../data_canonical/run47_fourth_cell_redpajama.json
"""
import importlib.util
from pathlib import Path
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("r37", HERE / "run37_inferential_nulls.py")
r37 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r37)
r37.MODELS = [("togethercomputer/RedPajama-INCITE-Base-3B-v1", "mps")]
r37.OUT = HERE.parent / "data_canonical" / "run47_fourth_cell_redpajama.json"
r37.main()
