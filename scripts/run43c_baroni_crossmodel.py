"""Run 43c — Baroni 64-pair contrasts across the remaining battery models.

WHY: run43b certified a label-linked signal beyond minimal-pair carriers on
Pythia-2.8B (13/33 strat layers) and 160m (3/13); the user asked the natural
next question — does the external-benchmark arrangement signal scale and
does it track corpus, as the hand-written structural axis does? Same script,
same 64-pair design, four more models.

REGISTERED EXPECTATIONS (written before the run):
BC1: the fraction of strat-certified layers increases with Pythia scale
     (410m and 1B between 160m's 3/13 and 2.8B's 13/33), consistent with
     Baroni et al.'s finding that complexity effects strengthen with scale.
BC2: DECISION CELL, corpus vs architecture: if Pile-trained GPT-Neo carries
     the beyond-carrier signal and Dolma-trained OLMo-1B is weaker, the
     corpus-tracking of the structural axis (run29) extends to external
     stimuli; the reverse or both-null scopes it.

Out: ../data_canonical/run43c_baroni_crossmodel.json
"""
import importlib.util
from pathlib import Path
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("r43", HERE / "run43_baroni_complexity.py")
r43 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r43)
r43.N_PAIRS = 64
r43.MODELS = [
    ("EleutherAI/pythia-410m-deduped", "mps"),
    ("EleutherAI/pythia-1b-deduped", "mps"),
    ("allenai/OLMo-1B-hf", "mps"),
    ("EleutherAI/gpt-neo-1.3B", "mps"),
]
r43.OUT = HERE.parent / "data_canonical" / "run43c_baroni_crossmodel.json"
r43.main()
