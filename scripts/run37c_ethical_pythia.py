"""Run 37c — ethical-concept axis under the run37 machinery on Pythia 160m + 2.8B.

WHY: run44 found the order-averaged statistic certifies weak ethical-axis
organization on OLMo-2-1B and Qwen2.5-0.5B BASE models (10/17, 14/25 layers)
while the canonical statistic stays ~0, and instruct tuning adds nothing.
Whether that organization is a property of newer pretraining corpora or
simply what the sensitive statistic finds in ANY base model requires the
same measurement on Pythia (Pile, 2020-era). Same machinery, same artifact.

REGISTERED EXPECTATIONS (written before the run):
C1: canonical ethical stays uncertified at nearly all layers (the existing
    six-base-model zero, now exact).
C2: DECISION CELL: order-averaged ethical on Pythia — if certified at a
    comparable fraction of layers to OLMo-2/Qwen (~50-60%), the ethical
    zero was statistic-sensitivity and the axis is weakly organized in all
    base models; if near zero, the organization tracks the newer corpora.

Out: appended into ../data_canonical/run37_inferential_nulls.json
"""
import importlib.util
from pathlib import Path
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("r37", HERE / "run37_inferential_nulls.py")
r37 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r37)
r37.AXES = ["ethical"]
r37.DECLARED_DIRECTION["ethical"] = {"IE": +1, "PE": +1}
r37.MODELS = [
    ("EleutherAI/pythia-160m", "cpu"),
    ("EleutherAI/pythia-2.8b-deduped", "mps"),
]
r37.main()
