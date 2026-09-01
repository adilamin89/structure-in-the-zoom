"""Run 37b — extend the run37 inferential nulls to the remaining battery models.

WHY: Table 1 of the arXiv version reports the content/structural rows for
Pythia-410m, Pythia-1B, OLMo-1B, and GPT-Neo-1.3B with descriptive
sig/shuffle ratios only; run37 certified 160m and 2.8B inferentially. Same
machinery (500 label permutations per layer, order-averaged deltabar K=50,
identical seeds), same artifact (resume skips completed models), so the
whole table can speak permutation quantiles.

REGISTERED EXPECTATIONS (written before the run):
X1: wk per-layer positive certification at most layers in all four models
    (order-averaged statistic the sensitive one, per run37/42).
X2: structural-axis late-layer certification on GPT-Neo (the Pile-trained
    riser); OLMo-1B's non-rising profile expected to show fewer certified
    cells on lt (its trajectory differs, run25).

Out: appended into ../data_canonical/run37_inferential_nulls.json
"""
import importlib.util
from pathlib import Path
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("r37", HERE / "run37_inferential_nulls.py")
r37 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r37)
r37.MODELS = [
    ("EleutherAI/pythia-410m-deduped", "mps"),
    ("EleutherAI/pythia-1b-deduped", "mps"),
    ("allenai/OLMo-1B-hf", "mps"),
    ("EleutherAI/gpt-neo-1.3B", "mps"),
]
r37.main()
