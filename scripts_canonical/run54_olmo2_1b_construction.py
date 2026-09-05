"""Run 54 — is OLMo-1B the outlier, or the OLMo family? OLMo-2-1B on the content and
construction axes with run37's machinery.

WHY: after run53 (Mamba-2.8B on the Pile carries the construction-axis trajectory),
OLMo-1B (allenai/OLMo-1B-hf, Dolma) is the only model in the corpus-by-architecture
design without the rise, and it reads almost nothing on ANY axis (content 0/17
declared, 1/17 order-averaged; ethical exactly zero at every layer, App G). The
paper's own rule says delta = 0 means the probe is blind, not that the system lacks
structure. OLMo-2-0425-1B (a different OLMo design and mix; already in the paper for
the ethical/base-instruct probe, where world_knowledge behaved as a normal positive
control) is the cheapest discriminating run: same machinery (500 permutations per
layer, order-averaged deltabar K=50, K_NULL_ORDERS=20, twenty-draw floors, paper
seeds), axes world_knowledge + language_type, fp16 on MPS.

REGISTERED EXPECTATIONS (written before the run, 2026-09-05 00:20 CDT):
N1: OLMo-2-1B shows the construction rise (language_type integrated excess AND peak
    excess p <= 0.01 one-sided positive; negative embedding value). If so, the outlier
    is OLMo-1B v1 specifically (its representation geometry makes the linear probe
    blind on every axis), not the OLMo family or the Dolma corpus, and the Section 8.4
    attribution closes: the rise is architecture-general among models the probe can read.
N2: world_knowledge positive at the embedding with layerwise declared-order
    certification at more than chance (the run44 positive control behaved normally).
If N1 fails with N2 passing, the OLMo family lacks the construction rise while reading
content: the corpus/design reading survives and is stated as such.
Out: ../data_canonical/run54_olmo2_1b_construction.json (log alongside).
"""
import importlib.util
from pathlib import Path
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("r37", HERE / "run37_inferential_nulls.py")
r37 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r37)
r37.OUT = HERE.parent / "data_canonical" / "run54_olmo2_1b_construction.json"
r37.MODELS = [("allenai/OLMo-2-0425-1B", "mps")]
r37.AXES = ["world_knowledge", "language_type"]
r37.main()
