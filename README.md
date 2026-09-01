# Supplementary code and data — "Structure is in the zoom"

Every number in the paper traces to a JSON artifact in `data_canonical/` produced by a
script in `scripts_canonical/`. The mapping below covers the paper's headline claims;
the full number -> artifact -> script manifest ships with the repository.

## Two ways to use this repository

**Reproduce the paper.** `scripts_canonical/` + `data_canonical/` trace every number to its
generating script and committed artifact (table below); registered
expectations are in each script's docstring, committed before the run.

**Use the instrument on your own data.** `theta_zoom.py` is a single-file,
numpy-only implementation of the estimator: declared-axis shift, label-blind
floor, 500-permutation per-layer null, order-averaged statistic for
unordered axes, and the nuisance-preserving stratified null. Rows-by-features
in, statistics out:

```python
import numpy as np
from theta_zoom import zoom

res = zoom(X, labels, n_perm=500)          # X: (n_samples, n_features)
res["delta"], res["p_two"]                 # declared-order shift + exact null
res["delta_orderavg"], res["p_two_orderavg"]   # order-free partition statistic
res = zoom(X, labels, strata=carrier_ids)  # label reading beyond composition
res["strat_p_two"]
```

Or from the command line (`pip install -e .`):

```bash
theta-zoom data X.npy labels.npy --strata strata.npy --n-perm 500
theta-zoom llm --model EleutherAI/pythia-160m --axis axes.json \
    --device mps --out battery.json     # axes.json: {class: [prompts...]}
```

For a language model: collect last-token hidden states per layer for a
prompt battery (one forward pass per prompt) and call `zoom()` per layer;
`scripts_canonical/run37_inferential_nulls.py` is the full worked example. Before
reading any nonzero shift as label representation, ask what nuisance
structure your partition preserves (carriers, topics, sessions) and pass it
as `strata` — the paper's two-null-levels rule.

## Layout

The folder names mirror the scripts' hard-coded relative paths
(`HERE.parent / "data_canonical"`), so every script runs unmodified from a
clone. `figures_canonical/` holds the paper figures and is where the
`make_fig*` scripts write.

- `scripts_canonical/` — 82 analysis scripts. Registered expectations, where used, are
  recorded in each script's docstring and were written before the run.
- `data_canonical/` — 90 result JSONs (committed outputs of the scripts).
- Raw data are not included; all sources are public (below).

## Headline claim -> artifact

| Claim | Artifact | Script |
|---|---|---|
| V1 direction-aligned shifts, shuffle control | (repo: stringer decomposition JSONs) | shuffle_label_control.py |
| Ladder-design robustness (32/32 cells) | run8_ladder_robustness.json | run8_ladder_robustness.py |
| Accumulation-order reversal (8/8) | antipodal_order.json | accumulation_order.py |
| Allen mixed model + partials | run1_allen_partial_mixed.json | run1_allen_partial_mixed.py |
| Allen split-half cross-fit | s69_allen_crossfit.json | allen_crossfit.py |
| Calibrated model bracket (5 seeds) | run2b_corotating_seeds.json | run2b_corotating_seeds.py |
| Alignment-calibrated prediction (fresh draw) | run11b_fresh_draw_prediction.json | run11_bootstrap_prediction.py |
| pi-periodic alignment + residualized control | run9_alignment_225.json, run3b_principal_angles_residualized.json | run9_alignment_225.py, run3b_principal_angles_residualized.py |
| Transport test (no common rotation) | run7_transport_test.json | run7_transport_test.py |
| Ising / nematic lattices | ising_L128_tc.json, nematic_polar_delta.json | ising_L128_tc.py, nematic_polar_delta.py |
| CNN double dissociation (5 seeds) | run5b_cnn_seeds.json | run5b_cnn_seeds.py |
| CNN sector content + stimulus baseline | run5c_cnn_multipole_fixed.json, run14_stimulus_baseline.json | run5c_cnn_multipole_fixed.py |
| CNN prospective ordering | run12_cnn_ordering.json | run12_cnn_ordering.py |
| V1 label-permutation exact test (16/16, p=1/201) | run38_v1_label_permutations.json | run38_v1_label_permutations.py |
| LLM 500-permutation nulls + order-averaged shift (6 models, 3 axes) | run37_inferential_nulls.json | run37_inferential_nulls.py (+37b/37c wrappers) |
| Allen A_even/A_odd direct mixed model | run39_allen_aeven_mixed.json | run39_allen_aeven_mixed.py |
| Spont state axes, permutation + circular-shift nulls | run40_spont_state_axis.json | run40_spont_state_axis.py |
| CNN ordering per-seed paired stats | run41_cnn_ordering_perseed.json | run41_cnn_ordering_perseed.py |
| BLiMP native stratified floor (within-pair swaps) | run42_blimp_battery.json | run42_blimp_battery.py |
| Baroni complexity contrasts, 16 and 64 pairs | run43_baroni_complexity.json, run43b_baroni_64pairs.json | run43_baroni_complexity.py, run43b_baroni_64pairs.py |
| Base-instruct ethical comparison (two pairs) | run44_base_instruct_ethical.json | run44_base_instruct_ethical.py |
| Topic-stratified floor for the construction axis | run45_lt_stratified_floor.json | run45_lt_stratified_floor.py |
| LLM battery, Pythia 160m/410m/1B/2.8B | run17/18/19/26_*.json | run17/18/19/26_*.py |
| LLM robustness (prompt/order/pooling) | run20_robustness_battery.json | run20_robustness_battery.py |
| Cross-model ranking rho=1.0, slope law | run22_cross_prediction.json | run22_cross_prediction.py |
| Prompt-diversity dilution + OLMo 32/class | run23_expanded_prompts.json | run23_expanded_prompts.py |
| Kernel eff-rank diagnostic | run24_kernel_analysis.json | (inline; JSON committed) |
| OLMo-1B matched 16/class battery | run25_olmo1b_16pc_battery.json | run25_olmo1b_16pc_battery.py |
| Static-session axis search (discovery mode) | run27_static_axis_search.json | run27_static_axis_search.py |
| Planted-C8 axes (compass/clock) | run28_cyclic_axis_llm.json | run28_cyclic_axis_llm.py |
| Figures 5-6 | (generated from JSONs above) | make_fig5_mechanism.py, make_fig6_llm.py |

## Raw data sources (all public)

- Stringer et al. V1 recordings: figshare (stringer_v1 releases).
- Allen Brain Observatory Neuropixels: AllenSDK / brain-map.org.
- MNIST: torchvision.
- LLMs: EleutherAI Pythia + allenai OLMo on Hugging Face; TruthfulQA,
  HellaSwag, ARC-Challenge via their public repositories.
- All LLM prompts appear verbatim inside the run scripts.

## Environment

See `requirements.txt`. Scripts run on CPU except where noted (LLM batteries
use MPS/CUDA if available; every model fits in 16 GB at fp16 except
Pythia-2.8B, which wants ~8 GB for weights alone).
- Scripts keep their original relative output paths; all committed outputs are provided in data_canonical/.
