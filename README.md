# Structure is in the zoom

**Probing neural symmetry through dimensionality scaling.** Adil Amin, ZEHEN Labs.
arXiv preprint, 2026 (identifier added on posting); a shorter version appeared at the
NeurReps workshop, NeurIPS 2026. This repository holds the code, the data artifacts
behind every number in the paper, and `theta-zoom`, the measurement as a tool.

## The finding in one paragraph

Dimensionality scaling exponents are treated as properties of a neural population or
of a network layer. They are not. Measured three ways on the same ten-thousand-neuron
patch of mouse V1, the exponent comes out 0.25, 0.31, and 0.35, and the three numbers
mean three different things: the first is pure sampling floor, the second is mostly
orientation structure, the third is a weak spatial redundancy. An exponent cannot be
read without the axis it was measured along. The paper decomposes every exponent,
exactly, into a label-blind sampling floor plus a shift **delta** earned only against a
declared probe axis, and tests the shift with exact permutation nulls and a second,
nuisance-preserving null.

![One recording, two axes](figures_canonical/fig_zoom_ladder_dir.png)

*Same recording (GT3, 11,311 neurons). Random stimulus subsets (gray) define the
empirical floor. The direction-aligned ladder (red) starts low and climbs steeply; the
slope difference is the shift, delta = +0.24.*

## What the paper shows

- **Cortex.** The direction-aligned shift is positive in all eight grating recordings and
  exceeds every one of 200 label permutations in each (p = 1/201). It replicates across
  167 Neuropixels populations from 32 mice in a second laboratory and tracks
  orientation-tuning strength (r = +0.41, mixed model p = 3e-7). Where the declared axis
  is degenerate, the instrument screens candidate axes and finds departures along
  session time, spatial frequency, and behavioral state.
- **Symmetry.** The axis that works follows the code's approximate O(2) symmetry. The
  even (orientation) and odd (direction) harmonic sectors are read out separately by
  ladder orderings, and the measured harmonics predicted two of three registered
  accumulation-order effects in advance, on cortex and on a rotation-equivariant CNN
  whose invariant layers give the exact null the theory demands.
- **Ground truth.** Ising and nematic lattices and the equivariant network fix the sign
  of the shift: conditioning on a scalar order parameter removes dimensions, accumulating
  the classes of a group orbit adds them, architectural invariance gives zero.
- **Language models.** Carried per layer into six model families, the instrument separates
  content axes inherited from tokens from construction axes built along the declared
  path. Label linkage is certified at nearly every depth under an order-free statistic;
  the depth profile's shape belongs to the declared class arrangement. A pre-registered
  alignment probe fails informatively: instruction tuning does not measurably reorganize
  moral-category covariance at 0.5-1B, on two taxonomies.
- **Two null levels.** The paper's sharpest methodological lesson: a nonzero shift means
  the partition changes covariance accumulation relative to its declared null, and reading
  it as representation of the label requires a null that preserves nuisance composition.
  On BLiMP minimal pairs a |z| of about 20 against the label-free permutation is absorbed
  entirely by a within-pair swap null; on the Baroni complexity contrasts a genuine
  label-linked signal survives it.

![Inferential results](figures_canonical/fig_nulls.png)

*(a) Pythia-2.8B: declared-path shift and order-averaged shift against their permutation
nulls. (b) BLiMP: composition only. (c) Baroni contrasts: signal beyond carriers.
(d) Spontaneous V1 sessions: circular-shift z against frame-permutation z.*

## The tool: `theta-zoom`

`theta_zoom.py` is a single-file, numpy-only implementation of the estimator: the
declared-axis shift, the label-blind floor, the 500-permutation null, the order-averaged
statistic for unordered class axes, and the stratified second null. It works on any
samples-by-features array and on any Hugging Face model.

```bash
pip install -e .            # numpy only; add [models] for torch + transformers
```

```python
from theta_zoom import zoom

res = zoom(X, labels, n_perm=500)              # X: (n_samples, n_features)
res["delta"], res["p_two"]                     # declared-order shift, exact null
res["delta_orderavg"], res["p_two_orderavg"]   # order-free partition statistic
res = zoom(X, labels, strata=carrier_ids)      # second null: label vs composition
res["strat_p_two"]
```

```bash
theta-zoom data X.npy labels.npy --strata strata.npy --n-perm 500
theta-zoom llm --model EleutherAI/pythia-160m --axis axes.json \
    --device mps --out battery.json          # axes.json: {class: [prompts...]}
theta-zoom llm ... --paper-seeds             # reproduces the paper's Table 5 cells exactly
theta-zoom plot battery.json --out profile.png   # depth profile with null bands
```

The paper's prompt battery ships in `axes/` (seven axes, eight classes of sixteen
prompts: world_knowledge, language_type, ethical, tqa_category, hs_activity,
arc_topic, plus INDEX.json), so `theta-zoom llm --model M --axis
axes/language_type.json` runs the paper's measurement on any model out of the box.

### Using it right

The instrument is easy to fool, and the paper documents each way. Follow these and
the numbers mean what they say:

1. **Declare the axis before you look.** Classes, class count, and accumulation order are
   fixed first; everything downstream is conditional on that declaration.
2. **Use six or more classes.** Two-class designs fit a slope through two points and are
   noise regardless of prompt count (the paper retired them).
3. **For unordered classes, read the order-averaged shift.** The declared-path profile is
   a property of one path through the classes; its shape, including its embedding-layer
   sign, can change with the order. The order-averaged statistic is the partition-level
   claim. Both are reported by `zoom()`.
4. **Ask what nuisance structure your partition preserves, and pass it as `strata`.**
   Templates, carriers, topics, sessions, minimal pairs. If the shift survives the
   within-stratum permutation, the label is doing work; if not, the ordinary floor was
   reading composition. This is the two-null-levels rule.
5. **Read counts as descriptive, profile statistics as inferential.** Per-layer p-values
   at alpha 0.05 carry about 1.7 false positives per 33 layers; the paper's inference rests
   on pre-registered profile statistics and on the order-averaged statistic, which survives
   Benjamini-Hochberg control intact.
6. **Within-class coherence beats prompt count.** Adding prompts that widen within-class
   topic diversity raises within-class dimensionality and weakens structural axes. Match
   final tokens across classes where you can.

## Reproduce the paper

```bash
python render_all.py        # headline tables and numbers from the committed artifacts
```

Every reported number traces to a JSON artifact in `data_canonical/` produced by a
script in `scripts_canonical/`. Registered expectations, where used, are in each
script's docstring and were committed before the run. The folder names mirror the
scripts' relative paths, so every script runs unmodified from a clone. Raw neural data
are public (Stringer et al. figshare releases; Allen Brain Observatory Neuropixels) and
are not included; `run3*`-`run4*` language-model scripts download models from the
Hugging Face Hub.

### Repository layout

- `theta_zoom.py` — the instrument and CLI (`theta-zoom`, `theta-zoom-render`).
- `scripts_canonical/` — 84 analysis scripts (V1, Allen, lattices, CNN, LLM battery).
- `data_canonical/` — 91 result JSONs, the committed outputs of the scripts.
- `figures_canonical/` — the paper's figures and where the `make_fig*` scripts write.
- `render_all.py` — prints the paper's tables from the artifacts.

### Headline claim -> artifact

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

## Citation

```bibtex
@article{amin2026zoom,
  author  = {Amin, Adil},
  title   = {Structure is in the zoom: probing neural symmetry through
             dimensionality scaling},
  journal = {arXiv preprint},
  year    = {2026},
  note    = {Code and data: https://github.com/adilamin89/structure-in-the-zoom}
}
```

## License

Code is released under the MIT License (see `LICENSE`). Result artifacts in
`data_canonical/` may be reused with attribution to the paper.
