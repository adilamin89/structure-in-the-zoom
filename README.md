# Structure is in the zoom

**Probing neural symmetry through dimensionality scaling.** Adil Amin, ZEHEN Labs.
arXiv 2026 (identifier added on posting) · NeurReps @ NeurIPS 2026 (short version).
Code, data, and `theta-zoom`, the measurement as a tool.

## Quick start (sixty seconds)

```bash
pip install -e ".[models]"                       # numpy core; [models] adds torch + transformers
theta-zoom llm --model EleutherAI/pythia-160m --axis axes/ --device mps --out pythia160m.json
theta-zoom plot pythia160m.json --out pythia160m.png
```

That runs the paper's whole prompt battery (seven declared axes, eight classes of
sixteen prompts each, in `axes/`) through every layer of the model and gives you,
per layer and per axis:

- `delta` and `p_two`: the axis-resolved shift along the declared class order and its
  exact 500-permutation p-value;
- `delta_orderavg` and `p_two_orderavg`: the order-averaged shift, the partition-level
  statistic for unordered classes;
- `strat_p_two` where a nuisance map exists (`axes/language_type.strata.json`): the
  shift against the nuisance-preserving null, which is the difference between "the
  labels organize this representation" and "the labels happen to preserve composition".

The plot shows one panel per axis: both statistics against their null bands.

## What it measures

A dimensionality scaling exponent is not a property of a system. On the same
ten-thousand-neuron patch of mouse V1 it reads 0.25, 0.31, and 0.35 along three probe
axes, and the three numbers mean three different things: pure sampling floor, mostly
orientation structure, a weak spatial redundancy. `theta-zoom` decomposes any exponent
exactly into a label-blind sampling floor plus a shift delta earned against a declared
axis, tests the shift with exact permutation nulls, and, when you tell it what nuisance
structure your partition preserves, with a second null that isolates what the labels add.

![One recording, two axes](figures_canonical/fig_zoom_ladder_dir.png)

## Automate it

**Checkpoint sweeps** (any Hugging Face revision; Pythia publishes 154 per model):

```bash
for r in step1000 step4000 step16000 step64000 step143000; do
  theta-zoom llm --model EleutherAI/pythia-410m-deduped --revision $r \
      --axis axes/language_type.json --device mps --out lt_410m_$r.json
done
```

**Model sweeps**: loop `--model`; every JSON has the same shape, so a few lines of
pandas give the cross-model table. `--paper-seeds` reproduces the paper's Table 5
cells to the last digit (tested: max |CLI - artifact| = 0).

**Your own axes.** An axis is a JSON file `{class_name: [prompt, ...]}`. Six or more
classes, sixteen or more prompts each, prompts within a class coherent in structure and
varied in content. To use the second null, add `<name>.strata.json` with the same shape
holding one nuisance label per prompt (topic, template, carrier, session); the tool picks
it up automatically.

**Python.**

```python
from theta_zoom import zoom, llm_battery

res = llm_battery("EleutherAI/pythia-160m", ["axes/"], device="mps", out_path="b.json")

# any samples-by-features array: trials x neurons, frames x units, prompts x hidden
r = zoom(X, labels, n_perm=500)                    # declared order + order-averaged
r = zoom(X, labels, strata=session_ids, n_perm=500) # + nuisance-preserving null
r["delta"], r["p_two"], r["delta_orderavg"], r["p_two_orderavg"], r["strat_p_two"]
```

`theta-zoom data X.npy labels.npy --strata strata.npy` is the same from the shell.

## Using it right

The instrument is easy to fool, and the paper documents each way:

1. **Declare before you look.** Classes, class count, and accumulation order are fixed
   first; everything downstream is conditional on that declaration.
2. **Six or more classes.** Two-class designs fit a slope through two points and are
   noise regardless of prompt count.
3. **For unordered classes, read the order-averaged shift.** The declared-path profile,
   including its sign at any one layer, is a property of one path through the classes.
4. **Pass your nuisance structure as strata.** If the shift survives the within-stratum
   permutation, the label is doing work; if not, the ordinary floor was reading
   composition. On BLiMP minimal pairs a |z| near 20 vanished under this null.
5. **Counts are descriptive; profile statistics are inferential.** Expect about 1.7
   false positives per 33 uncorrected layers; the order-averaged statistic survives
   Benjamini-Hochberg control intact, the declared-path counts partly do not.
6. **Coherence beats prompt count.** Prompts that widen within-class topic diversity
   raise within-class dimensionality and weaken structural axes.

## The paper in five results

- **Cortex.** The direction-aligned shift is positive in all eight grating recordings and
  exceeds every one of 200 label permutations (p = 1/201 each); it replicates across 167
  Neuropixels populations from 32 mice in a second laboratory and tracks orientation-
  tuning strength (r = +0.41, mixed model p = 3e-7). Where the declared axis is
  degenerate the instrument screens candidate axes and finds departures along session
  time, spatial frequency, and behavioral state.
- **Symmetry.** The working axis follows the code's approximate O(2) symmetry; ladder
  orderings read the even (orientation) and odd (direction) harmonic sectors separately,
  and the measured harmonics predicted two of three registered ordering effects in
  advance, on cortex and on a rotation-equivariant CNN.
- **Ground truth.** Ising and nematic lattices and the equivariant network fix the sign:
  conditioning on a scalar order parameter removes dimensions, accumulating a group orbit
  adds them, architectural invariance gives exactly zero.
- **Language models.** Across six model families the instrument separates content axes
  inherited from tokens from construction axes built along the declared path; label
  linkage is certified at nearly every depth under the order-free statistic, while the
  profile shape belongs to the class arrangement. A pre-registered alignment probe fails
  informatively: instruction tuning does not measurably reorganize moral-category
  covariance at 0.5-1B on two taxonomies.
- **Two null levels.** The methodological lesson: a nonzero shift means the partition
  changes covariance accumulation relative to its declared null; reading it as
  representation of the label needs a null that preserves nuisance composition.

![Inferential results](figures_canonical/fig_nulls.png)

## Reproduce every number

```bash
python render_all.py        # the paper's tables from the committed artifacts
```

Every reported number traces to a JSON artifact in `data_canonical/` produced by a
script in `scripts_canonical/`; registered expectations are in each script's docstring
and were committed before the run. Folder names mirror the scripts' relative paths so
every script runs unmodified from a clone. Raw neural data are public (Stringer et al.
figshare releases; Allen Brain Observatory Neuropixels) and are not included.

- `theta_zoom.py` — the instrument and CLI (`theta-zoom`, `theta-zoom-render`)
- `axes/` — the paper's prompt battery (+ `language_type.strata.json`, the topic map)
- `scripts_canonical/` — 84 analysis scripts · `data_canonical/` — 91 result JSONs
- `figures_canonical/` — the paper's figures; `make_fig*` scripts write here

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

Code: MIT (see `LICENSE`). Result artifacts in `data_canonical/` may be reused with
attribution to the paper.
