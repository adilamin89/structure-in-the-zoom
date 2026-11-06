# Structure is in the zoom

**Probing neural symmetry through dimensionality scaling.** Adil Amin, ZEHEN Labs. NeurReps 2026 workshop (Proceedings of
Machine Learning Research). This repository holds `rung`, the paper's measurement as a tested tool, and the code and data
behind every number in the paper.

A dimensionality-scaling exponent, the slope of the participation ratio against subsample size on log-log axes, measures a
population and the probe that sampled it together. One recording of 11,311 mouse V1 neurons returns 0.25 when stimuli are added
at random, 0.30 when stimulus directions are added class by class, and 0.35 when neighbouring neurons are added. Two of the three
are floors, what random subsets of the same sizes return through the same estimator; direction alone leaves its floor
(0.305 = a floor of 0.065 + a shift of 0.240). `rung` makes that split for any reported exponent.

![One population, many probes](figures_canonical/fig_probes.png)

## What rung tells you

For a samples-by-features array with one label per row (trials by neurons, prompts by hidden units, images by channels), and for
every layer of a language model at any training checkpoint, with no probe or model trained:

- **how much of an exponent is floor**, and the shift a declared label adds to it, against label permutations and nulls that keep
  a nuisance, an acquisition block or the labels' time order fixed (`theta_obs = theta_floor + delta`, exact whenever the declared
  ladder and its floor share rung sizes);
- **what carries the shift**: fluctuations that differ between classes, or the class means, with the sign law that says which way
  the class means push;
- **whether two measured terms account for it**, nothing fitted, and whether each class is sampled well enough to read it;
- **which of several candidate labels organizes a population**, discovered on one half of the data and replicated on the other;
- **what other samplings keep**: neighbouring neurons, features in a given order, blocks of features pooled by a score;
- **for a language model**, how much a declared label (a topic, a construction) adds to each layer's exponent, how much of that is
  the final token, and which attention heads to test first.

A decoder tells whether a label can be read out; the shift tells how much of a reported exponent the label accounts for and what
sets that share. The grouping of samples into classes carries the shift; the order in which the classes enter moves it little, so
for unordered classes `rung` reports the order-averaged shift beside the declared one.

## Install and run

```bash
pip install git+https://github.com/adilamin89/structure-in-the-zoom      # numpy, scipy, matplotlib
rung data X.npy labels.npy --decompose --out r.json --plot r.png          # any samples-by-features array, one integer label per row
rung summarize r.json                                                     # the reading, in plain language
```

For a language model, clone the repository (the paper's prompt axes are in `axes/`) and add the model extras:

```bash
git clone https://github.com/adilamin89/structure-in-the-zoom && cd structure-in-the-zoom
pip install -e ".[models]"                                                # + torch, transformers, datasets
rung llm --model EleutherAI/pythia-160m --axis axes/world_knowledge32.json --match-last-token --out lm.json
rung plot lm.json --out lm.png && rung summarize lm.json
```

`pip install -e ".[nwb]"` adds h5py for NWB and MATLAB v7.3 files; `".[scripts]"` adds what the analysis scripts need (h5py,
numba, scikit-learn). `rung --version` prints the release; every option's default is in `rung <command> --help`.

## Worked examples

**A recording.** GT3, 4,442 drifting-grating presentations by 11,311 neurons, eight direction classes in angular order:

```bash
rung data gt3_X.npy gt3_labels.npy --sectors --decompose --n-perm 200 --out gt3.json --plot gt3.png
```

![rung's ladder figure for GT3](figures_canonical/worked_example_gt3_rung.png)

The ladder starts low (one class spans few directions) and climbs faster than its floor as each class brings new directions. The
run prints the floor and the shift with its permutation p, the floor's own standard error, the split into the within-class part
D (+0.358) and the class-mean part Tb + Cb + Cx (-0.127), the sign law and the class-mean harmonics. `--account` adds the
two-term account, Figure 3 for this one recording.

**A model.** An axis is a JSON file `{"class": ["prompt", ...], ...}`. `--paper-seeds` reproduces the paper's per-layer cells.

```bash
rung llm --model EleutherAI/pythia-160m --axis axes/world_knowledge32.json --paper-seeds --match-last-token --out lm.json
```

![The language-model battery](figures_canonical/fig_llm_pipeline.png)

Declared topics lift every layer, more with depth; `--match-last-token` adds, at every layer, subsamples matched on the number of
distinct final tokens per class and size-matched random subsamples, so the lexical part of a lift is the drop from the second to
the first. For checkpoints add `--revision step1000`; for a site other than the residual stream, collect one feature vector per
prompt at that site (an MLP's neurons, a sparse autoencoder's latents) and call `zoom(X, labels)` on it.

**Attention heads.** `rung llm --heads` scores every head's write at the final token along the axis:

```bash
rung llm --model openai-community/gpt2 --axis axes/ioi_names.json --heads --paper-seeds \
         --heads-known 9.6,9.9,10.0,10.10,10.6,10.2,10.1,11.2,9.7,9.0,11.9 --out heads.json
```

On GPT-2 small's indirect-object circuit (eight indirect-object names, 16 prompts each), the heads' permutation z ranks the eleven
name-mover and backup heads that path patching found at AUC 0.92 and their class-mean variance share at 0.96, with nothing trained
(a trained probe: 0.95). It is a training-free pre-screen that ranks heads for causal follow-up; the split says whether a head
scores through its class means or its fluctuations.

## Feature reference

Each entry: what it returns, the question it answers, one command, and what the paper found with it.

**`rung data X labels`**: the floor `theta_floor` and its standard error, the shift `delta`, the order-averaged shift, each against
label permutations. Before comparing an exponent across conditions, recordings or models: how much of it is floor.
`rung data X.npy labels.npy --plot r.png`. GT3's three exponents: two are floors (Section 3).

**`--decompose`** (`decompose`): the within-class part D and the class-mean part Tb + Cb + Cx, which sum to the shift; the sign law
(Eq. 3: the full-set class-mean remainder c, negative exactly when PR(W)/PR(B) exceeds 1 + 2(1 - a)/beta, a the means' alignment
with the fluctuations); `--n-bias` the finite-sample bias null. What carries a lift. In imaging the within-class part is +0.29 to
+0.64 and the class-mean part -0.08 to -0.35 (Section 5).

**`--account`** (`account`): the measured within-class part beside the mean-field overlap law ln[K/(1+(K-1) sigma)]/ln K (sigma from
split halves) and the identity at the rungs' sample counts; the class-mean part beside Eq. 3 carried through the rungs; their sum
beside the shift; the two corrections beyond mean field; a held-out arm. Whether two measured terms give the shift, and whether the
classes are sampled enough to read it. The terms give the sign in 8 of 8 imaging recordings and 152 of 167 Allen populations
(Figure 3).

**`--candidates L1.npy L2.npy ...`** (`candidates`): each candidate label's shift against its own permutations, inside the joint cells
of the other candidates with `--candidate-strata others`, ranked by z; `--halves oddeven` discovers on one half and replicates on
the other. Which label organizes a population. Screening four labels picks direction on 16 Allen mice and again on the other 16
(Section 8, Appendix H).

**`rung features X`** (`feature_ladder`): a ladder over features, the nearest neighbours of random anchors (`--positions`) or features
in a given order (`--order`, `--score`), against random features, with the exchangeable control. Whether anatomy or a feature
ranking concentrates dimensions. `rung features X.npy --positions cells.npy`. Neighbouring neurons sit just below their floor
(mean -0.033, negative in 8 of 10 recordings; Appendix D).

**`--block-by phase1|phase2|random|score.npy --block-size 1,4,16,32`** (`blocks`): the shift on blocks of features pooled by a score
(preferred direction, preferred orientation, a file) and on random blocks, each against its own floor. What a coarser recording
keeps. Direction-sorted blocks keep GT3's shift at every block size, random blocks lose it (+0.027 at 32; Appendix G).

**`--sectors`** (`sectors`): the harmonics of the class-mean kernel on a cyclic axis, its even (orientation) and odd (direction)
amplitudes. Which tuning the shift reads (Section 4).

**`--split-by score.npy`** (`split_by`): the top, bottom and a random third of the features at matched size, each against its own
floor. Whether a feature property carries the shift; the most direction-selective third shifts more in all eight recordings (Section 4).

**`--order sequential|antipodal|<list>`, `--antipode pairs.json`**: a declared accumulation order and, with antipodal pairs, the share of the climb left after half the circle. Order
effects and what sets them (Appendix E).

**`--strata`, `--time-blocks E`, `--null-shift`**: nulls that keep a nuisance, an acquisition block or the labels' time order fixed.
Whether a shift is the label or its composition or drift (Appendices I and K).

**`--valid mask.npy`, `--sample-times`, `--invalid-intervals`**: samples a data release marks invalid, dropped before every statistic
(Allen's invalid intervals, Section 6).

**`--cov-floors`** (`cov_floors`): the principal-angle dissimilarity between classes against its label-blind floor, predicted from the
pooled spectrum. A median 69% of that dissimilarity across 167 Allen populations is floor (Section 8, Appendix H).

**`--standardize`**: every feature z-scored first, the repair when one direction dominates the spectrum; `rung data` always prints the
z-scored shift beside the linear one.

**`--paper-floor`**, `--floor-draws`, `--floor-seed`, `--floor-by-rung`, `--rungs`: the floor's draws and the ladder's rungs.
`--paper-floor` draws the floor as the paper's imaging, Allen and macaque generators do and replays their printed values.

**`rung llm`** (`--revision`, `--match-last-token`, `--frames`, `--heads`): every layer's shift in the linear and the z-scored frame
at any checkpoint, the final-token report and control; every attention head with `--heads`. Topics lift every layer of four Pythia
models, and the late lift survives final-token matching (Appendix I).

**`rung axis`, `rung summarize`, `rung plot`**: an axis JSON from a Hugging Face dataset (it reports texts cut at `--max-chars`, which
can make the final token follow the class); a plain-language reading of any result; the figure for any result.

**Loaders**: `.npy`, `.npz:key`, `.mat:variable`, `.csv`, text; `--transpose` when X is stored features by samples.
`rung nwb session.nwb --intervals trials --label direction --out s` writes trials-by-units spike counts, labels and each row's time
window from an NWB 2 file (h5py); `--units-where location=VISp` keeps one area. NWB 1 files (Allen Visual Coding two-photon) read
with h5py directly.

## Reading the numbers

1. **Declare before you look.** Classes, class count and accumulation order are fixed first.
2. **Six or more classes, ten or more samples per rung.** A two-class ladder fits a slope through two points.
3. **For unordered classes, read the order-averaged shift**, and for ordered ones too when the declared shift lies within two SDs
   over random orders of zero or more than two from their average: the declared-order value is a property of one path, and
   `rung summarize` says when.
4. **Pass your nuisance structure as strata.** A shift that survives the within-stratum permutation is the label's; one the
   stratified null absorbs is composition.
5. **Read the spectrum first**, and standardize when one direction dominates.
6. **Compare subsets at matched size only.** The shift depends on the number of features.
7. **Read D and the class-mean part** wherever the within-class scale changes with the class: the raw pooling and trace terms
   each carry that scale, with opposite signs.
8. **Sample each class well before reading the split.** In the paper's grating recordings the within-class part turned positive
   only once each class held 0.46 to 1.2 times its own fluctuation dimension in samples. The floor's first rungs estimate the pooled
   fluctuations, of dimension about m PR_class with m = K/[1+(K-1)σ] (σ the classes' mean fluctuation overlap), from as many
   samples as a declared rung spends on one class, so `D` sits below its population value, by about (m-1) PR_class/(n ln K) at n
   samples per class (an endpoint, first-order estimate), while the class-mean terms (`Tb + Cb + Cx`) move the other way.
   Evaluated at each rung's sample counts (`scripts_canonical/finite_n_identity.py`, `--account`), the identity puts the within-class
   part of eight equal Gaussian classes of PR_class = 30 within 0.05 of the population identity's value on the same rungs once a
   class holds about 20, 8 and 3 times PR_class samples at σ = 0.2, 0.44 and 0.7, at higher ratios n/PR_class for classes of
   lower PR. `--decompose` prints each class's own fluctuation dimension beside PR(W), which pools the classes.
9. **For a depth profile, the inference is on the profile**, the integrated excess of the shift over its embedding value against the
   permutation null; per-layer counts are descriptive.
10. **Report a model's shift in both frames.** Where the linear and the z-scored readings disagree in sign, or only one separates
    from its null, the linear one can follow a few high-variance units.
11. **The shift reads how classes spread.** Classes that differ only by one small offset between near-identical items give it
    nothing to read, while a linear probe can still find that offset.
12. **Any covariance statistic inherits a floor.** Read a principal-angle or other covariance comparison against its label-blind
    floor (`--cov-floors`) before comparing it across designs.
13. **Quote a shift with the floor's standard error.** Two runs with independent floors of the same draw count differ with an SD of
    about 1.4 times it.

## rung against the paper

Run as a user would, on arrays built from the public releases, rung returns the paper's numbers. With the floor the paper's
generators drew (`--paper-floor`: ten draws, seed 42, drawn rung by rung; the lattices' own seeds) it replays the printed shifts and
within-class parts exactly. With its default floor (20 draws) a shift differs from the printed one by the two runs' floor draws,
whose size the spread column shows.

| Dataset | rung, default floor | its spread over 20 seeds | rung, the paper's floor | paper | largest residual |
|---|---|---|---|---|---|
| GT3 direction ladder (Table 9) | +0.231 | +0.234 ± 0.004 | +0.233 | +0.233 | 3e-7 (float32) |
| Allen VISp, one mouse (715093703) | +0.188 | +0.204 ± 0.010 | +0.215 | +0.215 | 2e-15 |
| Allen VISp, TF-residualized arm | +0.226 | +0.239 ± 0.009 | +0.258 | +0.258 | 9e-16 |
| macaque V1, array 1 | +0.153 | +0.153 ± 0.004 | +0.146 | +0.146 | 7e-16 |
| macaque V1, array 2 | -0.353 | -0.355 ± 0.004 | -0.354 | -0.354 | 1e-15 |
| macaque V1, array 3 | +0.249 | +0.248 ± 0.002 | +0.245 | +0.245 | 8e-16 |
| XY nematic lattice, L 16, T 1.0 | -0.537 | -0.538 ± 0.002 | -0.542 | -0.542 | 4e-16 |
| XY lattice, polar side, T 0.7 | +0.445 | +0.438 ± 0.005 | +0.442 | +0.442 | 4e-16 |
| spin waves | +0.449 | +0.448 ± 0.001 | +0.449 | +0.449 | 7e-8 (float32) |

Residuals are the largest of the shift's and the within-class part's; "float32" marks a generator that worked in float32, where
rung computes in float64. Beyond the table: the worked examples of Appendix M return its printed values (GT3 and Pythia-160m,
every row); the candidate screen replays Appendix H's Allen screen exactly (each half's z, p and shift); the feature ladder
replays Appendix D's GT3 row; the blocks replay Appendix G's table to float32 rounding; `--account` on the eight grating
recordings gives the shift's sign in 8 of 8 with r = 0.88 (Figure 3), and the identity at the rungs' sample counts tracks the
held-out within-class part with a mean error of -0.004; `rung llm --heads --paper-seeds` replays Appendix I's ranking of GPT-2's
circuit heads (AUC 0.92), and over ten default seeds the AUC is 0.93 ± 0.01. Generators: `rung_acceptance_replay.py`,
`rung_account_imaging.py`, `rung_heads_ioi.py`.

## What the paper shows

- **There is no probe-free exponent.** The GT3 recording's three exponents split exactly into floor and shift: random stimuli
  0.253 (floor 0.232, shift +0.020), neighbouring neurons 0.346 (floor 0.342), direction 0.305 (floor 0.065, shift +0.240).
  Random stimulus subsets stay at their floors in all 18 recordings (mean shift -0.002).
- **Direction lifts every grating recording**, from 0.20 (low contrast) to 0.45 (localized), above all 200 label permutations;
  classes added in random orders give shifts of the same size, so the grouping carries it. The shift reads the orientation and
  direction sectors of the tuning and is larger in the most direction-selective neurons.
- **What carries it.** Direction-dependent fluctuations around the class means lift the exponent, while in these recordings the
  class means pull it down. Two measured terms give the shift's sign with nothing fitted: one from how much the classes'
  fluctuations overlap, one from how the class means sit against them.
- **Two more laboratories.** In Allen Neuropixels spike counts the shift is positive in 32 of 32 mice in primary visual cortex and
  in 158 of 167 populations across visual areas; the two terms, predicted from each population's counts, give its sign in 152 of
  167. Allen's largest fluctuations are shared, untuned modes that partly track running and pupil; the class-mean term reverses
  there. The same terms predict macaque V1 and lattices of known structure.
- **What the shift adds.** Linear decoders read the class means. With the means projected out they fall to chance while the shift
  survives, and a quadratic decoder given each class's covariance still reads direction (accuracy 0.21 to 0.69; chance 0.125).
  Other covariance statistics inherit floors too: a median 69% of the principal-angle dissimilarity across the 167 Allen
  populations is label-blind floor, which the random-matrix model predicts within a median 2% of the
  permutation floors in imaging and 6% in Allen spike counts.

![The two measured terms against the shift](figures_canonical/fig_twoterm.png)

Every number above is read from its artifact by `tests/test_readme_numbers.py`.

## Runs on data the paper never saw

`rung` was installed from this repository and run on a recording and on model families the paper never used.

**Allen Brain Observatory two-photon imaging** (Visual Coding; 61 primary visual cortex experiments in three excitatory lines;
drifting gratings, eight directions, about 75 presentations each, 86 to 474 cells). The release is NWB 1, read with h5py
(`heldout_allen2p_extract.py` reproduces the release's own per-presentation response); temporal frequency serves as strata.

```bash
rung data X.npy direction.npy --antipode pairs.json --sectors --decompose --cov-floors --strata tf.npy --time-blocks 48
```

- Direction leaves its floor. The order-averaged shift is positive in 48 of 61 experiments and above its label permutations in 49
  (median +0.07), and it stays above the time-blocked null in 47 of those 49.
- The fluctuations lift and the class means pull down, as in the paper's imaging. The within-class part is positive in 47 of 61
  (median +0.21) and the class-mean part is negative in 53 (median -0.07). The class means lie along the strong fluctuations
  (median alignment a = 1.38), and a orders the class-mean part across experiments (Spearman -0.82). One fluctuation shape for every
  class turns the order-averaged shift negative in 56 of 61; equal class means leave it positive in 57.
- The class means are orientation-led in 60 of 61, and a median 68% of the principal-angle dissimilarity is label-blind floor.
- The response definition is part of the probe. The release divides each presentation by its own one-second baseline; without that
  division a falls in 55 of 61 experiments (median 1.01) and the class-mean part turns positive in 25. The sign follows a in both.
- At this design, read the order-averaged shift. With 75 presentations per class and a few hundred cells the classes differ in their
  own fluctuation dimension, and the shift along the declared angular order spans -0.62 to +0.72. Its SD over random orders (median
  0.19) follows that difference (Spearman 0.98 with the spread of single-class log PR) and falls to 0.05 when every class has one
  fluctuation shape. In the Allen Neuropixels populations the same SD is 0.05 (`allen_order_spread.json`).

**Qwen2.5-0.5B** on `axes/world_knowledge32.json` (`rung llm --match-last-token`): the order-averaged shift is positive at all 25
layers, from +0.014 at the embedding to +0.112 at the last layer, and the final-token-matched shift stays positive at every layer of
the second half. **OLMo-2-1B at three training checkpoints** (`--revision`): the second-half matched shift is -0.013 untrained,
+0.032 after 21B tokens and +0.047 at the final checkpoint, where the order-averaged shift is certified at all 17 layers.

**A new axis from a benchmark** (`axes/mmlu_subjects8_answer.json`, built with `rung axis` from `cais/mmlu`, eight subjects of 32
questions). MMLU subjects end on few distinct tokens ('?', ':', '_____.'), and `rung llm` warns that the final token can carry the
shift. Ending every prompt on the standard suffix `\nAnswer:` removes that: the subject shift then stays above its permutations at 22
of the 24 layers past the embedding, at about half its size with the varied endings (mean over those layers +0.094 against +0.193).
Give prompts a common ending whenever the classes differ in format.

## Further analyses in this repository

The repository also carries analyses beyond the camera-ready: the language-model battery at four Pythia sizes and five
architectures, the moral-concept axis before and after instruction tuning, the BLiMP and Baroni minimal pairs read against their
swap nulls, the equivariant network whose invariant layers read zero, and the calibrated population models. Their scripts and
artifacts are listed in the second table below.

## What is where

```
rung.py                  the instrument: zoom(), decompose(), account(), candidates(), feature_ladder(), blocks(), sectors(),
                         cov_floors(), llm_battery(), head_scores(), build_axis(), nwb_counts(), and the rung command line
                         (data | features | nwb | llm | axis | plot | summarize); numpy core, torch and transformers for models
theta_zoom.py            alias of rung (the name through 1.1.0)
tests/                   pytest suite: the decomposition identity, the nulls, the share of the climb left after half the circle, the split at matched size, the
                         sector fit, the declared orders, the validity mask, the floor law, the account, the candidate screen,
                         the feature ladder, the blocks, the loaders, the command line, and every number this README prints;
                         regressions against the paper's artifacts with RUNG_DATA_TESTS=1 (raw data under raw/)
render_all.py            prints the paper's tables and headline numbers from the artifacts
axes/                    every prompt in the paper, with strata sidecars; PROVENANCE.md gives origin and license per file
scripts_canonical/       one script per analysis, named for what it computes; each docstring names the artifact it writes;
                         make_fig_*.py regenerate the figures from the artifacts
data_canonical/          the result files behind every reported number
figures_canonical/       the figures (drawn with matplotlib 3.10; later releases render text a few pixels differently)
pyproject.toml           pip install -e . gives the rung command (and theta-zoom as an alias)
```

Raw neural data are public and not included. The grating and static recordings are the Figshare release 8279387 of Stringer et
al. (2021, Cell); the spontaneous sessions the release of Stringer et al. (2019, Science); the Allen Brain Observatory Neuropixels
visual-coding sessions; the CRCNS pvc-11 macaque arrays (Kohn and Smith 2016, doi:10.6080/K0NC5Z4X). The scripts read `raw/`:
each grating or static session as `raw/<session>.npy` (a dict with `sresp`, neurons by presentations, `istim`, the stimulus
angle, and `stat`, cell positions), spontaneous sessions under `raw/spont/`, the pvc-11 arrays under
`raw/pvc11/spikes_gratings/`, and the per-session Allen spike counts and running speeds under `raw/allen_counts/` and
`raw/allen_running/` (written from the NWB files by `allen_counts_local.py`, or on a cloud runner by `allen_counts_modal.py` and
`allen_running_modal.py`). Language-model hidden states go to `raw/llm_states/` (`llm_battery.py encode`, `llm_encode.py`;
several GB). Each docstring names the files it reads.

### Allen invalid intervals

Every Allen number excludes what the release itself marks invalid: presentations that overlap an `invalid_times` interval tagged
with the population's probe (the probe sent no data), `all_probes` or `stimulus`; units on electrodes with `valid_data` False; and
presentations in which a population whose valid-time rate predicts 10 or more spikes fires none (one population, session
743475441 VISp, is silent before 2135 s without a flag). `allen_fetch_nwb.sh` fetches the 32 session NWBs into `raw/allen_nwb/`;
`allen_invalid_map.py` writes the per-population masks to `data_canonical/allen_masks/` (shipped). `allen_engine.pop_view` takes
every Allen generator's presentations and units through the masks; `ALLEN_VARIANT=none` reproduces the unmasked values, `A` uses
the release's flags alone, `B` (the default) is the paper. The `allen_*_local.py` scripts run the cloud-run Allen generators on the
local counts through the same path; unmasked, they reproduce the cloud outputs to 5e-7. `rung --invalid-intervals` applies the same
exclusion to any recording.

The Allen two-term predictions in `allen_twoterm_predictions.json` were written by `allen_twoterm_predict.py` and committed before
the measured Allen results were compared with them; the two-term figure (`make_fig_twoterm.py`) shows both.

### Claim, artifact, script: the paper

| Claim | Artifact | Script |
|---|---|---|
| Random-stimulus ladders sit at their floors in 18 recordings; the floor follows the sampling crossover | stringer_theta_decomposition.json, withinclass_shuffle_identification.json, floor_crossover.json | random_zoom_all_recordings.py, withinclass_shuffle_identification.py, floor_crossover.py |
| Direction lifts every grating recording, above all 200 label permutations | run38_v1_label_permutations.json, orientation_zoom.json | run38_v1_label_permutations.py, shuffle_label_control.py |
| The order of the classes moves the shift little; the order effect over sixteen origins and its two parts | cr_order_effect_origins.json, fluctuation_overlap_profile.json | cr_order_effect_origins.py, fluctuation_overlap_profile.py, cr_appE_class_means.py |
| Neighbouring neurons stay near their floor | neuron_zoom_spatial.json | spatial_neuron_ladder.py, exchangeable_neuron_zoom.py |
| The two symmetry sectors; the most direction-selective third shifts more, also cross-fitted | multipole_harmonics_8dir.json, run59_shift_by_direction_selectivity.json, cr_ds_thirds_crossfit.json | multipole_harmonics_8dir.py, run59_shift_by_direction_selectivity.py, cr_ds_thirds_crossfit.py, make_fig_multipole.py |
| The exact split of the shift into a within-class and a class-mean part | run72_paper_designs_split.json | run72_paper_designs_split.py |
| What carries the shift: within-class shuffle, pooled fluctuations, tuning inside each bin removed | cr_withinclass_controls.json, cr_order_and_withinbin_controls.json, withinclass_shuffle_identification.json | cr_withinclass_controls.py, cr_order_and_withinbin_controls.py, withinclass_shuffle_identification.py |
| The overlap law and the class-mean term; the sign law and the alignment a | overlap_identity_checks.json, overlap_identity_followup.json, classmean_sign_condition.json, classmean_fluctuation_alignment.json, signal_noise_alignment.json | overlap_identity_checks.py, overlap_identity_followup.py, classmean_sign_condition.py, classmean_fluctuation_alignment.py, signal_noise_alignment.py, make_fig_overlap_laws.py |
| The two terms against the shift: imaging, Allen, macaque, lattices | twoterm_imaging.json, allen_twoterm_predictions.json, allen_sessions/session_*.json, macaque_pvc11_twoterm.json, xy_twoterm.json, xy_polar_twoterm.json, fig_twoterm_marker.json | make_fig_twoterm.py, twoterm_imaging.py, allen_twoterm_predict.py, allen_run.py, macaque_pvc11_twoterm.py, xy_twoterm.py, xy_polar_twoterm.py, fig_twoterm_marker.py |
| Beyond mean field: the overlap identity at the rungs' sample counts (imaging, lattices, the sampling rule) | finite_n_imaging.json, finite_n_lattice.json, finite_n_rule.json | finite_n_identity.py, finite_n_imaging.py, finite_n_lattice_regen.py, finite_n_rule.py |
| Allen with temporal frequency separated: per-session shifts, splits, controls, verdicts and misses | allen_sessions/session_*.json, allen_verdicts.json, allen_twoterm_misses.json, allen_miss_threshold.json | allen_run.py, allen_engine.py, allen_score.py, allen_twoterm_misses.py, allen_miss_threshold.py |
| Allen's state modes held out, running, pupil and lagged state | allen_heldout_modes.json, allen_heldout_modes_mechanism.json, allen_heldout_modes_control.json, allen_modes_removed.json, allen_modes_running.json, allen_modes_pupil.json, allen_modes_state_basis.json | allen_heldout_modes.py, allen_heldout_modes_mechanism.py, allen_heldout_modes_control.py, allen_modes_removed.py, allen_modes_running.py, allen_modes_pupil.py, allen_modes_state_basis.py |
| Choosing an axis among candidates on half the Allen mice | allen_axis_screen.json, allen_axis_screen_rescore.json, allen_axes_halves.json | allen_axis_screen.py, allen_axis_screen_rescore.py, allen_axes_halves.py |
| The Allen invalid-interval masks | allen_masks/, allen_invalid_map.json | allen_invalid_map.py, allen_fetch_nwb.sh, allen_engine.py |
| Macaque V1, three Utah arrays | macaque_pvc11.json, macaque_pvc11_twoterm.json | macaque_pvc11.py, macaque_pvc11_twoterm.py |
| What the shift adds: decoders, the means projected out, decoding against the means' strength | meanfree_shift_exact.json, meanfree_shift.json, meanfree_crossfit.json, decoding_vs_beta.json, v1_decoders.json | meanfree_shift_exact.py, meanfree_shift.py, meanfree_crossfit.py, decoding_vs_beta.py, v1_decoders.py |
| Floors under other covariance statistics, and the floor law | allen_covariance_floors_decoders.json, covariance_floors_trial_ladder.json, principal_angle_floor_imaging.json, principal_angle_floor_allen.json | allen_covariance_floors_decoders.py, covariance_floors_trial_ladder.py, principal_angle_floor_law.py, principal_angle_floor_imaging.py, principal_angle_floor_allen.py |
| Lattices and constructed systems of known structure | lattice_ground_truth_score.json, xy_groundtruth.json, xy_onset.json, ising_spectral_account.json | lattice_ground_truth.py, lattice_ground_truth_score.py, xy_groundtruth.py, xy_onset.py, xy_split.py, ising_spectral_account.py |
| Language models: the topic and construction lifts at four Pythia sizes, last-token controls, probes, sparse autoencoders, a known circuit, other architectures | llm_battery_*.json, llm_lasttoken_*.json, llm_lasttoken_fractions.json, llm_probes_*.json, llm_sae.json, llm_ioi.json, llm_architectures_*.json, llm_path_nulls.json | llm_battery.py, llm_lasttoken.py, llm_lasttoken_fractions.py, llm_probes.py, llm_sae.py, llm_ioi.py, llm_architectures.py, llm_path_nulls.py, llm_score.py |
| Worked examples (`rung` on GT3 and on Pythia-160m) | worked_example_gt3.json, appm_lm_check.json | worked_example_lm_check.py |
| Runs on data the paper never saw: Allen two-photon (release response and no per-presentation baseline), Section 5's controls, Qwen2.5-0.5B, OLMo-2-1B checkpoints, MMLU subjects | heldout_allen2p/, heldout_allen2p_nobaseline/, heldout_allen2p_controls.json, heldout_allen2p_summary.json, heldout_allen2p_experiments.json, heldout_qwen05_*.json, heldout_olmo2_1b_*.json | heldout_allen2p_extract.py, heldout_allen2p_controls.py, heldout_allen2p_score.py |
| Allen Neuropixels: the declared-order shift against the order-averaged one and its SD over orders | allen_order_spread.json | allen_order_spread.py |
| Allen's two-term misses against the finite-sample identity: the within-class part of all 167 populations, the shift's sign in 163 | allen_finite_n_identity.json | allen_finite_n_identity.py |
| rung against the paper: replays of the paper's floors, the account on eight recordings, the attention heads | rung_acceptance_replay.json, rung_account_imaging.json, rung_heads_ioi.json | rung_acceptance_replay.py, rung_account_imaging.py, rung_heads_ioi.py, make_axis_ioi.py |
| Appendix checks: coarse-graining, finite size, synthetic tuning width, the exact floor inverse | cr_graining_table.json, cr_finite_size_split.json, kappa_sweep.json, appendix_exception_tests.json | cr_graining_table.py, cr_finite_size_split.py, kappa_sweep.py, appendix_exception_tests.py |

### Claim, artifact, script: further analyses

| Analysis | Artifact | Script |
|---|---|---|
| Ladder-design variants; the axis search on static, natural-image and spontaneous sessions | run8_ladder_robustness.json, run27_static_axis_search.json, run40_spont_state_axis.json | run8_ladder_robustness.py, run27_static_axis_search.py, run40_spont_state_axis.py |
| The accumulation-order contrast; the sector balance under blocking; anatomical blocks | antipodal_order.json, sector_balance_scale.json, run51_spatial_blocking.json | accumulation_order.py, sector_balance_scale.py, run51_spatial_blocking.py |
| The second laboratory's even-sector correlate | run1_allen_partial_mixed.json, run39_allen_aeven_mixed.json | run1_allen_partial_mixed.py, run39_allen_aeven_mixed.py |
| Ising and nematic lattices; the two-species lattice | ising_L128_tc.json, nematic_polar_delta.json, run62_two_species_xy.json | ising_L128_tc.py, nematic_polar_delta.py, run62_two_species_xy.py |
| The equivariant network: the invariant zero, the sector content, the reversed order | run5b_cnn_seeds.json, run5c_cnn_multipole_fixed.json, run12_cnn_ordering.json, run41_cnn_ordering_perseed.json | run5b_cnn_seeds.py, run5c_cnn_multipole_fixed.py, run12_cnn_ordering.py, run41_cnn_ordering_perseed.py |
| The calibrated and cell-built population models | run2b_corotating_seeds.json, run11b_fresh_draw_prediction.json, run60_mixture_model.json, cr_model_overlap_calibration.json, cr_model_overlap_square.json | run2b_corotating_seeds.py, run11_bootstrap_prediction.py, run60_mixture_model.py, cr_model_overlap_calibration.py, cr_model_overlap_square.py |
| The deficit split in the network, the second laboratory and the language models | run63_deficit_decomposition.json, run67_cnn_deficit_decomposition.json, run66_allen_deficit_decomposition.json, run65_pythia_deficit_decomposition.json | run63_deficit_decomposition.py, run67_cnn_deficit_decomposition.py, run66_allen_deficit_decomposition_modal.py, run65_pythia_deficit_decomposition.py |
| The language-model battery's robustness, the moral-concept axis, architectures, the blind probe, planted axes, minimal pairs | run20_robustness_battery.json, run44_base_instruct_ethical.json, run47_fourth_cell_redpajama.json, run53_mamba_fifth_cell.json, run55_blind_probe_physics.json, run28_cyclic_axis_llm.json, run42_blimp_battery.json, run43b_baroni_64pairs.json | run20_robustness_battery.py, run44_base_instruct_ethical.py, run47_fourth_cell_redpajama.py, run53_mamba_fifth_cell.py, run55_blind_probe_physics.py, run28_cyclic_axis_llm.py, run42_blimp_battery.py, run43b_baroni_64pairs.py |

## Citation

```bibtex
@inproceedings{amin2026zoom,
  author    = {Amin, Adil},
  title     = {Structure is in the zoom: probing neural symmetry through
               dimensionality scaling},
  booktitle = {Proceedings of the NeurReps Workshop (Symmetry and Geometry in Neural Representations)},
  series    = {Proceedings of Machine Learning Research},
  year      = {2026},
  note      = {Code and data: https://github.com/adilamin89/structure-in-the-zoom}
}
```

## License

Code: MIT (see `LICENSE`). Result artifacts in `data_canonical/` may be reused with attribution to the paper. Prompts and stimuli
carry their own terms: the three axes composed for the paper are CC BY 4.0; the benchmark-derived axes and stored stimuli inherit
their sources' licenses (TruthfulQA Apache-2.0, HellaSwag MIT, ARC CC BY-SA 4.0, ETHICS MIT, BLiMP CC BY 4.0, Baroni et al. per
their release). Origins, authorship and licenses are listed item by item in `axes/PROVENANCE.md`.
