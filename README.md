# Structure is in the zoom

[![Open the quickstart in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adilamin89/structure-in-the-zoom/blob/main/examples/quickstart.ipynb)

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
sets that share. The grouping of samples into classes carries the shift. How much the order in which the classes enter moves it
depends on the design: little in the paper's grating recordings, more than the shift itself where classes differ in their own
fluctuation dimension and hold few samples (the Allen two-photon runs in [PAPER.md](PAPER.md)). `rung` reports the order-averaged shift and its SD over
orders beside the declared one, and `rung summarize` says which to read (rule 3).

## Install and run

```bash
pip install git+https://github.com/adilamin89/structure-in-the-zoom   # numpy, scipy, matplotlib
# X.npy: any samples-by-features array; labels.npy: one integer label per row
rung data X.npy labels.npy --decompose --out r.json --plot r.png
rung summarize r.json                                                  # the reading, in plain language
rung report r.json                                                     # r.html: figure, reading, numbers
```

[examples/quickstart.ipynb](examples/quickstart.ipynb) runs the toy below and holds the commands for a recording and a language
model; it renders on GitHub and opens in Jupyter, VS Code or Colab (the badge above).

The same from Python:

```python
import numpy as np
from rung import zoom, decompose, summarize_data

X, labels = np.load("X.npy"), np.load("labels.npy")
r = zoom(X, labels)                      # floor, shift, permutation p, order-averaged shift
r["decompose"] = decompose(X, labels)    # within-class and class-mean parts
summarize_data(r)                        # the plain-language reading
```

No data at hand? A toy population in which each of eight classes triples the variance of its own four features, so the classes
differ in their fluctuations and not in their means:

```bash
python - <<'EOF'
import numpy as np
rng = np.random.default_rng(0)
X = rng.standard_normal((512, 64))                     # 8 classes x 64 samples, 64 features
y = np.repeat(np.arange(8), 64)
X *= 1 + 2 * (y[:, None] == np.arange(64)[None, :] // 4)   # class c triples features 4c to 4c + 3
np.save("X.npy", X); np.save("labels.npy", y)
EOF
rung data X.npy labels.npy --decompose --out toy.json --plot toy.png && rung summarize toy.json
```

The shift comes out positive and above every label permutation, and the split puts nearly all of it in the within-class part,
as built: the classes differ only in their fluctuations.

**Permutation budget.** The defaults (500 label permutations, 50 random class orders, 20 floor draws) suit a few hundred samples.
At recording scale they take long: Appendix M's GT3 command, with 200 permutations on 4,442 samples by 11,311 features, takes
about 70 minutes on one laptop core, most of it the permutations and the floors of the random orders. For a first look there,
add `--n-perm 100 --k-orders 10 --floor-draws 5` and rerun with the defaults for the numbers you report. A
permutation p cannot fall below 1/(n_perm + 1): with fewer than 20 permutations no shift can reach p < 0.05, and `rung summarize`
calls such a reading unresolved, never unseparated.

For a language model, clone the repository (the paper's prompt axes are in `axes/`) and add the model extras:

```bash
git clone https://github.com/adilamin89/structure-in-the-zoom && cd structure-in-the-zoom
pip install -e ".[models]"                       # + torch, transformers, datasets
rung llm --model EleutherAI/pythia-160m --axis axes/world_knowledge32.json \
         --match-last-token --out lm.json
rung plot lm.json --out lm.png && rung summarize lm.json
```

The final-token arms of `--match-last-token` take about three quarters of that run (Appendix M: about 19 minutes on one laptop
core). Drop the flag for a first look at every layer's shift in both frames, and add it back before reading a lift as more than
its final tokens.

`pip install -e ".[nwb]"` adds h5py for NWB and MATLAB v7.3 files; `".[scripts]"` adds what the analysis scripts need (h5py,
numba, scikit-learn). `rung --version` prints the release; every option's default is in `rung <command> --help`.

## Worked examples

**A recording.** GT3, 4,442 drifting-grating presentations by 11,311 neurons, eight direction classes in angular order. The session
is `gratings_drifting_GT3_2019_04_05_1.npy` in the Figshare release 8279387 (Stringer et al. 2021; 489 MB), a dict with `sresp`
(neurons by presentations) and `istim` (the drift direction in radians):

```bash
curl -L -o gratings_drifting_GT3_2019_04_05_1.npy https://ndownloader.figshare.com/files/15479240
python - <<'EOF'
import numpy as np
d = np.load("gratings_drifting_GT3_2019_04_05_1.npy", allow_pickle=True).item()
np.save("gt3_X.npy", np.asarray(d["sresp"], np.float32).T)       # presentations by neurons
phi = np.asarray(d["istim"], float) % (2 * np.pi)                # drift direction in radians
np.save("gt3_labels.npy", np.clip(np.digitize(phi, np.linspace(0, 2 * np.pi, 9)) - 1, 0, 7))
EOF
rung data gt3_X.npy gt3_labels.npy --sectors --decompose --n-perm 200 --out gt3.json --plot gt3.png
```

The Python step writes presentations by neurons and eight 45-degree direction classes, the input of Appendix M's worked example.

<img src="figures_canonical/worked_example_gt3_rung.png" alt="rung's ladder figure for GT3" width="420">

The ladder starts low (one class spans few directions) and climbs faster than its floor as each class brings new directions. The
run prints the floor and the shift with its permutation p, the floor's own standard error, the split into the within-class part
D (+0.358) and the class-mean part Tb + Cb + Cx (-0.127), the sign law and the class-mean harmonics. `--account` adds the
two-term account, Figure 3 for this one recording.

**A model.** An axis is a JSON file `{"class": ["prompt", ...], ...}`. `--paper-seeds` reproduces the paper's per-layer cells.

```bash
rung llm --model EleutherAI/pythia-160m --axis axes/world_knowledge32.json \
         --paper-seeds --match-last-token --out lm.json
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
classes are sampled enough to read it. The terms give the sign in 8 of 8 imaging recordings (Figure 3). For Allen's populations,
whose responses are first residualized on temporal frequency, `allen_finite_n_identity.py` evaluates the identity through that
linear step; `--account` evaluates it for the array as given.

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

**`rung axis`, `rung summarize`, `rung plot`, `rung report`**: an axis JSON from a Hugging Face dataset (it reports texts cut at
`--max-chars`, which can make the final token follow the class); a plain-language reading of any result; the figure for any result;
one self-contained HTML page for any result (the figure, the reading, the numbers a reader quotes with their settings and, for
`rung data`, the shift against its nulls, its split and the harmonics), to share or archive beside the JSON.

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

## The paper

[PAPER.md](PAPER.md) holds what the paper shows, rung run against the paper's own numbers, runs on data the paper never saw,
the public raw data and how the scripts read them, and a claim-by-claim table of artifacts and the scripts that write them.

## Repository

```
rung.py                  the instrument: zoom(), decompose(), account(), candidates(),
                         feature_ladder(), blocks(), sectors(), cov_floors(), llm_battery(),
                         head_scores(), build_axis(), nwb_counts(), report(), and the rung
                         command line (data | features | nwb | llm | axis | plot | summarize |
                         report); numpy core, torch and transformers for models
theta_zoom.py            alias of rung (the name through 1.1.0)
tests/                   pytest suite: the decomposition identity, the nulls, the share of the
                         climb left after half the circle, the split at matched size, the
                         sector fit, the declared orders, the validity mask, the floor law,
                         the account, the candidate screen, the feature ladder, the blocks,
                         the loaders, the command line, and every number README.md and
                         PAPER.md print; regressions against the paper's artifacts with
                         RUNG_DATA_TESTS=1 (raw data under raw/)
render_all.py            prints the paper's tables and headline numbers from the artifacts
axes/                    every prompt in the paper, with strata sidecars; PROVENANCE.md gives
                         origin and license per file
scripts_canonical/       one script per analysis, named for what it computes; each docstring
                         names the artifact it writes; make_fig_*.py regenerate the figures
                         from the artifacts
data_canonical/          the result files behind every reported number
figures_canonical/       the figures (drawn with matplotlib 3.10; later releases render text a
                         few pixels differently)
pyproject.toml           pip install -e . gives the rung command (and theta-zoom as an alias)
```

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
