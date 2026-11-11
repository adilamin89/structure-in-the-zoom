# v3: generators behind the extended version's new numbers

The extended version of the paper (arXiv 2611.08506, version 3) adds results to the NeurReps version. Each new number comes from a
script in this folder, written beside it as JSON (and a log where the run printed one). The rest come from `scripts_canonical/` and
`data_canonical/`, as listed in [PAPER.md](../PAPER.md).

| Script | What it computes | In the extended version | Needs |
|---|---|---|---|
| `start_probe_all8.py` | every start and sense of the angular order (16 per recording), and the declared and order-averaged shift at eight wall positions, in all eight grating recordings | Section 3 (128 of 128 starts positive; GT3 +0.231 near the bottom of +0.229 to +0.465), Appendix C's table of starts | `raw/` gratings |
| `lm_split_orderavg.py` | Pythia-2.8B's topic split per layer and frame, averaged over 20 random class orders, each net of its own ten shape surrogates | Section 9 (net within-class part +0.030 to +0.067, 33 of 33 layers), Appendix I | LM states |
| `lm_split_bias.py` | the same split in the declared order: raw, net of the surrogates, and in excess of label shuffles | Appendix I's table, declared-order columns | LM states |
| `lm_pooled_control_recentered.py` | each topic's mean kept, the fluctuations pooled across topics and re-centred within each, declared and order-averaged | Section 9 and Figure 8B (+0.021 to +0.093 falls to -0.006 to +0.019) | LM states |
| `nongauss_floor_check.py` | Eq. 2 for samples of any distribution, against simulation | Appendix M | nothing |
| `appm_rerun_1131.py` | CPU time of the worked example's two commands, one core each | Appendix N (69 and 854 CPU-seconds on rung 1.13.1) | `raw/` GT3, `rung[models]` |
| `fl_one_half.py` | the one half that missed the Freedman-Lane null at 50 draws, re-read at 500 per null | Section 5 (p = 0.04, and 0.08 within contiguous epochs) | `raw/` low-contrast GT1 |
| `a3_macaque_finite_n.py` | the two terms at the macaque arrays' own sample counts | Section 7 (-0.346 against the measured -0.357) | `raw/pvc11/` |
| `g1_invariance.py` (with `g1_methods.py`) | what label CKA, RSA, a probe and mean-field capacity are blind to, on a planted design | Section 10 and Appendix L (within-class part -0.034 to +0.051 under a rotation that leaves mean-field capacity fixed; simulated capacity 0.37 to 0.19) | scikit-learn |

The 31 of 32 halves that clear the Freedman-Lane null (Section 5) are `data_canonical/meanfree_freedman_lane.json`, from
`scripts_canonical/meanfree_freedman_lane.py`.

**Data.** `raw/` is laid out as [PAPER.md](../PAPER.md) describes (the grating recordings as `raw/<session>.npy`, the macaque arrays
under `raw/pvc11/spikes_gratings/`). The language-model scripts read Pythia-2.8B-deduped's last-token states on the topic axis from
`raw/llm_states/pythia-2.8b-deduped__main__topics.npz`; write them once with
`python scripts_canonical/lm_shape_surrogate.py --device mps --save-states raw/llm_states/pythia-2.8b-deduped__main__topics.npz`
(the model extras and the 5.3 GB model; fp16 states move in the last digits between devices).

**Versions.** These numbers were computed with rung 1.13.0 and 1.13.1. The scripts import this repository's `rung.py` and pass the
options that reproduce those runs under rung 1.14: `order_average="random"` where a shift is averaged over class orders (1.14's
default for up to ten classes is the exact average over all orders) and `n_bias=0` where a split is read raw. `appm_rerun_1131.py`
times whichever rung is installed; install the tag `v1.13.1` to repeat Appendix N's timing.

**Run** from the repository root, e.g. `python v3/lm_split_orderavg.py`. Each script's docstring gives its arguments and output.
