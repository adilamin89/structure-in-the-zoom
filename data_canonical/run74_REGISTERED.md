# run74 registration (2026-10-05, before the run): 200 label permutations per Allen population
Why: reviewer tYDC: "three to five permutations used in the main analyses are insufficient for conventional
per-population permutation significance". The submitted paper used three per Allen population.
Design: run66's spike-count construction (drifting gratings, eight directions, every VIS area with >= 50 units, 167
session-area populations); the paper's direction ladder (rungs 1,2,3,4,6,8 classes); floor = mean log PR of 10 random
same-size subsets (unchanged under label permutation, which preserves rung sizes); 200 label permutations at fixed class
sizes; one-sided p = (1 + #{permuted delta >= observed}) / 201. PR through the double-centred Gram sub-blocks.
R1 (reproduction): median |recomputed delta - stored delta_dir8| < 0.02.
P1: VISp: p <= 0.05 in >= 30 of 32 sessions.
P2: all areas: p <= 0.05 in >= 140 of 167 populations.

## Amendment before the full run (2026-10-05)
A one-session smoke test (715093703) gave shifts 0.01-0.06 below the stored values. A floor-seed check on that session
(8 seeds of the 10-draw floor) shows floor-draw noise SD 0.010-0.026, with the stored values inside the seed range in 4
of 5 areas (VISl 3.6 SD above the seed mean). So the computation matches and the 0.02 reproduction bar was tighter than
the paper's own floor noise. Amended before any full-run result: N_NULL 10 -> 50 (stabler floor); R1 bar 0.02 -> 0.04
(twice the 10-draw floor SD). P1 and P2 unchanged. The smoke-test result file on the volume was written with N_NULL = 10
and is deleted before the full run so every population uses the amended design.

## Verdict (2026-10-05, run74_allen_permutations.json; Modal adila-75793, about 6 min)
R1 PASS (median |recomputed - stored| = 0.009 with the 50-draw floor). P1 PASS: VISp p <= 0.05 in 30/32 sessions (26 at
the minimum p = 1/201). P2 MISS: 112/167 populations at p <= 0.05 (registered >= 140). δ positive in 157/167.
Per area (p <= 0.05): VISp 30/32, VISal 18/24, VISpm 15/20, VISrl 17/27, VISam 14/26, VISl 12/24, VIS 2/8, small areas
4/6. Failure interrogation: per-population power is limited (about 75 trials per direction, 50-300 units; permutation
SD 0.04-0.12); significance follows the paper's area gradient, strongest in VISp. Camera-ready statement: VISp positive
32/32 and significant in 30/32 against 200 label permutations; across areas positive 157/167, significant 112/167,
strongest in VISp; the cross-area inference rests on the sign and on the animal level.
