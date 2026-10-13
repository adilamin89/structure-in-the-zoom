# run72 registration (written 2026-10-01 before the run; supersedes run71's ordering analysis, see AUDIT_OCT1_RUNS.md)
Rerun the paper's own designs, exactly, for two populations (all neurons, as the paper; tuned neurons, ANOVA p<0.01 on 8
direction bins, as run63/70), and add the noise-side / probe split per rung.
Designs: (A) direction ladder, 8 bins, rungs 1,2,3,4,6,8 classes, 10-draw random floor (orientation_zoom / run63);
(B) accumulation_order.py: 16 bins of 22.5 deg, 8 rungs of 2 bins, paired (k, k+8) vs sequential (2k, 2k+1), a 10-draw
floor per ladder at that ladder's sizes. Classes for the split = the design's bins (8 in A, 16 in B).
R1 (reproduction, all neurons): A's delta within 0.03 of orientation_zoom.json in 8/8; B's order effect (delta_paired -
   delta_sequential) negative in 8/8 and within 0.03 of neurreps_supplement/data/antipodal_order.json in >= 7/8.
P1 (split, design A, both populations): noise-side slope > delta and probe-part slope < 0 in >= 7/8 per population.
P2 (design B, which part carries the order effect): descriptive, both populations; report the paired-minus-sequential
   slope difference for the noise-side and probe parts.
P3 (instrument function on the paired design, same population): from the 16-bin class means of the SAME population and
   covariance, per-harmonic powers |c_l|^2 and f_S(l) = 1 - |mean_{j in S} e^{i l theta_j}|^2 predict the sign of
   Tr B_sequential - Tr B_paired at rungs 1-4 in >= 26/32 cells per population (4 rungs x 8 recordings).
Misses reported as misses; failure-interrogation questions apply.

## Verdict (2026-10-01, after the run; run72_paper_designs_split.json, 780 s)
R1 PASS: direction delta within 0.03 of orientation_zoom 8/8 (largest gap 0.013, C1); order effect negative 8/8 and
within 0.03 of the supplement 8/8 (largest gap 0.018, C1). The paper's numbers reproduce through an independent code path.
P1 PASS 8/8 all neurons and 8/8 tuned: noise-side slope > delta, probe-part slope < 0. ("Tuned" is 85-97 percent of
all neurons in these recordings, so the population choice barely matters here.)
P3 PASS 29/32 per population: the instrument function from the same population's 16-bin harmonic powers predicts the
sign of Tr B (sequential - paired) per rung.
P2 (descriptive): the order effect (paired - sequential) splits into noise -0.02 to -0.19 (negative 8/8) and probe
-0.07 to +0.14 (positive in D1, D2, L1, L2, L3; about zero in C1, C2; negative in D3). The noise side carries the
reversal in 6/8; the probe part alone goes the way the original registered prediction said (paired faster) in 5/8.
At rung 1, Tr B is larger for the antipodal pair than the adjacent pair in 7/8, as entry coherence says.
Reading: the paper's entry-coherence sentence is right for the class means and incomplete for the slope; in V1 the
first classes are alike in their fluctuations more than in their means. Text change proposed for arXiv Sec 4.1 and
the camera-ready (one clause); the CNN ordering (predicted from harmonics, 5/5) is unaffected.
