# run73 registration (2026-10-05, before the run): m-out-of-n bootstrap of the direction shift on all eight recordings
Why: reviewer tYDC noted that low-contrast GT2's point estimate (0.209) lies below its with-replacement bootstrap
interval [0.237, 0.318]. The arXiv version diagnosed this as diversity loss from resampling with replacement and checked
it on GT3 only (m = 0.8n per class without replacement, 100 replicates: centre +0.227, interval [0.203, 0.245], point
+0.237). This run extends the check to all eight grating recordings.
Design: all neurons; the paper's direction ladder (8 bins, rungs 1,2,3,4,6,8 classes); per replicate, 80 percent of each
class's trials drawn without replacement, δ = slope(ladder) − slope(mean log PR of 10 random same-size subsets); 100
replicates per recording; percentile 95 percent intervals.
P1: the full-data point estimate lies inside the m-out-of-n 95 percent interval in 8/8.
P2: the m-out-of-n centre is within 0.03 of the point estimate in >= 7/8.

## Verdict (2026-10-05, run73_m_out_of_n_all.json, 7388 s local)
P1 PASS 8/8 (point inside the m-out-of-n 95% interval), P2 PASS 8/8 (centre within 0.03). Low-contrast GT2 (tYDC's
case): point +0.210, interval [+0.188, +0.218]. Note for the text: in L1 and L2 the point sits at the upper bound
(+0.456 vs +0.457; +0.396 vs +0.396) and centres run 0.01-0.02 below the point in L1-L3 and C1-C2, the expected small
downward bias of 80-percent subsamples (fewer trials per rung). Camera-ready: replace the with-replacement intervals
of Table 2 / Figure 2 with these, labelled "m-out-of-n (80% of trials per class, without replacement, 100 replicates)".
