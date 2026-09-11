"""The exact four-term split of a ladder rung's deficit (run 63, S78b), as a module for runs 65-68.

Since rung 1.4.0 the implementation lives in arxiv_supplement/rung.py (`rung data ... --decompose`; `decompose()`
in Python); this module re-exports it so that the runs and the tool are one implementation. The physics, briefly:
for trials Y with class labels, C = B + W and log PR = 2 log(b1 + w1) - log(b2 + w2 + 2x); the deficit of a rung
below its floor splits exactly into A (trace), P (fluctuation pooling), Cb (between-class) and Cx (coupling); A and
P each carry the within-class scale S = 2 Delta log w1 with opposite signs and regroup as A + P = D + Tb (the pooled
within-class effective dimension; the trace share of the class means). The private slope is the slope of P (or D)
against log(k/K) over the rungs below the top one (App I's 1.03 for drifting GT1 is this fit). Label permutations
(n_shuffle) give each term's sampling baseline; `excess` is observed minus that mean.
"""
import os
import sys

try:
    import rung
except ImportError:  # the supplement is not installed: use the repository copy
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "arxiv_supplement"))
    import rung

FREE = rung.FREE
traces = rung.traces
logterms = rung.logterms
pr_centered = rung.pr_centered
decompose = rung.decompose
largest_term = rung.largest_term
_slope = rung._lsq_slope
