"""Allen Neuropixels theta decomposition: within-area vs cross-area.
Run on Colab with A100 (needs allensdk + ~5GB download per session).

Prediction: within-area neuron zoom gives δ < 0 (local redundancy),
cross-area pooled neuron zoom gives δ > 0 (distinct representations).
Tests Dahmen 2022's "local control of dimensionality" through our lens.

Usage on Colab:
    !pip install allensdk -q
    %run run_allen_neuropixels_delta.py
"""
import numpy as np
import json
import os

def pr_fast(X):
    """Participation ratio from a (samples x features) matrix."""
    Xc = X - X.mean(axis=0)
    n = Xc.shape[0]
    if n < 3:
        return 1.0
    G = Xc @ Xc.T / (n - 1)
    lam = np.linalg.eigvalsh(G)
    lam = lam[lam > 1e-10]
    if len(lam) == 0:
        return 1.0
    return float((lam.sum())**2 / (lam**2).sum())

def fit_theta(sizes, prs):
    """Log-log slope of PR vs size."""
    x = np.log(np.array(sizes, dtype=float))
    y = np.log(np.array([max(p, 1e-9) for p in prs], dtype=float))
    A = np.vstack([np.ones(len(x)), x]).T
    return float(np.linalg.lstsq(A, y, rcond=None)[0][1])

def neuron_zoom_delta(rates, rng, n_null=10):
    """Compute delta for a neuron-subsampling zoom on a (trials x neurons) matrix."""
    n_trials, n_units = rates.shape
    sizes = sorted(set([s for s in [10, 20, 50, 100, 200, 500, 1000] if s < n_units]))
    sizes.append(n_units)
    if len(sizes) < 3:
        return None

    obs_prs, null_prs = [], []
    for ns in sizes:
        if ns >= n_units:
            obs_prs.append(pr_fast(rates))
            null_prs.append(pr_fast(rates))
        else:
            obs_prs.append(np.mean([
                pr_fast(rates[:, rng.choice(n_units, ns, replace=False)])
                for _ in range(5)
            ]))
            null_prs.append(np.mean([
                pr_fast(rates[:, rng.choice(n_units, ns, replace=False)])
                for _ in range(n_null)
            ]))

    theta_obs = fit_theta(sizes, obs_prs)
    theta_floor = fit_theta(sizes, null_prs)
    return {
        "theta_obs": round(theta_obs, 4),
        "theta_floor": round(theta_floor, 4),
        "delta": round(theta_obs - theta_floor, 4),
        "n_units": n_units,
        "n_trials": n_trials,
        "sizes": sizes,
    }

def main():
    from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

    cache_dir = "/tmp/allen_cache"
    os.makedirs(cache_dir, exist_ok=True)
    cache = EcephysProjectCache.from_warehouse(
        manifest=os.path.join(cache_dir, "manifest.json")
    )

    sessions = cache.get_session_table()
    # Pick Brain Observatory sessions (have drifting gratings + multiple areas)
    good = sessions[sessions['session_type'] == 'brain_observatory_1.1']
    print(f"Brain Observatory sessions: {len(good)}")

    # Run on first 3 sessions
    all_results = []
    rng = np.random.default_rng(42)

    for idx, sid in enumerate(good.index[:3]):
        print(f"\n=== Session {sid} ({idx+1}/3) ===")
        session = cache.get_session_data(sid)
        units = session.units
        areas = units['ecephys_structure_acronym'].unique()
        print(f"  Areas: {list(areas)}")

        # Get drifting grating stimulus table
        try:
            stim = session.get_stimulus_table('drifting_gratings')
        except Exception:
            print(f"  No drifting gratings, skipping")
            continue
        print(f"  Trials: {len(stim)}")

        # Build rate matrix per area
        area_rates = {}
        for area in areas:
            area_units = units[units['ecephys_structure_acronym'] == area]
            if len(area_units) < 20:
                continue
            unit_ids = area_units.index.values
            rates = np.zeros((len(stim), len(unit_ids)))
            for j, uid in enumerate(unit_ids):
                spikes = session.spike_times[uid]
                for i in range(len(stim)):
                    t0 = stim.iloc[i]['start_time']
                    t1 = stim.iloc[i]['stop_time']
                    rates[i, j] = np.sum((spikes >= t0) & (spikes < t1)) / (t1 - t0)
            area_rates[area] = rates
            print(f"  {area}: {len(unit_ids)} units")

        # Within-area delta
        session_result = {"session_id": int(sid), "within_area": {}, "cross_area": None}
        for area, rates in area_rates.items():
            res = neuron_zoom_delta(rates, rng)
            if res:
                session_result["within_area"][area] = res
                print(f"    {area}: delta = {res['delta']:+.4f} ({res['n_units']} units)")

        # Cross-area: pool all neurons
        if len(area_rates) >= 2:
            pooled = np.hstack(list(area_rates.values()))
            res = neuron_zoom_delta(pooled, rng)
            if res:
                session_result["cross_area"] = res
                print(f"    POOLED: delta = {res['delta']:+.4f} ({res['n_units']} units)")

        all_results.append(session_result)

    # Save
    outpath = "data/allen_neuropixels_delta.json"
    with open(outpath, "w") as f:
        json.dump(all_results, f, indent=1)
    print(f"\nSaved to {outpath}")

    # Summary
    print("\n=== SUMMARY ===")
    within_deltas = []
    cross_deltas = []
    for r in all_results:
        for area, d in r["within_area"].items():
            within_deltas.append(d["delta"])
        if r["cross_area"]:
            cross_deltas.append(r["cross_area"]["delta"])
    print(f"Within-area: n={len(within_deltas)}, mean delta = {np.mean(within_deltas):+.4f}")
    print(f"Cross-area:  n={len(cross_deltas)}, mean delta = {np.mean(cross_deltas):+.4f}")
    print(f"Prediction: within < 0, cross > 0")
    print(f"Result:     within {'PASS' if np.mean(within_deltas) < 0 else 'FAIL'}, "
          f"cross {'PASS' if len(cross_deltas) > 0 and np.mean(cross_deltas) > 0 else 'CHECK'}")

if __name__ == "__main__":
    main()
