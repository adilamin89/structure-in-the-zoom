"""rung 1.11's speed on Appendix M's two commands, and that it returns the same numbers. Each command runs as its own process
under an older rung (OLD, e.g. a clone at tag v1.10.0) and under this one, with BLAS and torch held to one thread, and the CPU
time of each process (user + system, getrusage) is recorded: one core's time. The jobs run one after another on an otherwise idle
machine: concurrent processes contend for memory bandwidth and the matrix units, which inflates the CPU time of the fast arithmetic
(two-fold when this script ran all five jobs at once) more than the row-by-row one's. Every numeric field of the two results is compared.
  gt3      rung data gt3_X.npy gt3_labels.npy --sectors --decompose --n-perm 200        (Appendix M; README GT3 recipe makes the inputs)
  lm       rung llm --model EleutherAI/pythia-160m --axis axes/world_knowledge32.json --match-last-token
  lm_paper the same with --paper-seeds (row-by-row arithmetic: the paper's cells, bit for bit), this rung only
Run: python rung_speed.py OLD_DIR GT3_DIR      Out: data_canonical/rung_speed.json"""
import json
import math
import os
import platform
import resource
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
NEW = HERE.parent
OUT = NEW / "data_canonical" / "rung_speed.json"


def run(code_dir, args, cwd, out):
    """One rung command in its own process; returns (CPU seconds, the result JSON)."""
    env = dict(os.environ, PYTHONPATH=str(code_dir), VECLIB_MAXIMUM_THREADS="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", HF_HUB_OFFLINE=os.environ.get("HF_HUB_OFFLINE", "0"))
    code = ("import resource, runpy, sys; sys.argv = ['rung'] + sys.argv[1:]\n"
            "try:\n    runpy.run_module('rung', run_name='__main__')\nfinally:\n"
            "    u = resource.getrusage(resource.RUSAGE_SELF); print('CPU_SECONDS', u.ru_utime + u.ru_stime, file=sys.stderr)")
    p = subprocess.run([sys.executable, "-c", code, *args, "--out", out], cwd=cwd, env=env, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"{args[:2]} failed in {code_dir}: {p.stderr[-2000:]}")
    cpu = float(next(line.split()[1] for line in p.stderr.splitlines() if line.startswith("CPU_SECONDS")))
    return cpu, json.load(open(Path(cwd) / out))


def compare(a, b):
    """Largest relative difference over the numeric fields both results hold, and whether every p-value is identical."""
    worst, p_same = 0.0, True

    def walk(x, y, key=""):
        nonlocal worst, p_same
        if isinstance(x, dict):
            for k in x:
                if k in y:
                    walk(x[k], y[k], k)
        elif isinstance(x, list) and isinstance(y, list):
            for u, v in zip(x, y):
                walk(u, v, key)
        elif isinstance(x, (int, float)) and not isinstance(x, bool) and isinstance(y, (int, float)):
            if math.isnan(x) and math.isnan(y):
                return
            if "p_two" in key or key.startswith("p_"):
                p_same &= x == y
            else:
                worst = max(worst, abs(x - y) / max(1.0, abs(x)))
    walk(a, b)
    return worst, p_same


def main():
    old, gt3 = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    work = Path(tempfile.mkdtemp())
    for f in ("gt3_X.npy", "gt3_labels.npy"):
        os.symlink(gt3 / f, work / f)
    axis = str(NEW / "axes" / "world_knowledge32.json")
    cmds = {"gt3": ["data", "gt3_X.npy", "gt3_labels.npy", "--sectors", "--decompose", "--n-perm", "200"],
            "lm": ["llm", "--model", "EleutherAI/pythia-160m", "--axis", axis, "--match-last-token"]}
    jobs = [(name, tag, d, cmds[name] + (["--workers", "1"] if tag == "new" else []))
            for name in cmds for tag, d in (("old", old), ("new", NEW))]
    jobs.append(("lm_paper", "new", NEW, cmds["lm"] + ["--paper-seeds", "--workers", "1"]))
    res = {}
    for name, tag, d, args in jobs:                # one at a time, each process held to one thread
        res[(name, tag)] = run(d, args, work, f"{name}_{tag}.json")
        print(f"  {name} ({tag}): {res[(name, tag)][0]:.1f} s of CPU", flush=True)
    out = {"machine": f"{platform.machine()} {platform.platform()}", "cpu_count": os.cpu_count(), "python": platform.python_version()}
    for name in cmds:
        (c_old, r_old), (c_new, r_new) = res[(name, "old")], res[(name, "new")]
        worst, p_same = compare(r_old, r_new)
        out[name] = {"rung_old": r_old.get("rung_version"), "rung_new": r_new.get("rung_version"), "cpu_s_old": c_old, "cpu_s_new": c_new,
                     "cpu_ratio": c_old / c_new, "max_rel_diff": worst, "p_values_identical": p_same,
                     "arithmetic_new": r_new.get("arithmetic")}
        print(f"{name}: {c_old:.0f} s -> {c_new:.1f} s of CPU ({c_old / c_new:.0f}x); max relative difference {worst:.1e}; p identical {p_same}",
              flush=True)
    c_p, r_p = res[("lm_paper", "new")]
    out["lm_paper"] = {"cpu_s_new": c_p, "arithmetic": r_p.get("arithmetic")}
    print(f"lm --paper-seeds: {c_p:.0f} s of CPU (rows)", flush=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
