"""CPU time of the worked example's two commands (Appendix N of the extended version), each with --workers 1 so the time is one
core's, run one after the other with the rung installed in this Python (the extended version used rung 1.13.1:
pip install "rung @ git+https://github.com/adilamin89/structure-in-the-zoom@v1.13.1"):
  gt3  rung data gt3_X.npy gt3_labels.npy --sectors --decompose --n-perm 200 --workers 1
  lm   rung llm --model EleutherAI/pythia-160m --axis axes/world_knowledge32.json --paper-seeds --match-last-token --workers 1
Records each process's CPU time (getrusage user + system, which a throttled wall clock does not move) and the rung version. The
GT3 inputs are built from raw/gratings_drifting_GT3_2019_04_05_1.npy as REFERENCE.md builds them; the lm command needs
rung[models]. Time from a terminal on an otherwise idle machine. Out: v3/appm_rerun_1131.json, per-command outputs and logs in
v3/appm_work/. Run: python v3/appm_rerun_1131.py"""
import json
import os
import subprocess
import sys
import time

import numpy as np

from _common import AXIS, HERE, RAW

WORK = HERE / "appm_work"


def run(args, out):
    env = dict(os.environ, VECLIB_MAXIMUM_THREADS="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    env.pop("PYTHONPATH", None)                                     # the installed rung, not the clone's file
    code = ("import resource, runpy, sys; sys.argv = ['rung'] + sys.argv[1:]\n"
            "try:\n    runpy.run_module('rung', run_name='__main__')\nfinally:\n"
            "    u = resource.getrusage(resource.RUSAGE_SELF); print('CPU_SECONDS', u.ru_utime + u.ru_stime, file=sys.stderr)")
    t0 = time.time()
    p = subprocess.run([sys.executable, "-c", code, *args, "--out", out], cwd=WORK, env=env, capture_output=True, text=True)
    (WORK / (out + ".log")).write_text(p.stdout + "\n--- stderr ---\n" + p.stderr)
    if p.returncode != 0:
        raise RuntimeError(p.stderr[-3000:])
    cpu = float(next(l.split()[1] for l in p.stderr.splitlines() if l.startswith("CPU_SECONDS")))
    return cpu, time.time() - t0


def main():
    WORK.mkdir(exist_ok=True)
    if not (WORK / "gt3_X.npy").exists():
        d = np.load(RAW / "gratings_drifting_GT3_2019_04_05_1.npy", allow_pickle=True).item()
        np.save(WORK / "gt3_X.npy", np.asarray(d["sresp"], np.float32).T)
        phi = np.asarray(d["istim"], float) % (2 * np.pi)
        np.save(WORK / "gt3_labels.npy", np.clip(np.digitize(phi, np.linspace(0, 2 * np.pi, 9)) - 1, 0, 7))
    ver = subprocess.run([sys.executable, "-m", "rung", "--version"], cwd=WORK, capture_output=True, text=True,
                         env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"}).stdout.strip()
    res = {"rung": ver}
    cpu, wall = run(["data", "gt3_X.npy", "gt3_labels.npy", "--sectors", "--decompose", "--n-perm", "200", "--workers", "1"], "gt3.json")
    res["gt3"] = {"cpu_s": cpu, "wall_s": wall}
    print(f"gt3: {cpu:.1f} CPU-s ({wall:.0f} s wall)", flush=True)
    cpu, wall = run(["llm", "--model", "EleutherAI/pythia-160m", "--axis", str(AXIS), "--paper-seeds", "--match-last-token", "--workers", "1"], "lm.json")
    res["lm"] = {"cpu_s": cpu, "wall_s": wall}
    print(f"lm: {cpu:.1f} CPU-s ({wall:.0f} s wall)", flush=True)
    json.dump(res, open(HERE / "appm_rerun_1131.json", "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
