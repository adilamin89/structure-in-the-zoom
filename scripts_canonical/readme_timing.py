"""CPU time of the two commands the README times, under the rung installed in this Python, each with --workers 1 (one core), run one
after the other:
  llm  rung llm --model EleutherAI/pythia-160m --axis axes/world_knowledge32.json --match-last-token   (README, 'Your data')
  gt3  rung data gt3_X.npy gt3_labels.npy --sectors --decompose --n-perm 200                          (the paper's worked example)
Each process's CPU time is getrusage user + system, which a throttled wall clock does not move. The GT3 inputs are built from
raw/gratings_drifting_GT3_2019_04_05_1.npy as REFERENCE.md builds them; the llm command needs rung[models] and downloads the model on
first use. Run from a terminal on an otherwise idle machine. Out: data_canonical/readme_timing.json (work files in a temporary folder).
Run: python scripts_canonical/readme_timing.py"""
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data_canonical" / "readme_timing.json"


def run(args, cwd):
    env = dict(os.environ, VECLIB_MAXIMUM_THREADS="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    env.pop("PYTHONPATH", None)
    code = ("import resource, runpy, sys; sys.argv = ['rung'] + sys.argv[1:]\n"
            "try:\n    runpy.run_module('rung', run_name='__main__')\nfinally:\n"
            "    u = resource.getrusage(resource.RUSAGE_SELF); print('CPU_SECONDS', u.ru_utime + u.ru_stime, file=sys.stderr)")
    t0 = time.time()
    p = subprocess.run([sys.executable, "-c", code, *args, "--workers", "1", "--out", "out.json"], cwd=cwd, env=env,
                       capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr[-3000:])
    return float(next(l.split()[1] for l in p.stderr.splitlines() if l.startswith("CPU_SECONDS"))), time.time() - t0


def main():
    ver = subprocess.run([sys.executable, "-m", "rung", "--version"], capture_output=True, text=True,
                         env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"}).stdout.strip()
    cpu = (subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
           if sys.platform == "darwin" else platform.processor())
    res = {"rung": ver, "cpu": cpu, "python": platform.python_version(), "workers": 1}
    with tempfile.TemporaryDirectory() as tmp:
        c, w = run(["llm", "--model", "EleutherAI/pythia-160m", "--axis", str(ROOT / "axes" / "world_knowledge32.json"),
                    "--match-last-token"], tmp)
        res["llm"] = {"cpu_s": c, "wall_s": w}; print(f"llm {c:.1f} CPU-s ({w:.0f} s wall)", flush=True)
        d = np.load(ROOT / "raw" / "gratings_drifting_GT3_2019_04_05_1.npy", allow_pickle=True).item()
        np.save(Path(tmp) / "gt3_X.npy", np.asarray(d["sresp"], np.float32).T)
        phi = np.asarray(d["istim"], float) % (2 * np.pi)
        np.save(Path(tmp) / "gt3_labels.npy", np.clip(np.digitize(phi, np.linspace(0, 2 * np.pi, 9)) - 1, 0, 7))
        c, w = run(["data", "gt3_X.npy", "gt3_labels.npy", "--sectors", "--decompose", "--n-perm", "200"], tmp)
        res["gt3"] = {"cpu_s": c, "wall_s": w}; print(f"gt3 {c:.1f} CPU-s ({w:.0f} s wall)", flush=True)
    OUT.write_text(json.dumps(res, indent=1) + "\n")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
