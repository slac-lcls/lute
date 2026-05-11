"""
test_thermometry.py — 5-line thermometry test using LUTE's infer_temperature.

Usage:
    python test_thermometry.py /path/to/smalldata.h5 /path/to/sim_curves.npz

The sim .npz file must contain:
    q      (N_q,)        Q axis in Å⁻¹
    curves (N_T, N_q)    simulated I(Q,T) profiles
    temps  (N_T,)        temperatures in °C

Optional args:
    --ipm   IPM variable name (auto-detected if not given)
    --det   Detector name     (auto-detected if not given)
    --out   Output directory  (default: ./thermo_output)
"""

import argparse, os, warnings
import numpy as np
import panel as pn

parser = argparse.ArgumentParser()
parser.add_argument("smd_path")
parser.add_argument("sim_path")
parser.add_argument("--ipm", default=None)
parser.add_argument("--det", default=None)
parser.add_argument("--out", default="./thermo_output")
args = parser.parse_args()

os.makedirs(args.out, exist_ok=True)

# ── 1. Load data ─────────────────────────────────────────────────────────────
from lute.io.models.base import AnalysisHeader
from lute.io.models.smd import AnalyzeSmallDataXSSParameters
from lute.tasks.smalldata import AnalyzeSmallDataXSS

import h5py
ipm_var = args.ipm
if ipm_var is None:
    _CANDS = ["ipm2/sum", "ipm4/sum", "ipm3/sum", "ipm5/sum", "ipm2", "ipm3"]
    with h5py.File(args.smd_path, "r") as _f:
        for _c in _CANDS:
            if _c in _f:
                ipm_var = _c; break
    if ipm_var is None:
        raise RuntimeError("Cannot auto-detect ipm_var. Pass --ipm explicitly.")

header = AnalysisHeader(experiment="local_test", run=0, work_dir=args.out)
params = AnalyzeSmallDataXSSParameters(
    lute_config=header, smd_path=args.smd_path,
    ipm_var=ipm_var, xss_detname=args.det,
)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    task = AnalyzeSmallDataXSS(params=params, use_mpi=False)

# ── 2. Extract standard XSS data ─────────────────────────────────────────────
task._extract_standard_data()
task._extract_xss(params.xss_event_codes)

# ── 3. Infer temperature ──────────────────────────────────────────────────────
results = task.infer_temperature(args.sim_path)

# ── 4. Report summary ────────────────────────────────────────────────────────
T = results["T_mean"]
on, off = results["on_mask"], results["off_mask"]
print(f"Laser-on  shots: {on.sum()},  mean T = {T[on].mean():.1f} ± {T[on].std():.1f} °C")
print(f"Laser-off shots: {off.sum()}, mean T = {T[off].mean():.1f} ± {T[off].std():.1f} °C")

# ── 5. Save interactive plots ─────────────────────────────────────────────────
out_html = os.path.join(args.out, "thermometry.html")
results["plot"].save(out_html, embed=True)
print(f"Saved: {out_html}")
