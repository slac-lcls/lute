"""
make_sim_npz.py — Compute I(Q,T) from MD trajectories and save as .npz.

This script uses the build_sim_curves() function from your temperature_project
preprocessing module to compute X-ray scattering curves from the .dcd/.pdb
trajectory files, then saves them in the format expected by infer_temperature().

Usage:
    python make_sim_npz.py \
        --traj_dir /path/to/sim_runs/md_plain \
        --tp_dir   /path/to/temperature_project \
        --out      /path/to/sim_curves.npz

Arguments:
    --traj_dir   Directory containing traj_tip4pew_<T>C_plain.dcd/.pdb files.
    --tp_dir     Path to the temperature_project root (where preprocessing.py lives).
    --out        Output .npz file path (default: ./sim_curves.npz).
    --temps      Space-separated list of temperatures in °C to compute
                 (default: all found in traj_dir).
    --stride     Frame stride for trajectory loading (default: 1 = all frames).
"""

import argparse
import os
import sys
import re
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--traj_dir", required=True,
                    help="Directory with .dcd/.pdb trajectory files.")
parser.add_argument("--tp_dir", required=True,
                    help="Path to temperature_project root (contains preprocessing.py).")
parser.add_argument("--out", default="./sim_curves.npz",
                    help="Output .npz file (default: ./sim_curves.npz).")
parser.add_argument("--temps", nargs="+", type=int, default=None,
                    help="Temperatures in °C to compute. Default: auto-detect from files.")
parser.add_argument("--stride", type=int, default=1,
                    help="Frame stride for trajectory (default: 1).")
args = parser.parse_args()

# ── Import preprocessing from temperature_project ────────────────────────────
sys.path.insert(0, os.path.abspath(args.tp_dir))
import preprocessing as pp

# ── Auto-detect temperatures if not given ────────────────────────────────────
if args.temps is None:
    pattern = re.compile(r"traj_tip4pew_(\d+)C_plain\.dcd")
    found = sorted({
        int(m.group(1))
        for f in os.listdir(args.traj_dir)
        if (m := pattern.match(f))
    })
    if not found:
        print(f"ERROR: No traj_tip4pew_*C_plain.dcd files found in {args.traj_dir}")
        sys.exit(1)
    print(f"Auto-detected temperatures: {found}")
    temps_c = found
else:
    temps_c = sorted(args.temps)
    print(f"Using temperatures: {temps_c}")

# ── Compute scattering curves ─────────────────────────────────────────────────
print(f"\nLoading trajectories from: {args.traj_dir}")
print(f"Stride: {args.stride}  (using every {args.stride}th frame)")
print("Computing I(Q,T) via RDF method — this may take several minutes...\n")

Q_common, curves_dict = pp.build_sim_curves(
    temps_c=temps_c,
    outdir=args.traj_dir,
    stride=args.stride,
)

# ── Pack into arrays ─────────────────────────────────────────────────────────
# Sort by temperature
temps_sorted = np.array(sorted(curves_dict.keys()), dtype=float)
curves_arr = np.stack([curves_dict[int(T)] for T in temps_sorted], axis=0)

print(f"\nQ range : {Q_common.min():.3f} – {Q_common.max():.3f} Å⁻¹  ({len(Q_common)} points)")
print(f"Temps   : {temps_sorted.tolist()} °C")
print(f"Curves  : {curves_arr.shape}  (N_temps × N_q)")

# ── Save ──────────────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
np.savez(args.out, q=Q_common, curves=curves_arr, temps=temps_sorted)
print(f"\nSaved: {args.out}")
print("Keys: q, curves, temps")
print("Ready to use with:  python test_thermometry.py <smd.h5> " + args.out)
