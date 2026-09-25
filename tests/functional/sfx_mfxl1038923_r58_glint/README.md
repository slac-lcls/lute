# SFX GLINT: mfxl1038923 run 58

LCLS-I (psana1), ePix10k2M at about 50 mm, ~9.8 keV. No unit cell was ever recorded for this sample,
so both indexers run BLIND and the DAG stops at the concatenated stream (no merge).

GLINT's cross-frame consensus on this run finds a cell near 43.6/67.8/89.1 A. The peakfinder8
thresholds are a first guess for this detector and are expected to need tuning.

## Inputs committed with this test

`r0058.geom`: Panel X/Y from mfxx49820 results/btx/geom/r0016.geom (same detector model); psana's
deployed geometry for this experiment has a placeholder distance. The 50 mm distance comes from a
consensus-support scan and agrees with a 3.669 A ice ring at 49.4 mm. The interaction point moved
between runs (r0278 sits about 8 mm further), so this file is specific to r0058.

Paths are resolved through `$LUTE_PATH`, so they point at this directory in the checkout the
tests run from.

## GLINT

GLINTIndexer reads the .cxi files FindPeaksSFX writes and reuses their stored peakfinder8 peaks
(`peakfinder: stored`), so it indexes exactly the peaks the CrystFEL test does. It runs blind
(cross-frame consensus). GLINT is not bundled with LUTE: `executable` defaults to a GLINT checkout's
`lute/glint_launch.sh`, and the indexing step needs a GPU node (ampere). If run_functional.py is
given `--account=...`, that account must also be valid on ampere.
