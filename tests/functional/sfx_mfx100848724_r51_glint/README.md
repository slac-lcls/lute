# SFX GLINT: mfx100848724 run 51

Tetragonal lysozyme (P4_32_12, 79.17/79.17/37.96 A), Jungfrau-16M at 356.5 mm, ~11.6 keV. This is
the one run of the experiment in the public data area.

The beamline's cctbx.xfel.process integrated 225 of 17,872 events (1.3%), so 12,000 events should
give roughly 150 indexed frames, enough for the pass criterion below (4,000 gave about 50). Runs
52-55 index at 2.5-4%.

Peak finding: On 16 calibrated frames of this run the 99.9th pixel percentile is 615; a 1000
threshold leaves a median 28 connected 2-30 px components per frame, 500 leaves ~3,500.

## Pass criterion

The workflow completes, the merge has at least ~100 crystals, and CompareHKL's overall CC1/2 is above
0.3 (`fom: "CC"`, the "Overall CC" line in its log). run_functional.py checks only that the
workflow completes; the CC1/2 and the crystal count are read by hand. The beamline's indexing rates
quoted above are floors, not targets: the peak lists here include crystals it did not index.

## Inputs committed with this test

`mfx100848724_356mm_refined.geom`: Converted from the refined DIALS geometry
results/common/geom/refined_356mm_17jun.expt with a 180-degree rotation about y (x -> -x, z -> -z).
No CrystFEL geometry for this experiment existed.

`lyso_tetragonal_p43212.cell`: the reference cell (79.17 79.17 37.96 90.00 90.00 90.00).

The config reads these files from `/sdf/group/lcls/ds/tools/lute/test_utilities/sfx/`; the copies
here are the source to deploy there (`maybe_lyso.cell` is already in `test_utilities/`).

## GLINT

GLINTIndexer reads the .cxi files FindPeaksSFX writes and reuses their stored peakfinder8 peaks
(`peakfinder: stored`), so it indexes exactly the peaks the CrystFEL test does. It is given the same
cell and integrates its own reflections (`integrate: true`), so the stream goes straight to
partialator. GLINT is not bundled with LUTE: `executable` defaults to a GLINT checkout's
`lute/glint_launch.sh`, and the indexing step needs a GPU node (ampere). If run_functional.py is
given `--account=...`, that account must also be valid on ampere.
