# SFX GLINT: mfx101343025 run 194

Monoclinic lysozyme (P2_1, 28.0/62.5/60.9 A, beta 90.8) in about 1% of frames, Jungfrau-16M at 124.3
mm, ~15 keV.

The beamline's cctbx.xfel.process indexed 238 of 28,650 events in this run; 22 of those fall in the
first 2,000 events, so 3,000 events should give roughly 30 indexed frames.

Peak finding: On 16 calibrated frames of this run the water ring sits at 400-600 (99.9th pixel
percentile 588); a 1000 threshold leaves a median 49 connected 2-30 px components per frame, 500
leaves ~10,000.

The threshold is 5000, not 1000. At 1000, 64% of the stored peaks fall in a 4-6 A band (median 195
peaks per frame), and both indexers index that band as if it were a lattice. Matched to cctbx by
timestamp, the 31 cctbx-indexed frames among the first 3,000 events have a median brightest peak
of 134,937 ADU against 1,750 for the rest. On the 1000-threshold peak lists, "at least 10 peaks
above 5000 ADU" keeps 28 of the 31 and 310 other frames. Those 310 are not ice (their peaks sit no
closer to the ice rings than chance), so they are probably crystals cctbx did not index, and the
cctbx rate above is a floor.

## Pass criterion

The workflow completes, the merge has at least ~100 crystals, and CompareHKL's overall CC1/2 is above
0.3 (`fom: "CC"`, the "Overall CC" line in its log). run_functional.py checks only that the
workflow completes; the CC1/2 and the crystal count are read by hand. The beamline's indexing rates
quoted above are floors, not targets: the peak lists here include crystals it did not index.

## Inputs committed with this test

`mfx101343025_r0139_refined.geom`: Converted from the BayFAI-refined DIALS geometry
results/common/geom/r0139_psana_imported.expt (distance 124.285 mm) with a 180-degree rotation about
y (x -> -x, z -> -z), which preserves the hand. results/bayfai/lute_output/geom/r0139.geom is the
same geometry mirrored in x and should not be used. The shared test_utilities/jf16mgeom.geom is
nominal (0.2 m) and does not index this run.

`maybe_lyso.cell`: the reference cell (28.00 62.50 60.90 90.00 90.80 90.00).

The config reads these files from `/sdf/group/lcls/ds/tools/lute/test_utilities/sfx/`; the copies
here are the source to deploy there (`maybe_lyso.cell` is already in `test_utilities/`).

## GLINT

GLINTIndexer reads the .cxi files FindPeaksSFX writes and reuses their stored peakfinder8 peaks
(`peakfinder: stored`), so it indexes exactly the peaks the CrystFEL test does. It is given the same
cell and writes only orientations (`tofile`, lattice `mPb`). CrystFELIndexer then reads them with
`--indexing=file`, refines them, checks them against the cell and integrates, so both tests share
CrystFEL's integration and differ only in who found the orientation.

Why not GLINT's own integration (`integrate: true`): on the round-4 validation of mfx100848724 r51,
on the 171 frames both indexers indexed (same orientation on 166), CrystFEL's integration merged to
CC1/2 0.27 and GLINT's to 0.08. Handing GLINT's orientations to CrystFEL gave 0.27. CrystFEL's
refinement also rejected all 887 frames that only GLINT had indexed, which the strict gate had passed.

GLINT is not bundled with LUTE: `executable` defaults to a GLINT checkout's `lute/glint_launch.sh`,
and the indexing step needs a GPU node (ampere). If run_functional.py is given `--account=...`, that
account must also be valid on ampere.
