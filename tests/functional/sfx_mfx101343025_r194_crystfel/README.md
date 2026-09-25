# SFX CrystFEL: mfx101343025 run 194

Monoclinic lysozyme (P2_1, 28.0/62.5/60.9 A, beta 90.8) in about 1% of frames, Jungfrau-16M at 124.3
mm, ~15 keV.

The beamline's cctbx.xfel.process indexed 238 of 28,650 events in this run; 22 of those fall in the
first 2,000 events, so 3,000 events should give roughly 30 indexed frames.

Peak finding: On 16 calibrated frames of this run the water ring sits at 400-600 (99.9th pixel
percentile 588); a 1000 threshold leaves a median 49 connected 2-30 px components per frame, 500
leaves ~10,000.

## Inputs committed with this test

`mfx101343025_r0139_refined.geom`: Converted from the BayFAI-refined DIALS geometry
results/common/geom/r0139_psana_imported.expt (distance 124.285 mm) with a 180-degree rotation about
y (x -> -x, z -> -z), which preserves the hand. results/bayfai/lute_output/geom/r0139.geom is the
same geometry mirrored in x and should not be used. The shared test_utilities/jf16mgeom.geom is
nominal (0.2 m) and does not index this run.

`maybe_lyso.cell`: the reference cell (28.00 62.50 60.90 90.00 90.80 90.00).

The config reads these files from `/sdf/group/lcls/ds/tools/lute/test_utilities/sfx/`; the copies
here are the source to deploy there (`maybe_lyso.cell` is already in `test_utilities/`).
