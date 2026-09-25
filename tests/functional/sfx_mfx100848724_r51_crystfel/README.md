# SFX CrystFEL: mfx100848724 run 51

Tetragonal lysozyme (P4_32_12, 79.17/79.17/37.96 A), Jungfrau-16M at 356.5 mm, ~11.6 keV. This is
the one run of the experiment in the public data area.

The beamline's cctbx.xfel.process integrated 225 of 17,872 events (1.3%), so 4,000 events should
give roughly 50 indexed frames. Runs 52-55 index at 2.5-4%.

## Inputs committed with this test

`mfx100848724_356mm_refined.geom`: Converted from the refined DIALS geometry
results/common/geom/refined_356mm_17jun.expt with a 180-degree rotation about y (x -> -x, z -> -z).
No CrystFEL geometry for this experiment existed.

`lyso_tetragonal_p43212.cell`: the reference cell (79.17 79.17 37.96 90.00 90.00 90.00).

The config reads these files from `/sdf/group/lcls/ds/tools/lute/test_utilities/sfx/`; the copies
here are the source to deploy there (`maybe_lyso.cell` is already in `test_utilities/`).
