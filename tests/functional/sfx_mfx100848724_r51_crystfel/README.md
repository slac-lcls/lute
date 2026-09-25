# SFX CrystFEL: mfx100848724 run 51

Tetragonal lysozyme (P4_32_12, 79.17/79.17/37.96 A), Jungfrau-16M at 356.5 mm, ~11.6 keV. This is
the one run of the experiment in the public data area.

The beamline's cctbx.xfel.process integrated 225 of 17,872 events (1.3%), so 4,000 events should
give roughly 50 indexed frames. Runs 52-55 index at 2.5-4%.

## Inputs committed with this test

`refined_356mm.geom`: Converted from the refined DIALS geometry
results/common/geom/refined_356mm_17jun.expt with a 180-degree rotation about y (x -> -x, z -> -z).
No CrystFEL geometry for this experiment existed.

`lyso_tetragonal.cell`: the reference cell (79.17 79.17 37.96 90.00 90.00 90.00).

Paths are resolved through `$LUTE_PATH`, so they point at this directory in the checkout the
tests run from.
