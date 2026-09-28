# SFX GLINT: cxil1015922 run 136

LCLS-I (psana1). The users' protein 'B2', monoclinic C2 (111.94/172.23/41.23 A, beta 106.2),
Jungfrau-4M at 109.4 mm, 8.85 keV.

The users' CrystFEL 0.10.2 run indexed 1,433 crystals in this run's 37,522 events (about 3.8%), so
3,000 events should give roughly 110.

Peak finding: On 30 calibrated events of this run (every 10th) a 200 threshold leaves a median 0
connected 2-30 px components per frame (max 60); 100 leaves a median 58, mostly the water ring.

## Pass criterion

The workflow completes, the merge has at least ~100 crystals, and CompareHKL's overall CC1/2 is above
0.3 (`fom: "CC"`, the "Overall CC" line in its log). run_functional.py checks only that the
workflow completes; the CC1/2 and the crystal count are read by hand. The beamline's indexing rates
quoted above are floors, not targets: the peak lists here include crystals it did not index.

## Inputs committed with this test

`cxil1015922_r0136_b2.geom`: The geometry the users indexed with, recovered from the header of their
stream results/b2/indexing/r0136-b2-100-6-350.process/r0136-b2-100-6-350.lst.6.50.stream (the
original file is gone). Photon energy is fixed at 8852 eV there, as in their run. The lysozyme runs
of this experiment (30, 33, 34) have been purged from disk.

`cxil1015922_b2.cell`: the reference cell (111.94 172.23 41.23 90.00 106.20 90.00).

The config reads these files from `/sdf/group/lcls/ds/tools/lute/test_utilities/sfx/`; the copies
here are the source to deploy there (`maybe_lyso.cell` is already in `test_utilities/`).

## GLINT

GLINTIndexer reads the .cxi files FindPeaksSFX writes and reuses their stored peakfinder8 peaks
(`peakfinder: stored`), so it indexes exactly the peaks the CrystFEL test does. It is given the same
cell and integrates its own reflections (`integrate: true`), so the stream goes straight to
partialator. GLINT is not bundled with LUTE: `executable` defaults to a GLINT checkout's
`lute/glint_launch.sh`, and the indexing step needs a GPU node (ampere). If run_functional.py is
given `--account=...`, that account must also be valid on ampere.
