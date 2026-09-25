# SFX CrystFEL: cxil1015922 run 136

LCLS-I (psana1). The users' protein 'B2', monoclinic C2 (111.94/172.23/41.23 A, beta 106.2),
Jungfrau-4M at 109.4 mm, 8.85 keV.

The users' CrystFEL 0.10.2 run indexed 1,433 crystals in this run's 37,522 events (about 3.8%), so
3,000 events should give roughly 110.

## Inputs committed with this test

`r0136_b2.geom`: The geometry the users indexed with, recovered from the header of their stream
results/b2/indexing/r0136-b2-100-6-350.process/r0136-b2-100-6-350.lst.6.50.stream (the original file
is gone). Photon energy is fixed at 8852 eV there, as in their run. The lysozyme runs of this
experiment (30, 33, 34) have been purged from disk.

`b2.cell`: the reference cell (111.94 172.23 41.23 90.00 106.20 90.00).

Paths are resolved through `$LUTE_PATH`, so they point at this directory in the checkout the
tests run from.
