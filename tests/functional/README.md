# Functional Tests

## Usage
- All tests should be placed here in a sub-directory.
- The `run_functional.py` script (one-level up) will run the tests for every sub-directory.
- Each sub-directory **must contain:**
  - `config.yaml`: The LUTE configuration YAML for this test. **It must have a specific** `experiment` **and** `run` defined in the LUTE header.
  - `dag.yaml`: The workflow definition YAML for this test.
- Each sub-directory may also contain:
  - `SHOULD_FAIL`: An empty file indicating the workflow is intended to return a failure status.
  - `README.md`: An explanatory README for the test.

## List of tests

| Test Name                     | Workflow                                                                                                   | Experiment   | Run | Additional Comments                                                   |
|:-----------------------------:|:----------------------------------------------------------------------------------------------------------:|:------------:|:---:|:---------------------------------------------------------------------:|
| basic_tests                   | Basic LUTE test Tasks                                                                                      | xpptut15     | 670 | Experiment/run are not used by test Tasks. Required for compatibility |
| smd_xpp_default               | SmallDataProducer                                                                                          | xpptut15     | 650 | xpplv9818 run 127. This is to test default production only.           |
| smd_mfx_default               | SmallDataProducer                                                                                          | mfxx49820    | 16  | Default small data production                                         |
| smd2_mfx_prod                 | SmallDataProducer2                                                                                         | mfx101344525 | 70  | LCLS2 non-default SMD (MFX).                                          |
| smd2_multi_node               | SmallDataProducer2                                                                                         | mfx101262725 | 96  | LCLS2 non-default SMD (MFX) submitted across multiple nodes.          |
| param_generation              | Basic LUTE tests + param generation                                                                        | xpptut15     | 670 | Experiment/run not used by the test but required for compatibility.   |
| peakfinder8_lcls2             | Run FindPeaksSFX with the peakfinder8 v1 algorithm on LCLS2 data.                                          | mfx101343025 | 170 | Some Jungfrau16M data.                                                |
| peakfinder8_lcls2_compression | Run FindPeaksSFX with the peakfinder8 v1 algorithm on LCLS2 data but do compress/decompress with RoiBinSZ. | mfx101343025 | 170 | Some Jungfrau16M data.                                                |
| basic_rest                    | Basic tests of `maestro` REST APIs                                                                         | xpptut15     | 670 | Experiment/run are not used by test Tasks. Required for compatibility |
| smd_bayfai                    | SmallDataProducer → BayFAIOptimizer                                                                        | mfx100824024 | 5   | psana1, epix10k2M, LaB6 calibrant. Powder auto-resolved from SMD.    |
| smd2_bayfai                   | SmallDataProducer2 → BayFAIOptimizer2                                                                      | mfx100852324 | 298 | psana2, jungfrau, AgBh calibrant. Powder auto-resolved from SMD2.    |
| smd2_xss                      | SmallDataProducer2 → SmallDataXSSAnalyzer                                                                  | mfx101344525 | 82  | psana2, jungfrau, PyFAI azint (r0082.poni), lxt/lens scan.           |
| sfx_mfx101343025_r194_crystfel | FindPeaksSFX → CrystFEL (xgandalf) → merge | mfx101343025 | 194 | Jungfrau16M, monoclinic lysozyme, BayFAI-refined r0139 geometry. |
| sfx_mfx101343025_r194_glint | FindPeaksSFX → GLINT → merge | mfx101343025 | 194 | Same peaks as the CrystFEL test. GLINT needs an A100 node. |
| sfx_mfx100848724_r51_crystfel | FindPeaksSFX → CrystFEL (xgandalf) → merge | mfx100848724 | 51 | Jungfrau16M, tetragonal lysozyme, refined 356 mm geometry. |
| sfx_mfx100848724_r51_glint | FindPeaksSFX → GLINT → merge | mfx100848724 | 51 | Same peaks as the CrystFEL test. GLINT needs an A100 node. |
| sfx_cxil1015922_r136_crystfel | FindPeaksSFX (psana1) → CrystFEL → merge | cxil1015922 | 136 | LCLS-I, Jungfrau4M, monoclinic C2 'B2' protein. |
| sfx_cxil1015922_r136_glint | FindPeaksSFX (psana1) → GLINT → merge | cxil1015922 | 136 | LCLS-I. Same peaks as the CrystFEL test. |
| sfx_mfxl1038923_r58_crystfel | FindPeaksSFX (psana1) → CrystFEL, blind | mfxl1038923 | 58 | LCLS-I, epix10k2M. No known cell: indexing only, no merge. |
| sfx_mfxl1038923_r58_glint | FindPeaksSFX (psana1) → GLINT, blind | mfxl1038923 | 58 | LCLS-I, epix10k2M. No known cell: indexing only, no merge. |
|                               |                                                                                                            |              |     |                                                                       |
