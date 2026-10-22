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
|                               |                                                                                                            |              |     |                                                                       |
