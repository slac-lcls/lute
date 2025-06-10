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

| Test Name | Workflow              | Experiment | Run | Additional Comments                                                   |
|:---------:|:---------------------:|:----------:|:---:|:---------------------------------------------------------------------:|
| Test1     | Basic LUTE test Tasks | xpptut15   | 670 | Experiment/run are not used by test Tasks. Required for compatibility |
| Test2     | SmallDataProducer     | xpptut15   | 650 | xpplv9818 run 127. This is to test default production only.           |
|           |                       |            |     |                                                                       |
