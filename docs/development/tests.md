# Running LUTE Tests
## Functional Tests

LUTE provides a framework for running a series of workflows as a functional test suite. This provides a mechanism to both test changes in LUTE code, as well as verify that underlying code run by various `Task`s still behaves as expected. This latter use-case allows testing against new `psana` release, for example.

### Test Organization

Functional tests are organized as sub-directories under the `tests/functional` directory of the main repository. An alternative directory can be provided to run against, however, it must maintain the same structure described here.

Under `tests/functional` each test workflow to run is given a sub-directory which **must** contain the following:

- `config.yaml`: to specify the LUTE configuration YAML for running the test workflow. This YAML **MUST specify** an experiment and run. It **CANNOT** rely on retrieving this information from the eLog or the command-line.
- `dag.yaml`: A workflow specification in line with the dyanmic DAG YAML syntax for LUTE.

Each test sub-directory may also contain:

- A `README.md` file to explain the test.
- A `SHOULD_FAIL` file which is empty. This should be provided if the workflow is expected to fail and the test should be marked succesful if it does.

### Running Tests

**NOTE:** Currently, the full test suite requires making a dual-Python version installation as one of the tests requires Python 3.11. Refer to the installation instructions regarding how this is done. The rest of the tests can be run with a normal single-Python (3.9) installation. This will be needed until the standard psana environment is advanced to Python 3.11.

There are two main scripts for test submission:

- `run_functional.py`: This does the bulk of the work.
- `submit_run_functional.sh`: a thin wrapper to submit a SLURM job.

As an alternative to a SLURM job, `run_functional.py` can be run in the background with output redirected to a file, or perhaps more securely using, e.g. screen. `run_functional.py` can be used like:

```bash
usage: run_functional.py [-h] [-a] [--test_airflow] [--use_prefect] [--git_pr_id GIT_PR_ID] [--git_tag GIT_TAG] -r RUN_DIR [--no_delete] [--run_tests RUN_TESTS] [--tests_dir TESTS_DIR] [--use_local_tests]

Run the LUTE functional test suite.

optional arguments:
  -h, --help            show this help message and exit
  -a, --admin           Run as Airflow admin. Requires permissions. Ignored if using prefect.
  --test_airflow        Use test Airflow instance. Ignored if using prefect.
  --use_prefect         Use prefect (experimental) instead of Airflow.
  --git_pr_id GIT_PR_ID
                        Check out a specific GitHub PR ID of LUTE to run (a PR branch).
  --git_tag GIT_TAG     Check out a specific git tag of LUTE to run (e.g. a release).
  -r RUN_DIR, --run_dir RUN_DIR
                        Directory to install LUTE to.
  --no_delete           If passed, do not delete output files when tests are finished.
  --run_tests RUN_TESTS
                        Provide a comma-separated string of tests to run. If provided, this script will only run those, rather than the default behaviour of running all tests. E.g: --run_these_tests
                        test2,test5. Tests that do not exist are silently ignored.
  --tests_dir TESTS_DIR
                        Specify an alternative path to tests than those from the LUTE clone. Must have the same directory structure: $DIR/test1/... $DIR/test2/... If this flag and --use_local_tests are both
                        passed, this one is used.
  --use_local_tests     Use the tests from the installation of LUTE where this script is called, rather than those from the clone of LUTE which is run against, or another directory if passed. If this flag
                        and --tests_dir are both passed, --tests_dir is used.

Refer to https://github.com/slac-lcls/lute for more information.
```

`submit_run_functional.sh` transparently passes parameters so has the same interface, SLURM arguments will be required however. These arguments are just for the job running the `run_functional.py` script, as the SLURM arguments for each step of the underlying workflow are specified in the `dag.yaml` file.

#### Example

Checkout a copy of `LUTE` and run the following command:

```bash
./lute/tests/submit_run_functional.sh --run_dir <$OUTPUT_DIR> [--use_prefect]
```

Substituting `<$OUTPUT_DIR>` for any directory you would like to work in. Within this directory, the test script will create the following directories:

```
OUTPUT_DIR
|
|------------ lute # (The clone of the repository)
|
|------------ lute_output # (All output files will be placed here)
```

**Note:** A new copy of LUTE will be cloned unless the `--use_local_tests` argument is passed. Using the appropriate arguments listed above, the new clone of LUTE can be checked out to use a specific tag or PR branch.

`OUTPUT_DIR` must exist before the script is called. The `lute` and `lute_output` directories will be deleted after the tests complete, unless the `--no_delete` option is passed.

The results of the tests can be examined from the SLURM log file:

```bash
> tail -n 4 slurm-59804105.out
INFO:Launch_Func_Tests:Ran 2 tests. 2 were successful.
DEBUG:Launch_Func_Tests:Removing duplicate Kerberos credentials.
INFO:Launch_Func_Tests:Cleaning up /sdf/scratch/users/d/dorlhiac/func_tests/lute
INFO:Launch_Func_Tests:Cleaning up /sdf/scratch/users/d/dorlhiac/func_tests/lute_output
```
