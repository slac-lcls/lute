# Setup
LUTE is publically available on [GitHub](https://github.com/slac-lcls/lute). In order to run it, the first step is to clone the repository:

```bash
# Navigate to the directory of your choice.
git clone@github.com:slac-lcls/lute
```
The repostiory directory structure is as follows:

```
lute
  |--- config             # Configuration YAML files (see below) and templates for third party config
  |--- docs               # Documentation (including this page)
  |--- launch_scripts     # Entry points for using SLURM and communicating with Airflow
  |--- lute               # Code
        |--- run_task.py  # Script to run an individual managed Task
        |--- ...
  |--- utilities          # Help utility programs

```

In general, most interactions with the software will be through scripts located in the `launch_scripts` directory. Some users (for certain use-cases) may also choose to run the `run_task.py` script directly - it's location has been highlighted within hierarchy. To begin with you will need a YAML file, templates for which are available in the `config` directory. The structure of the YAML file and how to use the various launch scripts are described in more detail below.

### A note on utilties
In the `utilities` directory there are two useful programs to provide assistance with using the software:
- `utilities/dbview`: LUTE stores all parameters for every analysis routine it runs (as well as results) in a database. This database is stored in the `work_dir` defined in the YAML file (see below). The `dbview` utility is a TUI application (Text-based user interface) which runs in the terminal. It allows you to navigate a LUTE database using the arrow keys, etc. Usage is: `utilities/dbview -p <path/to/lute.db>`.
- `utilities/lute_help`: This utility provides help and usage information for running LUTE software. E.g., it provides access to parameter descriptions to assist in properly filling out a configuration YAML. It's usage is described in slightly more detail below.

# Basic Usage
## Overview
LUTE runs code as `Task`s that are managed by an `Executor`. The `Executor` provides modifications to the environment the `Task` runs in, as well as controls details of inter-process communication, reporting results to the eLog, etc. Combinations of specific `Executor`s and `Task`s are already provided, and are referred to as **managed** `Task`s. **Managed** `Task`s are submitted as a single unit. They can be run individually, or a series of independent steps can be submitted all at once in the form of a workflow, or **directed acyclic graph** (**DAG**). This latter option makes use of Airflow to manage the individual execution steps.

Running analysis with LUTE is the process of submitting one or more **managed** `Task`s. This is generally a two step process.
1. First, a configuration YAML file is prepared. This contains the parameterizations of all the `Task`s which you may run.
2. Individual **managed** `Task` submission, or workflow (**DAG**) submission.

These two steps are described below.

## Preparing a Configuration YAML
All `Task`s are parameterized through a single configuration YAML file - even third party code which requires its own configuration files is managed through this YAML file. The basic structure is split into two documents, a brief header section which contains information that is applicable across all `Task`s, such as the experiment name, run numbers and the working directory, followed by per `Task` parameters:
```yaml
%YAML 1.3
---
title: "Some title."
experiment: "MYEXP123"
# run: 12 # Does not need to be provided
date: "2024/05/01"
lute_version: 0.1
task_timeout: 600
work_dir: "/sdf/scratch/users/d/dorlhiac"
...
---
TaskOne:
  param_a: 123
  param_b: 456
  param_c:
    sub_var: 3
    sub_var2: 4

TaskTwo:
  new_param1: 3
  new_param2: 4

# ...
...
```

In the first document, the header, it is important that the `work_dir` is properly specified. This is the root directory from which `Task` outputs will be written, and the LUTE database will be stored. It may also be desirable to modify the `task_timeout` parameter which defines the time limit for individual `Task` jobs. By default it is set to 10 minutes, although this may not be sufficient for long running jobs. This value will be applied to **all** `Task`s so should account for the longest running job you expect.

The actual analysis parameters are defined in the second document. As these vary from `Task` to `Task`, a full description will not be provided here. An actual template with real `Task` parameters is available in `config/test.yaml`. Your analysis POC can also help you set up and choose the correct `Task`s to include as a starting point. The template YAML file has further descriptions of what each parameter does and how to fill it out. You can also refer to the `lute_help` program described under the following sub-heading.

Some things to consider and possible points of confusion:
- While we will be submitting **managed** `Task`s, the parameters are defined at the `Task` level. I.e. the **managed** `Task` and `Task` itself have different names, and the names in the YAML refer to the latter. This is because a single `Task` can be run using different `Executor` configurations, but using the same parameters. The list of **managed** `Task`s is in `lute/managed_tasks.py`. A table is also provided below for some routines of interest..


| **Managed** `Task`       | The `Task` it Runs       | `Task` Description                                             |
|:------------------------:|:------------------------:|:--------------------------------------------------------------:|
| `SmallDataProducer`      | `SubmitSMD`              | Smalldata production                                           |
| `CrystFELIndexer`        | `IndexCrystFEL`          | Crystallographic indexing                                      |
| `PartialatorMerger`      | `MergePartialator`       | Crystallographic merging                                       |
| `HKLComparer`            | `CompareHKL`             | Crystallographic figures of merit                              |
| `HKLManipulator`         | `ManipulateHKL`          | Crystallographic format conversions                            |
| `DimpleSolver`           | `DimpleSolve`            | Crystallographic structure solution with molecular replacement |
| `PeakFinderPyAlgos`      | `FindPeaksPyAlgos`       | Peak finding with PyAlgos algorithm.                           |
| `PeakFinderPsocake`      | `FindPeaksPsocake`       | Peak finding with psocake algorithm.                           |
| `StreamFileConcatenator` | `ConcatenateStreamFiles` | Stream file concatenation.                                     |


### How do I know what parameters are available, and what they do?
A summary of `Task` parameters is available through the `lute_help` program.

```bash
> utilities/lute_help -t [TaskName]
```

Note, some parameters may say "Unknown description" - this either means they are using an old-style defintion that does not include parameter help, or they may have some internal use. In particular you will see this for `lute_config` on every `Task`, this parameter is filled in automatically and should be ignored. E.g. as an example:

```bash
> utilities/lute_help -t IndexCrystFEL
INFO:__main__:Fetching parameter information for IndexCrystFEL.
IndexCrystFEL
-------------
Parameters for CrystFEL's `indexamajig`.

There are many parameters, and many combinations. For more information on
usage, please refer to the CrystFEL documentation, here:
https://www.desy.de/~twhite/crystfel/manual-indexamajig.html


Required Parameters:
--------------------
[...]

All Parameters:
-------------
[...]

highres (number)
	Mark all pixels greater than `x` has bad.

profile (boolean) - Default: False
	Display timing data to monitor performance.

temp_dir (string)
	Specify a path for the temp files folder.

wait_for_file (integer) - Default: 0
	Wait at most `x` seconds for a file to be created. A value of -1 means wait forever.

no_image_data (boolean) - Default: False
	Load only the metadata, no iamges. Can check indexability without high data requirements.

[...]
```

## Running Managed `Task`s and Workflows (DAGs)
After a YAML file has been filled in you can run a `Task`. There are multiple ways to submit a `Task`, but there are 3 that are most likely:
1. Run a single **managed** `Task` interactively by running `python ...`
2. Run a single **managed** `Task` as a batch job (e.g. on S3DF) via a SLURM submission `submit_slurm.sh ...`
3. Run a DAG (workflow with multiple **managed** `Task`s).

These will be covered in turn below; however, in general all methods will require two parameters: the path to a configuration YAML file, and the name of the **managed** `Task` or workflow you want to run. When submitting via SLURM or submitting an entire workflow there are additional parameters to control these processes.


### Running single managed `Task`s interactively
The simplest submission method is just to run Python interactively. In most cases this is not practical for long-running analysis, but may be of use for short `Task`s or when debugging. From the root directory of the LUTE repository (or after installation) you can use the `run_task.py` script:

```bash
> python -B [-O] run_task.py -t <ManagedTaskName> -c </path/to/config/yaml>
```

The command-line arguments in square brackets `[]` are optional, while those in `<>` must be provided:
- `-O` is the flag controlling whether you run in debug or non-debug mode. **By default, i.e. if you do NOT provide this flag you will run in debug mode** which enables verbose printing. Passing `-O` will turn off debug to minimize output.
- `-t <ManagedTaskName>` is the name of the **managed** `Task` you want to run.
- `-c </path/...>` is the path to the configuration YAML.

### Submitting a single managed `Task` as a batch job
On S3DF you can also submit individual **managed** `Task`s to run as batch jobs. To do so use `launch_scripts/submit_slurm.sh`

```bash
> launch_scripts/submit_slurm.sh -t <ManagedTaskName> -c </path/to/config/yaml> [--debug] $SLURM_ARGS
```

As before command-line arguments in square brackets `[]` are optional, while those in `<>` must be provided
- `-t <ManagedTaskName>` is the name of the **managed** `Task` you want to run.
- `-c </path/...>` is the path to the configuration YAML.
- `--debug` is the flag to control whether or not to run in debug mode.

In addition to the LUTE-specific arguments, SLURM arguments must also be provided (`$SLURM_ARGS` above). You can provide as many as you want; however you will need to at least provide:
- `--partition=<partition/queue>` - The queue to run on, in general for LCLS this is `milano`
- `--account=lcls:<experiment>` - The account to use for batch job accounting.

You will likely also want to provide at a minimum:
- `--ntasks=<...>` to control the number of cores in allocated.

In general, it is best to prefer the long-form of the SLURM-argument (`--arg=<...>`) in order to avoid potential clashes with present or future LUTE arguments.

### Workflow (DAG) submission
Finally, you can submit a full workflow (e.g. SFX analysis, smalldata production and summary results, geometry optimization...). This can be done using a single script, `submit_launch_airflow.sh`, similarly to the SLURM submission above:

```bash
> launch_scripts/submit_launch_airflow.sh /path/to/lute/launch_scripts/launch_airflow.py -c </path/to/yaml.yaml> -w <dag_name> [--debug] [--test] $SLURM_ARGS
```
The submission process is slightly more complicated in this case. A more in-depth explanation is provided under "Airflow Launch Steps", in the advanced usage section below if interested. The parameters are as follows - as before command-line arguments in square brackets `[]` are optional, while those in `<>` must be provided:
- The **first argument** (must be first) is the full path to the `launch_scripts/launch_airflow.py` script located in whatever LUTE installation you are running. All other arguments can come afterwards in any order.
- `-c </path/...>` is the path to the configuration YAML to use.
- `-w <dag_name>` is the name of the DAG (workflow) to run. This replaces the task name provided when using the other two methods above. A DAG list is provided below.
- `--debug` controls whether to use debug mode (verbose printing)
- `--test` controls whether to use the test or production instance of Airflow to manage the DAG.

The `$SLURM_ARGS` must be provided in the same manner as when submitting an individual **managed** `Task` by hand to be run as batch job with the script above. **Note** that these parameters will be used as the starting point for the SLURM arguments of **every managed** `Task` in the DAG; however, individual steps in the DAG may have overrides built-in where appropriate to make sure that step is not submitted with potentially incompatible arguments.

**DAG List**
- `find_peaks_index`
- `psocake_sfx_phasing`
- `pyalgos_sfx`


# Advanced Usage
## Variable Substitution in YAML Files
Using `validator`s, it is possible to define (generally, default) model parameters for a `Task` in terms of other parameters. It is also possible to use validated Pydantic model parameters to substitute values into a configuration file required to run a third party `Task` (e.g. some `Task`s may require their own JSON, TOML files, etc. to run properly). For more information on these types of substitutions, refer to the `new_task.md` documentation on `Task` creation.

These types of substitutions, however, have a limitation in that they are not easily adapted at run time. They therefore address only a small number of the possible combinations in the dependencies between different input parameters. In order to support more complex relationships between parameters, variable substitutions can also be used in the configuration YAML itself. Using a syntax similar to `Jinja` templates, you can define values for YAML parameters in terms of other parameters or environment variables. The values are substituted before Pydantic attempts to validate the configuration.

It is perhaps easiest to illustrate with an example. A test case is provided in `config/test_var_subs.yaml` and is reproduced here:

```yaml
%YAML 1.3
---
title: "Configuration to Test YAML Substitution"
experiment: "TestYAMLSubs"
run: 12
date: "2024/05/01"
lute_version: 0.1
task_timeout: 600
work_dir: "/sdf/scratch/users/d/dorlhiac"
...
---
OtherTask:
  useful_other_var: "USE ME!"

NonExistentTask:
  test_sub: "/path/to/{{ experiment }}/file_r{{ run:04d }}.input"         # Substitute `experiment` and `run` from header above
  test_env_sub: "/path/to/{{ $EXPERIMENT }}/file.input"                   # Substitute from the environment variable $EXPERIMENT
  test_nested:
    a: "outfile_{{ run }}_one.out"                                        # Substitute `run` from header above
    b:
      c: "outfile_{{ run }}_two.out"                                      # Also substitute `run` from header above
      d: "{{ OtherTask.useful_other_var }}"                               # Substitute `useful_other_var` from `OtherTask`
  test_fmt: "{{ run:04d }}"                                               # Subsitute `run` and format as 0012
  test_env_fmt: "{{ $RUN:04d }}"                                          # Substitute environment variable $RUN and pad to 4 w/ zeros
...
```

Input parameters in the config YAML can be substituted with either other input parameters or environment variables, with or without limited string formatting. All substitutions occur between double curly brackets: `{{ SUB_VAR }}`. Environment variables are indicated by `$` in front of the variable name. Parameters from the header, i.e. the first YAML document (top section) containing the `run`, `experiment`, version fields, etc. can be substituted without any qualification. If you want to use the `run` parameter, you can substitute it using `{{ run }}`. All other parameters, i.e. from other `Task`s or within `Task`s, must use a qualified name. Nested levels are delimited using a `.`. E.g. consider a structure like:

```yaml
Task:
  param_set:
    a: 1
    b: 2
    c: 3
```
In order to use parameter `c`, you would use `{{ Task.param_set.c }}` as the substitution.

Take care when using substitutions! This process will not try and guess for you. When a substitution is not available, e.g. due to misspelling, one of two things will happen:
- If it was an environment variable that does not exist, no substitution will be performed, although a message will be printed. I.e. you will be left with `param: /my/failed/{{ $SUBSTITUTION }}` as your parameter. This may or may not fail the model validation step, but is not what you intended.
- If it was an attempt at substituting another YAML parameter which does not exist, an exception will be thrown and the program will exit.

### Gotchas!
**Found unhashable key**
To avoid YAML parsing issues when using the substitution syntax, be sure to quote your substitutions, partcularly if also using string formatting options. Before substitution is performed, a dictionary is first constructed by the `pyyaml` package which parses the document - it may fail to parse the document and raise an exception if the substitutions are not quoted.
E.g.
```yaml
# USE THIS
MyTask:
  var_sub: "{{ other_var:04d }}"

# **DO NOT** USE THIS
MyTask:
  var_sub: {{ other_var:04d }}
```

## Debug Environment Variables
Special markers have been inserted at certain points in the execution flow for LUTE. These can be enabled by setting the environment variables detailed below. These are intended to allow developers to exit the program at certain points to investigate behaviour or a bug. For instance, when working on configuration parsing, an environment variable can be set which exits the program after passing this step. This allows you to run LUTE otherwise as normal (described above), without having to modify any additional code or insert your own early exits.

Types of debug markers:
- `LUTE_DEBUG_EXIT`: Will exit the program at this point if the corresponding environment variable has been set.

Developers can insert these markers as needed into their code to add new exit points, although as a rule of thumb they should be used sparingly, and generally only after major steps in the execution flow (e.g. after parsing, after beginning a task, after returning a result, etc.).

In order to include a new marker in your code:
```py
from lute.execution.debug_utils import LUTE_DEBUG_EXIT

def my_code() -> None:
    # ...
    LUTE_DEBUG_EXIT("MYENVVAR", "Additional message to print")
    # If MYENVVAR is not set, the above function does nothing
```

You can enable a marker by setting to 1, e.g. to enable the example marker above while running `Tester`:
```bash
MYENVVAR=1 python -B run_task.py -t Tester -c config/test.yaml
```

### Currently used environment variables
- `LUTE_DEBUG_EXIT_AT_YAML`: Exits the program after reading in a YAML configuration file and performing variable substitutions, but BEFORE Pydantic validation.
- `LUTE_DEBUG_BEFORE_TPP_EXEC`: Exits the program after a ThirdPartyTask has prepared its submission command, but before `exec` is used to run it.

## Airflow Launch and DAG Execution Steps
The Airflow launch process actually involves a number of steps, and is rather complicated. There are two wrapper steps prior to getting to the actual Airflow API communication.
1. `launch_scripts/submit_launch_airflow.sh` is run.
2. This script calls `/sdf/group/lcls/ds/tools/lute_launcher` with all the same parameters that it was called with.
3. `lute_launcher` runs the `launch_scripts/launch_airflow.py` script which was provided as the first argument. This is the **true** launch script
4. `launch_airflow.py` communicates with the Airflow API, requesting that a specific DAG be launched. It then continues to run, and gathers the individual logs and the exit status of each step of the DAG.
5. Airflow will then enter a loop of communication where it asks the JID to submit each step of the requested DAG as batch job using `launch_scripts/submit_slurm.sh`.

There are some specific reasons for this complexity:
- The use of `submit_launch_airflow.sh` as a thin-wrapper around `lute_launcher` is to allow the true Airflow launch script to be a long-lived job. This is for compatibility with the eLog and the ARP. When run from the eLog as a workflow, the job submission process must occur within 30 seconds due to a timeout built-in to the system. This is fine when submitting jobs to run on the batch-nodes, as the submission to the queue takes very little time. So here, `submit_launch_airflow.sh` serves as a thin script to have `lute_launcher` run as a batch job. It can then run as a long-lived job (for the duration of the entire DAG) collecting log files all in one place. This allows the log for each stage of the Airflow DAG to be inspected in a single file, and through the eLog browser interface.
- The use `lute_launcher` as a wrapper around `launch_airflow.py` is to manage authentication and credentials. The `launch_airflow.py` script requires loading credentials in order to authenticate against the Airflow API. For the average user this is not possible, unless the script is run from within the `lute_launcher` process.
