# Task Configuration
## Preparing a Configuration YAML
All `Task`s are parameterized through a single configuration YAML file - even third party code which requires its own configuration files is managed through this YAML file. By default all parameters are taken from this YAML document, and when the `Task` is completed (whether successful or not) the parameters and results are written to a database. Submissions are generally done on a run-by-run basis, however, because the database stores parameter sets, there are mechanisms to use historic data to repopulate the YAML, or to manage parameterizing `Task`s for multiple runs at a time. These more advanced mechanisms are described in the following sections on `RUN_GROUPS` and `lute_cfg` following a general overview of the YAML structure.

The basic structure is split into two documents, a brief header section which contains information that is applicable across all `Task`s, such as the experiment name, run numbers and the working directory, followed by per `Task` parameters:
```yaml
%YAML 1.3
---
title: "Some title."
experiment: "MYEXP123"
# run: 12 # Does not need to be provided
date: "2024/05/01"
lute_version: 0.3.0
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

**Some things to consider and possible points of confusion:**

- While we will be submitting **managed** `Task`s, the parameters are defined at the `Task` level. I.e. the **managed** `Task` and `Task` itself have different names, and the names in the YAML refer to the latter. This is because a single `Task` can be run using different `Executor` configurations, but using the same parameters. The list of **managed** `Task`s is in `lute/managed_tasks.py`. A (non-exhaustive) set is also provided in the tables below for some routines of interest..

**Crystallography**

| **Managed** `Task`       | The `Task` it Runs       | `Task` Description                                             |
|:------------------------:|:------------------------:|:--------------------------------------------------------------:|
| `CrystFELIndexer`        | `IndexCrystFEL`          | Crystallographic indexing                                      |
| `PartialatorMerger`      | `MergePartialator`       | Crystallographic merging                                       |
| `HKLComparer`            | `CompareHKL`             | Crystallographic figures of merit                              |
| `HKLManipulator`         | `ManipulateHKL`          | Crystallographic format conversions                            |
| `DimpleSolver`           | `DimpleSolve`            | Crystallographic structure solution with molecular replacement |
| `PeakFinderPyAlgos`      | `FindPeaksPyAlgos`       | Peak finding with PyAlgos algorithm.                           |
| `PeakFinderPsocake`      | `FindPeaksPsocake`       | Peak finding with psocake algorithm.                           |
| `StreamFileConcatenator` | `ConcatenateStreamFiles` | Stream file concatenation.                                     |

**General Reduction and HDF5 Production**

| **Managed** `Task`       | The `Task` it Runs       | `Task` Description                                             |
|:------------------------:|:------------------------:|:--------------------------------------------------------------:|
| `SmallDataProducer`      | `SubmitSMD`              | Smalldata production (LCLS1)                                   |
| `SmallDataProducer2`     | `SubmitSMD`              | Smalldata production (LCLS2)                                   |
| `SmallDataProducer2Test` | `SubmitSMD`              | Smalldata production (LCLS2 Test Environment)                  |
| `SmallDataProducerXpp`   | `SubmitSMD`              | Smalldata production (GPU Utilities Environment)               |

**Geometry and Conversion Utilites**

| **Managed** `Task`       | The `Task` it Runs       | `Task` Description                                             |
|:------------------------:|:------------------------:|:--------------------------------------------------------------:|
| `BayFAIOptimizer`        | `BayFAI`                 | Geometry optimization with BayFAI (LCLS1)                      |
| `BayFAIOptimizer2`       | `BayFAI`                 | Geometry optimization with BayFAI (LCLS2)                      |
| `Xtc1Reader`             | `ReadXtc1`               | Read XTC1 datasets to convert to XTC2 (use with `Xtc2Writer`)  |
| `Xtc2Writer`             | `WriteXtc2`              | Write XTC2 datasets received from `Xtc1Reader`                 |

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

`lute_help` can also be used to retrieve a list of all currently available `Task`s. (Truncated for brevity)

```bash
> ./lute/utilities/lute_help -l
INFO:__main__:Fetching Task list.
Task List
---------

- AnalyzeSmallDataXAS
  Current Managed Tasks: SmallDataXASAnalyzer,
	TaskParameter model for AnalyzeSmallDataXAS Task.

	This Task does basic analysis of XAS data based on a SmallData HDF5 output
	file. It calculates difference absorption and signal binned by various
	scanned motors.

- AnalyzeSmallDataXES
  Current Managed Tasks: SmallDataXESAnalyzer,
	TaskParameter model for AnalyzeSmallDataXES Task.

	This Task does basic analysis of XES data based on a SmallData HDF5 output
	file. It calculates difference emission and signal binned by various
	scanned motors.
```

**NOTE:** Associated **managed** `Task`s are provided under each `Task` name. These will be the names passed to submission scripts.

## `RUN_GROUPS`: Managing and reprocessing sets of runs

By default, any parameters that are not declared with default values by the `Task` parameter model, are filled with the values found in the YAML file. When beginning processing for an experiment this is generally what you want - you declare the values you want to use. However, over time, you may need to reprocess old runs, or want to try changing some values, but not others.

On its own, YAML does not support complex conditional logic or control structures. The database that LUTE uses, though, maintains all the necessary information to allow using previous parameter sets - it just needs to be accessible during configuration. The `lute_cfg` tool described in the next section shows how you can recreate an entire YAML file, but often you only want to access certain parameters, not all of them. This is where `RUN_GROUPS` come in.

### Creating Semantic Ranges with `RUN_GROUPS`

`RUN_GROUPS` are essentially semantic labels attached to some set or range of experiment runs. Once defined, they can then be referred to in order to define parameter policies for the entire set of runs at a time.

To declare a group, you can define a label in the first global configuration document of the YAML file under the `RUN_GROUPS` key:

```yaml
%YAML 1.3
---
title: "LUTE Task Configuration"
experiment: "FAKEEXPERIMENT"
run: "{{ $RUN }}"
work_dir: "/my/fake/work/directory"

# Setup RUN_GROUPS once globally - Give you semantic range of runs
RUN_GROUPS:
  DARK_RUNS:
    applies_to:
      range: [0, 40]
...
---
SubmitSMD:
  # My parameters....
...
```

In this case, we have defined a group called `DARK_RUNS`. It applies to the range of runs `0-40`. These are inclusive of the bounds (i.e., it is a closed set).

The groups can be configured  either using a range as above, or by providing a specific list of individual runs. For example, we declare two groups below, using these two mechanisms:

```yaml
%YAML 1.3
---
title: "LUTE Task Configuration"
experiment: "FAKEEXPERIMENT2"
run: "{{ $RUN }}"
work_dir: "/my/fake/work/directory"

RUN_GROUPS:
  SAMPLE1:
    applies_to:
      range: [0, 40]
  SAMPLE2:
    applies_to:
      runs: [4, 13, 50]
...
---
SubmitSMD:
  # My parameters....
...
```

Here we defined two different groups:

- `SAMPLE1`: Includes all runs in the range `0-40`
- `SAMPLE2`: Includes, specifically, runs `4`, `13`, and `50`

When a run is defined in two groups, the more specific, i.e., a list of runs instead of a range, will win.

The `RUN_GROUPS` field can also appear under the individual `Task` configuration blocks. However, while the run range, or list or list of runs, can be defined in this `Task` block, the intention is that the global section defines the applicability. In the `Task` section, the policies to apply to the group are defined as discussed in the next section.

### Applying policies to `RUN_GROUPS`

At this point, we have defined a label that applies to a group of runs, but this doesn't help us yet with configuration. This is where `Task` definitions of policies come in.

As previously mentioned, the `RUN_GROUPS` header can appear both in the global section, and under each `Task`. Globally, we define the applicability of the label to a set of runs. In each `Task`, we define how we want to fill in the parameters for that label.

In particular, there are three policies that can be used by using special directives:

- `@db:current` : Use the most recent parameter value that **was stored in the database FOR THIS RUN**. I.e. if I have previously processed run 15, and am now processing run 15 again, use the value that was logged in the database, not whatever may be present in the YAML file. This allows you to go back and reprocess datasets after having changed the values in the file.
- `@db:run:NNN` : Use the most recent parameter value that **was stored in the database for RUN NNN**. I.e., if I specify `@db:run:15`, regardless of which run I am currently processing, I will use the value that was logged in the database for RUN 15. That is instead of using the value that was stored for this run, or the value that is currently in the YAML file.
- `@literal` : Use the value from the YAML value as it is literally written. This is implicitly also the final global fallback. In the event that no database entries are found, you will eventually fallback to using the literal values from the YAML file.

These directives appear underneath the `Task` configuration and can be applied to all the parameters using the `default` field, and then overridden for specific parameters by name.

It is most helpful to view some examples. In the following case, we have again defined one group globally, we then set the default policy to be to use the database value for the run being processed, except for the parameter `ROIs`.

```yaml
%YAML 1.3
---
title: "LUTE Task Configuration"
experiment: "FAKEEXPERIMENT2"
run: "{{ $RUN }}"
work_dir: "/my/fake/work/directory"

RUN_GROUPS:
  SAMPLE1:
    applies_to:
      range: [0, 40]
...
---
SubmitSMD:
  RUN_GROUPS:
    SAMPLE1:
      default: @db:current
      ROIs: @db:run:50
  ROIs: [[[1, 2], [157, 487], [294, 598]]]
  other_param: 10
  another: "string_value"
...
```

What does this mean? Let's unpack under a few situations. Let's assume that we have processed runs 0 through 60 at least once.

1. We are processing run 20. The group defines a range of `0-40` for included runs, so run 20 falls within it. The default policy is `@db:current`, so for `other_param` and `another`, if there are existing entries for run 20 in the database (i.e., we previously processed run 20), their values we will be taken from there. `ROIs`, on the other hand, is using `@db:run:50`. That means we use the value of `ROIs` recorded for **RUN 50** in the database - even though we are processing run 20.
2. We are processing run 70, for the first time. In this case, since there are no database values to use, the most general fallback option is always to use the values that are currently defined in the YAML file.

### Resolving conflicts - overload resolution

When there are possibly conflicting definitions or policies, the following rules are used to determine which is used. The entire `RUN_GROUPS` block can appear in both the global section and for each `Task` configuration. For simplicity, though, the recommendation is that globally, we define the ranges/lists of runs for various group labels. Then per-`Task` we define the policies for the parameters.

More specifically, though, the following rules are used:

1. For every `Task`, the set of `RUN_GROUPS` that apply to it is the union of the global set, and the set under its configuration block.
    - **NOTE:** If a particular group is defined only under one `Task`, that definition is NOT seen by other `Task`s. Again, for this reason, we recommend that you always define the run group ranges globally.
2. If a range/list of runs is defined for a `RUN_GROUP` both globally, and at the `Task` level, the `Task` level takes precedence. I.e., a `Task` can override the range of runs that the group applies to.
3. If a default parameter policy appears both globally and at the `Task` level, the `Task` level takes precedence.
    - **NOTE:** This is only really applicable for default policies, as a per-parameter policy does not make sense at a global level. There are no global parameters.
4. If two groups define overlapping or conflicting ranges/lists of runs, the more specific group wins:
    - If one group defines a range, and another group a list which includes a run (or multiple runs) from the range, the group defined by a list takes precedence.
    - If two groups define overlapping ranges, the group with the narrower range takes precedence.

### Final Example with All Features

The following shows an example using all the policies and features for completeness:

```yaml
%YAML 1.3
---

RUN_GROUPS:
  SAMPLE1:
    applies_to:
      range: [0, 40]
  SAMPLE2:
    applies_to:
      runs: [39, 50, 60]
  SAMPLE3:
    applies_to:
      range: [5, 9]
...
---
Task1:
  RUN_GROUPS:
    SAMPLE1:
      default: @db:current
      param_x: @db:run:39
    SAMPLE2:
      default: @db:run:39
      param_y: @literal
    SAMPLE3:
      default: @db:run:5
  param_x: "abcd"
  param_y: 10
...
```

Assuming we have processed runs 0 through runs 60 (inclusive), if we go back and now reprocess the runs, the following parameter sets will be chosen:

- Runs 5-9 (inclusive): `SAMPLE3` is the applicable group. Only a default policy is provided, so for both `param_x` and `param_y`, the database value from run 5 will be used (based on the `@db:run:5` policy).
- Runs 10-38 (inclusive): `SAMPLE1` is the applicable group. We have both a default policy and a specific one, so for `param_y`, the value from the database for each individual run will be used (based on `@db:current`). For `param_x`, regardless of run, the database value from run 39 will be used.
- Run 39: `SAMPLE2` is the applicable group. We have both a default policy and a specific one, so for `param_x`, the value from the database for run 39 will be used (based on `@db:run:39`). For `param_y`, the value in the YAML file (`10`) will be used.

## Restoring a Configuration YAML from a Previous Run (`lute_cfg`)

The LUTE database holds a record of all parameters used for all `Task`s that have been run. It also contains information about the version of the code that was used each time any of those `Task`s was run. E.g., this may be important in the case of `Task`s, like `smalldata_tools` (`SubmitSMD`) that are run on top of checked-out repositories containing Python code that can be modified between executions.

LUTE provides a utility to extract the information from this database and, if relevant, restore various code bases to the state they were in when a specific experimental run was analyzed.

To use the utility, first source your standard Python environment and setup the LUTE installation. For example, on S3DF this may look like:

```bash
> cd /sdf/<MY EXPERIMENT OF INTEREST>/results/lute
> source /sdf/group/lcls/ds/ana/sw/conda2/manage/bin/psconda.sh
> source install/bin/activate_installation
```

Then you can run the `lute_cfg` utility to create a config YAML based on the datbase for a particular run.

```bash
> lute_cfg --help
usage: lute_cfg [-h] -d DATABASE -r RUN -o OUTPUT

Reconstruct LUTE config YAML and restore git repositories to their state during a specific run.

optional arguments:
  -h, --help            show this help message and exit
  -d DATABASE, --database DATABASE
                        Path to the LUTE database.
  -r RUN, --run RUN     Run number to reconstruct config for.
  -o OUTPUT, --output OUTPUT
                        Location to output the new config YAML (directory or file path).
```

The YAML will be output to the location indicated by the `-o` flag. It does NOT overwrite any other YAML files unless you set this flag to the path of an existing file. The `-d` flag is used to point to the LUTE sqlite database. For example, on S3DF this is usually in `/sdf/<MY EXPERIMENT OF INTEREST/results/lute_output/lute.db`.

## Variable Substitution in YAML Files
Using `validator`s, it is possible to define (generally, default) model parameters for a `Task` in terms of other parameters. It is also possible to use validated Pydantic model parameters to substitute values into a configuration file required to run a third party `Task` (e.g. some `Task`s may require their own JSON, TOML files, etc. to run properly). For more information on these types of substitutions, refer to the `Creating a new Task` documentation on `Task` creation (in the Developer Documentation).

These types of substitutions, however, have a limitation in that they are not easily adapted at run time. They therefore address only a small number of the possible combinations in the dependencies between different input parameters. In order to support more complex relationships between parameters, variable substitutions can also be used in the configuration YAML itself. Using a syntax similar to `Jinja` templates, you can define values for YAML parameters in terms of other parameters or environment variables. The values are substituted before Pydantic attempts to validate the configuration.

It is perhaps easiest to illustrate with an example. A test case is provided in `config/test_var_subs.yaml` and is reproduced here:

```yaml
%YAML 1.3
---
title: "Configuration to Test YAML Substitution"
experiment: "TestYAMLSubs"
run: 12
date: "2024/05/01"
lute_version: 0.3.0
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

Input parameters in the config YAML can be substituted with either other input parameters or environment variables, with or without limited string formatting. All substitutions occur between double curly brackets: `{{ VARIABLE_TO_SUBSTITUTE }}`. Environment variables are indicated by `$` in front of the variable name. Parameters from the header, i.e. the first YAML document (top section) containing the `run`, `experiment`, version fields, etc. can be substituted without any qualification. If you want to use the `run` parameter, you can substitute it using `{{ run }}`. All other parameters, i.e. from other `Task`s or within `Task`s, must use a qualified name. Nested levels are delimited using a `.`. E.g. consider a structure like:

```yaml
Task:
  param_set:
    a: 1
    b: 2
    c: 3
```
In order to use parameter `c`, you would use `{{ Task.param_set.c }}` as the substitution.

Take care when using substitutions! This process will not try to guess for you. When a substitution is not available, e.g. due to misspelling, one of two things will happen:

- If it was an environment variable that does not exist, no substitution will be performed, although a message will be printed. I.e. you will be left with `param: /my/failed/{{ $SUBSTITUTION }}` as your parameter. This may or may not fail the model validation step, but is likely not what you intended.
- If it was an attempt at substituting another YAML parameter which does not exist, an exception will be thrown and the program will exit.

**Defining your own parameters**

The configuration file is **not** validated in its totality, only on a `Task`-by-`Task` basis, but it **is read** in its totality. E.g. when running `MyTask` only that portion of the configuration is validated even though the entire file has been read, and is available for substitutions. As a result, it is safe to introduce extra entries into the YAML file, as long as they are not entered under a specific `Task`'s configuration. This may be useful to create your own global substitutions, for example if there is a key variable that may be used across different `Task`s.
E.g. Consider a case where you want to create a more generic configuration file where a single variable is used by multiple `Task`s. This single variable may be changed between experiments, for instance, but is likely static for the duration of a single set of analyses. In order to avoid a mistake when changing the configuration between experiments you can define this special variable (or variables) as a separate entry in the YAML, and make use of substitutions in each `Task`'s configuration. This way the variable only needs to be changed in one place.

```yaml
# Define our substitution. This is only for substitutiosns!
MY_SPECIAL_SUB: "EXPMT_DEPENDENT_VALUE"  # Can change here once per experiment!

RunTask1:
  special_var: "{{ MY_SPECIAL_SUB }}"
  var_1: 1
  var_2: "a"
  # ...

RunTask2:
  special_var: "{{ MY_SPECIAL_SUB }}"
  var_3: "abcd"
  var_4: 123
  # ...

RunTask3:
  special_var: "{{ MY_SPECIAL_SUB }}"
  #...

# ... and so on
```

### Gotchas!
**Order matters**

While in general you can use parameters that appear later in a YAML document to substitute for values of parameters that appear earlier, the substitutions themselves will be performed in order of appearance. It is therefore **NOT possible** to correctly use a later parameter as a substitution for an earlier one, if the later one itself depends on a substitution. The YAML document, however, can be rearranged without error. The order in the YAML document has no effect on execution order which is determined purely by the workflow definition. As mentioned above, the document is not validated in its entirety so rearrangements are allowed. For example consider the following situation which produces an incorrect substitution:


```yaml
%YAML 1.3
---
title: "Configuration to Test YAML Substitution"
experiment: "TestYAMLSubs"
run: 12
date: "2024/05/01"
lute_version: 0.3.0
task_timeout: 600
work_dir: "/sdf/data/lcls/ds/exp/experiment/scratch"
...
---
RunTaskOne:
  input_dir: "{{ RunTaskTwo.path }}"  # Will incorrectly be "{{ work_dir }}/additional_path/{{ $RUN }}"
  # ...

RunTaskTwo:
  # Remember `work_dir` and `run` come from the header document and don't need to
  # be qualified
  path: "{{ work_dir }}/additional_path/{{ run }}"
...
```

This configuration can be rearranged to achieve the desired result:

```yaml
%YAML 1.3
---
title: "Configuration to Test YAML Substitution"
experiment: "TestYAMLSubs"
run: 12
date: "2024/05/01"
lute_version: 0.3.0
task_timeout: 600
work_dir: "/sdf/data/lcls/ds/exp/experiment/scratch"
...
---
RunTaskTwo:
  # Remember `work_dir` comes from the header document and doesn't need to be qualified
  path: "{{ work_dir }}/additional_path/{{ run }}"

RunTaskOne:
  input_dir: "{{ RunTaskTwo.path }}"  # Will now be /sdf/data/lcls/ds/exp/experiment/scratch/additional_path/12
  # ...
...
```

On the otherhand, relationships such as these may point to inconsistencies in the dependencies between `Task`s which may warrant a refactor.

**Found unhashable key**

To avoid YAML parsing issues when using the substitution syntax, be sure to quote your substitutions. Before substitution is performed, a dictionary is first constructed by the `pyyaml` package which parses the document - it may fail to parse the document and raise an exception if the substitutions are not quoted.
E.g.
```yaml
# USE THIS
MyTask:
  var_sub: "{{ other_var:04d }}"

# **DO NOT** USE THIS
MyTask:
  var_sub: {{ other_var:04d }}
```

During validation, Pydantic will by default cast variables if possible, because of this it is generally safe to use strings for substitutions. E.g. if your parameter is expecting an integer, and after substitution you pass `"2"`, Pydantic will cast this to the `int` `2`, and validation will succeed. As part of the substitution process limited type casting will also be handled if it is necessary for any formatting strings provided. E.g. `"{{ run:04d }}"` requires that run be an integer, so it will be treated as such in order to apply the formatting.
