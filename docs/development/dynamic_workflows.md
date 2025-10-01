# Dynamic Run-time workflows
## Run-time DAG creation
In most cases, standard workflows should be defined explicitly in Airflow. However, LUTE also provides support for the creation of DAGs dynamically by passing the definition of the workflow in the form of a dictionary. All the supported workflow backends (Airflow, prefect, etc.) will then construct the workflow as it is running.

A basic YAML syntax is used to construct a series of nested dictionaries which define a DAG. We can consider a serial-femtosecond crystallography DAG which may be given in the Airflow syntax as:

```python
peak_finder >> indexer >> merger >> hkl_comparer
```

We can alternatively define this DAG in YAML:

```yaml
task_name: PeakFinderPyAlgos
slurm_params: ''
next:
- task_name: CrystFELIndexer
  slurm_params: ''
  next: []
  - task_name: PartialatorMerger
    slurm_params: ''
    next: []
    - task_name: HKLComparer
      slurm_params: ''
      next: []
```

I.e. we define a tree where each node is constructed using `Node(task_name: str, slurm_params: str, next: List[Node])`.

- The `task_name` is the name of a **managed** `Task` as before, in the same way that would be passed to the `JIDSlurmOperator` in Airflow.
- A custom string of slurm arguments can be passed using `slurm_params`. This is a complete string of **all** the arguments to use for the corresponding **managed** `Task`. Use of this field is **all or nothing!** - if it is left as an empty string, the default parameters (passed on the command-line using the launch script) are used, otherwise this string is used in its stead. Because of this **remember to include a partition and account** if using it.
- The `next` field is composed of either an empty list (meaning no **managed** `Task`s are run after the current node), or additional nodes. All nodes in the list are run in parallel. 

As a second example, to run `task1` followed by `task2` and `task3` in parellel we would use:

```yaml
task_name: Task1
slurm_params: ''
next:
- task_name: Task2
  slurm_params: ''
  next: []
- task_name: Task3
  slurm_params: ''
  next: []
```

## Launching dynamic workflows
In order to run a DAG defined this way we pass the **path** to the YAML file we have defined it in to the launch script using `-W <path_to_dag>`. This is instead of calling it by name. E.g.

### Airflow

```bash
/path/to/lute/launch_scripts/submit_launch_airflow.sh /path/to/lute/launch_scripts/launch_airflow.py -e <exp> -r <run> -c /path/to/config -W <path_to_dag> --test [--debug] [SLURM_ARGS]
```

Note that fewer options are currently supported for configuring the operators for each step of the DAG. The slurm arguments can be replaced in their entirety using a custom `slurm_params` string but individual options cannot be modified.

### Prefect
```bash
/path/to/lute/launch_scripts/submit_launch_prefect.sh /path/to/lute/launch_scripts/launch_prefect.py -e <exp> -r <run> -c /path/to/config -W <path_to_dag> [--debug] [SLURM_ARGS]
```

### SLURM
```bash
/path/to/lute/launch_scripts/submit_launch_slurm.sh /path/to/lute/launch_scripts/launch_slurm.py -e <exp> -r <run> -c /path/to/config -W <path_to_dag> [--debug] [SLURM_ARGS]
```
## Notes and limitations
### Airflow
Airflow was not designed with this use-case in mind, so there are some caveats when launching one of these dynamic DAGs via Airflow. In particular, due to the mechanism used to store the dynamic workflow definition on the Airflow, it is currently only possible to run a **single** custom defintion at a time. I.e. two users **cannot** launch workflows with different YAML definitions concurrently on the same instance of Airflow (one could use the test instance, and the other production instance without conflict).

Seemingly, the process also appears to be slightly less stable, for reasons that are not understood, so it may be necessary to launch a dynamic workflow twice on occassion.

### Prefect
Prefect lends itself more naturally to this use-case, so there shouldn't be any issues with workflows designed in this manner when launched using this backend. Currently, the prefect backend only accepts these dynamic run-time workflow definitions, with no static workflows having been defined yet.

### SLURM job
The backup SLURM job-based workflow manager only accepts these dynamic workflow definitions. There are currently no plans to add support for providing static workflows using this backend.
