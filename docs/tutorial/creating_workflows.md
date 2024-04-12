# Workflows with Airflow
**Note:** Airflow uses the term **DAG**, or directed acyclic graph, to describe workflows of tasks with defined (and acyclic) connectivities. This page will use the terms workflow and DAG interchangeably.

## Relevant Components
In addition to the core LUTE package, a number of components are generally involved to run a workflow. The current set of scripts and objects are used to interface with Airflow, and the SLURM job scheduler. The core LUTE library can also be used to run workflows using different backends, and in the future these may be supported.

For building and running workflows using SLURM and Airflow, the following components are necessary, and will be described in more detail below:
- Airflow launch script: `launch_airflow.py`
- SLURM submission script: `submit_slurm.sh`
- Airflow operators:
  - `JIDSlurmOperator`

## Launch/Submission Scripts
## `launch_airflow.py`
Sends a request to an Airflow instance to submit a specific DAG (workflow). This script prepares an HTTP request with the appropriate parameters in a specific format.

A request involves the following information, most of which is retrieved automatically:
```py
dag_run_data: Dict[str, Union[str, Dict[str, Union[str, int, List[str]]]]] = {
    "dag_run_id": str(uuid.uuid4()),
    "conf": {
        "experiment": os.environ.get("EXPERIMENT"),
        "run_id": f"{os.environ.get('RUN_NUM')}{datetime.datetime.utcnow().isoformat()}",
        "JID_UPDATE_COUNTERS": os.environ.get("JID_UPDATE_COUNTERS"),
        "ARP_ROOT_JOB_ID": os.environ.get("ARP_JOB_ID"),
        "ARP_LOCATION": os.environ.get("ARP_LOCATION", "S3DF"),
        "Authorization": os.environ.get("Authorization"),
        "user": getpass.getuser(),
        "lute_params": params,
        "slurm_params": extra_args,
    },
}
```
Note that the environment variables are used to fill in the appropriate information because this script is intended to be launched primarily from the ARP (which passes these variables). The ARP allows for the launch job to be defined in the experiment eLog and submitted automatically for each new DAQ run. The environment variables `EXPERIMENT` and `RUN` can alternatively be defined prior to submitting the script on the command-line.

The script takes a number of parameters:

```bash
launch_airflow.py -c <path_to_config_yaml> -w <workflow_name> [--debug] [--test]
```

- `-c` refers to the path of the configuration YAML that contains the parameters for each **managed** `Task` in the requested workflow.
- `-w` is the name of the DAG (workflow) to run. By convention each DAG is named by the Python file it is defined in. (See below).
- `--debug` is an optional flag to run all steps of the workflow in debug mode for verbose logging and output.
- `--test` is an optional flag which will use the test Airflow instance. By default the script will make requests of the standard production Airflow instance.


## `submit_slurm.sh`
Launches a job on the S3DF batch nodes using the SLURM job scheduler. This script launches a single **managed** `Task` at a time. The usage is as follows:
```bash
submit_slurm.sh -c <path_to_config_yaml> -t <MANAGED_task_name> [--debug] [--SLURM_ARGS ...]
```
As a reminder the **managed** `Task` refers to the `Executor`-`Task` combination. The script does not parse any SLURM specific parameters, and instead passes them transparently to SLURM. At least the following two SLURM arguments must be provided:
```bash
--partition=<...> # Usually partition=milano
--account=<...> # Usually account=lcls:$EXPERIMENT
```
Generally, resource requests will also be included, such as the number of cores to use. A complete call may look like the following:
```bash
submit_slurm.sh -c /sdf/data/lcls/ds/hutch/experiment/scratch/config.yaml -t Tester --partition=milano --account=lcls:experiment --ntasks=100 [...]
```

When running a workflow using the `launch_airflow.py` script, each step of the workflow will be submitted using this script.

## Operators
`Operator`s are the objects submitted as individual steps of a DAG by Airflow. They are conceptually linked to the idea of a task in that each task of a workflow is generally an operator. Care should be taken, not to confuse them with LUTE `Task`s or **managed** `Task`s though. There is, however, usually a one-to-one correspondance between a `Task` and an `Operator`.

Airflow runs on a K8S cluster which has no access to the experiment data. When we ask Airflow to run a DAG, it will launch an `Operator` for each step of the DAG. However, the `Operator` itself cannot perform productive analysis without access to the data. The solution employed by `LUTE` is to have a limited set of `Operator`s which do not perform analysis, but instead request that a `LUTE` **managed** `Task`s be submitted on the batch nodes where it can access the data. There may be small differences between how the various provided `Operator`s do this, but in general they will all make a request to the **job interface daemon** (JID) that a new SLURM job be scheduled using the `submit_slurm.sh` script described above.

Therefore, running a typical Airflow DAG involves the following steps:
1. `launch_airflow.py` script is submitted, usually from a definition in the eLog.
2. The `launch_airflow` script requests that Airflow run a specific DAG.
3. The Airflow instance begins submitting the `Operator`s that makeup the DAG definition.
4. Each `Operator` sends a request to the `JID` to submit a job.
5. The `JID` submits the `elog_submit.sh` script with the appropriate **managed** `Task`.
6. The **managed** `Task` runs on the batch nodes, while the `Operator`, requesting updates from the JID on job status, waits for it to complete.
7. Once a **managed** `Task` completes, the `Operator` will receieve this information and tell the Airflow server whether the job completed successfully or resulted in failure.
8. The Airflow server will then launch the next step of the DAG, and so on, until every step has been executed.

Currently, the following `Operator`s are maintained:
- `JIDSlurmOperator`: The standard `Operator`. Each instance has a one-to-one correspondance with a LUTE **managed** `Task`.

### `JIDSlurmOperator` arguments
- `task_id`: This is nominally the name of the task on the Airflow side. However, for simplicity this is used 1-1 to match the name of a **managed** Task defined in LUTE's `managed_tasks.py` module. I.e., it should the name of an `Executor("Task")` object which will run the specific Task of interest. This **must** match the name of a defined managed Task.
- `max_cores`: Used to cap the maximum number of cores which should be requested of SLURM. By default all jobs will run with the same number of cores, which should be specified when running the `launch_airflow.py` script (either from the ARP, or by hand). This behaviour was chosen because in general we want to increase or decrease the core-count for all Tasks uniformly, and we don't want to have to specify core number arguments for each job individually. Nonetheless, on occassion it may be necessary to cap the number of cores a specific job will use. E.g. if the default value specified when launching the Airflow DAG is multiple cores, and one job is single threaded, the core count can be capped for that single job to 1, while the rest run with multiple cores.


# Creating a new workflow
Defining a new workflow involves creating a **new** module (Python file) in the directory `workflows/airflow`, creating a number of `Operator` instances within the module, and then drawing the connectivity between them. At the top of the file an Airflow DAG is created and given a name. By convention all `LUTE` workflows use the name of the file as the name of the DAG. The following code can be copied exactly into the file:

```py
from datetime import datetime
import os
from airflow import DAG
from lute.operators.jidoperators import JIDSlurmOperator # Import other operators if needed

dag_id: str = f"lute_{os.path.splitext(os.path.basename(__file__))[0]}"
description: str = (
    "Run SFX processing using PyAlgos peak finding and experimental phasing"
)

dag: DAG = DAG(
    dag_id=dag_id,
    start_date=datetime(2024, 3, 18),
    schedule_interval=None,
    description=description,
)
```

Once the DAG has been created, a number of `Operator`s must be created to run the various LUTE analysis operations. As an example consider a partial SFX processing workflow which includes steps for peak finding, indexing, merging, and calculating figures of merit. Each of the 4 steps will have an `Operator` instance which will launch a corresponding `LUTE` **managed** `Task`, for example:

```py
# Using only the JIDSlurmOperator
# syntax: JIDSlurmOperator(task_id="LuteManagedTaskName", dag=dag) # optionally, max_cores=123)
peak_finder: JIDSlurmOperator = JIDSlurmOperator(task_id="PeakFinderPyAlgos", dag=dag)

# We specify a maximum number of cores for the rest of the jobs.
indexer: JIDSlurmOperator = JIDSlurmOperator(
    max_cores=120, task_id="CrystFELIndexer", dag=dag
)

# Merge
merger: JIDSlurmOperator = JIDSlurmOperator(
    max_cores=120, task_id="PartialatorMerger", dag=dag
)

# Figures of merit
hkl_comparer: JIDSlurmOperator = JIDSlurmOperator(
    max_cores=8, task_id="HKLComparer", dag=dag
)
```

Finally, the dependencies between the `Operator`s are "drawn", defining the execution order of the various steps. The `>>` operator has been overloaded for the `Operator` class, allowing it to be used to specify the next step in the DAG. In this case, a completely linear DAG is drawn as:

```py
peak_finder >> indexer >> merger >> hkl_comparer
```

Parallel execution can be added by using the `>>` operator multiple times. Consider a `task1` which upon successful completion starts a `task2` and `task3` in parallel. This dependency can be added to the DAG using:

```py
#task1: JIDSlurmOperator = JIDSlurmOperator(...)
#task2 ...

task1 >> task2
task1 >> task3
```

As each DAG is defined in pure Python, standard control structures (loops, if statements, etc.) can be used to create more complex workflow arrangements.
