# LUTE Configuration Database Specification
**Date:** 2024-02-12
**VERSION:** v0.1

## Basic Outline
- The backend database will be sqlite, using the standard Python library.
- A high-level API is provided, so if needed, the backend database can be changed without affecting `Executor` level code.
- One LUTE database is created per working directory for this iteration of the database. Note that this database is independent of any database used by a workflow manager (e.g. Airflow) to manage task execution order.
- Each database has the following tables:
  - 1 table for `Executor` configuration
  - 1 table for general task configuration (i.e., `lute.io.config.AnalysisHeader`)
  - 1 table **PER** `Task`
    - Executor and general configuration is shared between `Task` tables by pointing/linking to the entry ids in the above two tables.
    - Multiple experiments can reside in the same table, although in practice this is unlikely to occur in production as the working directory will most likely change between experiments.

### `gen_cfg` table
The general configuration table contains entries which may be shared between multiple `Task`s. The format of the table is:

| id | title                | experiment | run | date       | lute_version | task_timeout |
|:--:|:--------------------:|:----------:|:---:|:----------:|:------------:|:------------:|
| 2  | "My experiment desc" | "EXPx00000 | 1   | YYYY/MM/DD | 0.1          | 6000         |
|    |                      |            |     |            |              |              |

These parameters are extracted from the `TaskParameters` object. Each of those contains an `AnalysisHeader` object stored in the `lute_config` variable. For a given experimental run, this value will be shared across any `Task`s that are executed.

#### Column descriptions
| **Column**     | **Description**                                                                                         |
|:--------------:|:-------------------------------------------------------------------------------------------------------:|
| `id`           | ID of the entry in this table.                                                                          |
| `title`        | Arbitrary description/title of the purpose of analysis. E.g. what kind of experiment is being conducted |
| `experiment`   | LCLS Experiment. Can be a placeholder if debugging, etc.                                                |
| `run`          | LCLS Acquisition run. Can be a placeholder if debugging, testing, etc.                                  |
| `date`         | Date the configuration file was first setup.                                                            |
| `lute_version` | Version of the codebase being used to execute `Task`s.                                                  |
| `task_timeout` | The maximum amount of time in seconds that a `Task` can run before being cancelled.                     |
|                |                                                                                                         |

### `exec_cfg` table
The `Executor` table contains information on the environment provided to the `Executor` for `Task` execution, the polling interval used for IPC between the `Task` and `Executor` and information on the communicator protocols used for IPC. This information can be shared between `Task`s or between experimental runs, but not necessarily every `Task` of a given run will use exactly the same `Executor` configuration and environment.

| id | env                   | poll_interval | communicator_desc                           |
|:--:|:---------------------:|:-------------:|:-------------------------------------------:|
| 2  | "VAR1=val1;VAR2=val2" | 0.1           | "PipeCommunicator...;SocketCommunicator..." |
|    |                       |               |                                             |

#### Column descriptions
| **Column**          | **Description**                                                                                                                                                                   |
|:-------------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| `id`                | ID of the entry in this table.                                                                                                                                                    |
| `env`               | Execution environment used by the Executor and by proxy any Tasks submitted by an Executor matching this entry. Environment is stored as a string with variables delimited by ";" |
| `poll_interval`     | Polling interval used for Task monitoring.                                                                                                                                        |
| `communicator_desc` | Description of the Communicators used.                                                                                                                                            |
|                     |                                                                                                                                                                                   |

**NOTE**: The `env` column currently only stores variables related to `SLURM` or `LUTE` itself.

### `Task` tables
For every `Task` a table of the following format will be created. The exact number of columns will depend on the specific `Task`, as the number of parameters can vary between them, and each parameter gets its own column. Within a table, multiple experiments and runs can coexist. The experiment and run are not recorded directly. Instead the first two columns point to the id of entries in the general configuration and `Executor` tables respectively. The general configuration table entry will contain the experiment and run information.

| id | timestamp             | gen_cfg_id | exec_cfg_id | P1 | P2 | ... | Pn | task_status | summary   | payload | impl_schemas       | valid_flag |
|:--:|:---------------------:|:----------:|:-----------:|:--:|:--:|:---:|:--:|:-----------:|:---------:|:-------:|:------------------:|:----------:|
| 2  | "YYYY-MM-DD HH:MM:SS" | 1          | 1           | 1  | 2  | ... | 3  | "COMPLETED" | "Summary" | "XYZ"   | "schema1;schema3;" | 1          |
| 3  | "YYYY-MM-DD HH:MM:SS" | 1          | 1           | 3  | 1  | ... | 4  | "FAILED"    | "Summary" | "XYZ"   | "schema1;schema3;" | 0          |
|    |                       |            |             |    |    |     |    |             |           |         |                    |            |

#### Column descriptions
| **Column**          | **Description**                                                                                                                                  |
|:-------------------:|:------------------------------------------------------------------------------------------------------------------------------------------------:|
| `id`                | ID of the entry in this table.                                                                                                                   |
| `CURRENT_TIMESTAMP` | Full timestamp for the entry.                                                                                                                    |
| `gen_cfg_id`        | ID of the entry in the general config table that applies to this `Task` entry. That table has, e.g., experiment and run number.                  |
| `exec_cfg_id`       | The ID of the entry in the `Executor` table which applies to this `Task` entry.                                                                  |
| `P1` - `Pn`         | The specific parameters of the `Task`. The `P{1..n}` are replaced by the actual parameter names.                                                 |
| `task_status`       | Reported exit status of the `Task`. Note that the output may still be labeled invalid by the `valid_flag` (see below).                           |
| `summary`           | Short text summary of the `Task` result. This is provided by the `Task`, or sometimes the `Executor`.                                            |
| `payload`           | Full description of result from the `Task`. If the object is incompatible with the database, will instead be a pointer to where it can be found. |
| `impl_schemas`      | A string of semi-colon separated schema(s) implemented by the `Task`. Schemas describe conceptually the type output the `Task` produces.         |
| `valid_flag`        | A boolean flag for whether the result is valid. May be `0` (False) if e.g., data is missing, or corrupt, or reported status is failed.           |
|                     |                                                                                                                                                  |

**NOTE:** The `payload` is distinct from the output files. Payloads are an optional summary of the results provided by the `Task`. E.g. this may include graphical descriptions of results (plots, figures, etc.). The output files themselves, if they exist, will most likely be pointed to be an output file parameter in one of the columns `P{1...n}`

## API
This API is intended to be used at the `Executor` level, with some calls intended to provide default values for Pydantic models. Utilities for reading and inspecting the database outside of normal `Task` execution are addressed in the following subheader.

### Write
- `record_analysis_db(cfg: DescribedAnalysis) -> None`: Writes the configuration to the backend database.
- ...
- ...

### Read
- `read_latest_db_entry(db_dir: str, task_name: str, param: str) -> Any`: Retrieve the most recent entry from a database for a specific Task.
- ...
- ...

## Utilities
### Scripts
- `invalidate_entry`: Marks a database entry as invalid. Common reason to use this is if data has been deleted, or found to be corrupted.
- ...

### TUI and GUI
- `dbview`: TUI for database inspection. Read only.
- ...
