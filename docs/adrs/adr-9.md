# [ADR-9] Airflow launch script will run as long lived batch job.

**Date:** 2024-04-15

## Status
**Proposed**

## Context and Problem Statement
- Each `Task` will produce its own log file.
- Log files from jobs (i.e. DAGs/workflows) run by different users will be in different locations/directories.
- None of these log files will be accessible from the Web UI of the eLog unless they are available to the initial launch script which starts the workflow.

## Decision
The Airflow launch script will be a long lived process, running for the duration of the entire DAG. It will provide basic status logging information, e.g. what `Task`s are running, if they succeed or failed. Additionally, at the end of each `Task` job, the launch job will collect the log file from that job and append it to its own log.

As the Airflow launch script is an entry point used from the eLog, only its log file is available to users using that UI. By converting the launch script into a long-lived monitoring job it allows the log information to be easily accessible.

In order to accomplish this, the launch script must be submitted as a batch job, in order to comply with the 30 second timeout imposed by jobs run by the ARP. This necessitates providing an additional wrapper script.

### Decision Drivers
* Log availability from the eLog.
* All logs available from a single location.

### Considered Options
* All jobs append to the same initial file, by specifying a log file. (`--open-mode=append` for SLURM)
  * Having a monitoring job provides the opportunity to include additional information.

## Consequences
* There needs to be an additional wrapper script: `submit_launch_airflow.sh` which submits the `launch_airflow.py` script (run by `lute_launcher`) as a batch job.
  * Jobs run by the ARP can not be long-lived - there is a 30 second timeout.
  * The ARP was intended to submit batch jobs - it captures the log file from batch jobs, so running the job directly or submitting as a batch job is equivalent in terms of presenting information to the eLog UI.
* Another core is used to run the job. Overhead is now two cores - 1 for the monitoring job (`launch_airflow.py`) and 1 for the `Executor` process. 

## Compliance


## Metadata
- This ADR WILL be revisited during the post-mortem of the first prototype.
- Compliance section will be updated as prototype evolves.
