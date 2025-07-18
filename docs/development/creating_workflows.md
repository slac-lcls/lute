# Creating Workflows
The core LUTE library was designed to be agnostic to the workflow management solution. Currently there are three backends implemented, with one being the recommended/primary for production usage.

1. `Airflow`: This is the recommended workflow orchestrator for production use. There are two instances running, one for testing, and the production instance. Workflows (generally) need to be explicitly defined beforehand, and then synced to the Airflow instance, however, which can make the development cycle slightly more difficult.
2. `Prefect`: `Prefect` is a newer platform than Airflow, and an experimental instance has been setup. It offers some nice features for the LUTE use-case, such as the ability to dynamically define workflows fitting more naturally into the prefect model. Currently, however, there are fewer resources allocated to the prefect instance, and it is generally not as stable.
3. `SLURM`-job: As a backup, when all else fails, a special script has been developed that can perform workflow orchestration from within a batch job on S3DF. This is intended as backup option only.

For experiment analysis during the beamtime, use the `Airflow` backend for the time being. In the future, the recommendation may be switched to `prefect`.
