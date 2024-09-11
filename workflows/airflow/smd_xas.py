"""Run basic XAS analysis based on smalldata output.

Runs only XAS analysis based on smalldata output. Assumes smalldata_tools
has been run separately.

Note:
    The task_id MUST match the managed task name when defining DAGs - it is used
    by the operator to properly launch it.

    dag_id names must be unique, and they are not namespaced via folder
    hierarchy. I.e. all DAGs on an Airflow instance must have unique ids. The
    Airflow instance used by LUTE is currently shared by other software - DAG
    IDs should always be prefixed with `lute_`. LUTE scripts should append this
    internally, so a DAG "lute_test" can be triggered by asking for "test"
"""

from datetime import datetime
import os
from airflow import DAG
from lute.operators.jidoperators import JIDSlurmOperator

dag_id: str = f"lute_{os.path.splitext(os.path.basename(__file__))[0]}"
description: str = "Produce basic analysis for XAS from SmallData hdf5 files."

dag: DAG = DAG(
    dag_id=dag_id,
    start_date=datetime(2024, 9, 3),
    schedule_interval=None,
    description=description,
)

xas: JIDSlurmOperator = JIDSlurmOperator(
    max_cores=2, task_id="SmallDataXASAnalyzer", dag=dag
)

# Run summary
xas
