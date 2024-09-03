"""SFX Processing DAG (W/ Experimental Phasing)

Runs SFX processing, beginning with the PyAlgos peak finding algorithm.
This workflow is used for data requiring experimental phasing. Do NOT use this
DAG if you are using molecular replacement.

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
description: str = (
    "Produce basic analysis for XSS, XAS, and XES from SmallData hdf5 files."
)

dag: DAG = DAG(
    dag_id=dag_id,
    start_date=datetime(2024, 9, 3),
    schedule_interval=None,
    description=description,
)

xss: JIDSlurmOperator = JIDSlurmOperator(
    max_cores=2, task_id="SmallDataXSSAnalyzer", dag=dag
)

xas: JIDSlurmOperator = JIDSlurmOperator(
    max_cores=2, task_id="SmallDataXASAnalyzer", dag=dag
)

xes: JIDSlurmOperator = JIDSlurmOperator(
    max_cores=4, task_id="SmallDataXESAnalyzer", dag=dag
)

# Run summaries
xss
xas
xes
