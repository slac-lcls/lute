"""Run smalldata_tools and basic analysis.

Runs smalldata_tools and then basic analysis for XSS, XAS and XES.

Note:
    The task_id MUST match the managed task name when defining DAGs - it is used
    by the operator to properly launch it.

    dag_id names must be unique, and they are not namespaced via folder
    hierarchy. I.e. all DAGs on an Airflow instance must have unique ids. The
    Airflow instance used by LUTE is currently shared by other software - DAG
    IDs should always be prefixed with `lute_`. LUTE scripts should append this
    internally, so a DAG "lute_test" can be triggered by asking for "test"
"""

import os
from datetime import datetime
from typing import Any, Dict

from airflow import DAG
from airflow.decorators import task

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

@task.branch(task_id="Psana1v2Brancher")
def psana1v2_branch_func(**context) -> str:
    if "dag_run" in context:
        conf: Dict[str, Any] = context["dag_run"].conf
        if "is_daq2" in conf:
            if conf["is_daq2"]:
                return "SmallDataProducer2"
            elif conf["is_daq2"] is None:
                ... # Not sure what to do if we couldn't determine?
    return "SmallDataProducer"

smd_producer: JIDSlurmOperator = JIDSlurmOperator(task_id="SmallDataProducer", dag=dag)
smd2_producer: JIDSlurmOperator = JIDSlurmOperator(task_id="SmallDataProducer2", dag=dag)

psana_brancher = psana1v2_branch_func()

# Update trigger rules since there are now branches
xss: JIDSlurmOperator = JIDSlurmOperator(
    max_cores=2, task_id="SmallDataXSSAnalyzer", dag=dag, trigger_rule="none_failed"
)

xas: JIDSlurmOperator = JIDSlurmOperator(
    max_cores=2, task_id="SmallDataXASAnalyzer", dag=dag, trigger_rule="none_failed"
)

xes: JIDSlurmOperator = JIDSlurmOperator(
    task_id="SmallDataXESAnalyzer", dag=dag, trigger_rule="none_failed"
)

psana_brancher >> [smd_producer, smd2_producer]

# Run summaries
smd_producer >> xss
smd_producer >> xas
smd_producer >> xes

smd2_producer >> xss
smd2_producer >> xas
smd2_producer >> xes
