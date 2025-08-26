"""Run geometry optimization for centering detector on the beam.

Performs a Bayesian Optimization coupled with pyFAI least squares fitting of distance, beam center and tilt angles.

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
description: str = "DAG to test branching based on run_type."

dag: DAG = DAG(
    dag_id=dag_id,
    start_date=datetime(1970, 1, 1),
    schedule_interval=None,
    description=description,
    is_paused_upon_creation=False,
)

@task.branch(task_id="Psana1v2Brancher")
def psana1v2_branch_func(**context) -> str:
    if "dag_run" in context:
        conf: Dict[str, Any] = context["dag_run"].conf
        if "experiment" in conf and conf["experiment"][:3].lower() == "mfx":
            return "SmallDataProducer2"
    return "SmallDataProducer"

psana1v2_brancher = psana1v2_branch_func()

smd_producer: JIDSlurmOperator = JIDSlurmOperator(task_id="SmallDataProducer", dag=dag)

smd_producer2: JIDSlurmOperator = JIDSlurmOperator(task_id="SmallDataProducer2", dag=dag)

geom_optimizer: JIDSlurmOperator = JIDSlurmOperator(max_cores=120, task_id="PyFAIGeometryOptimizer", dag=dag)

geom_optimizer2: JIDSlurmOperator = JIDSlurmOperator(max_cores=120, task_id="PyFAIGeometryOptimizer2", dag=dag)

# Branch Workflow depending on available psana version
psana1v2_brancher >> [smd_producer, smd_producer2]

smd_producer >> geom_optimizer

smd_producer2 >> geom_optimizer2