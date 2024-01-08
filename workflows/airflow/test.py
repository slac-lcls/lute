"""Test Airflow DAG.

Runs all managed Tasks either in sequence or parallel.

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
description: str = "Run managed test Task in sequence and parallel"

dag: DAG = DAG(
    dag_id=dag_id,
    start_date=datetime(1970, 1, 1),
    schedule_interval=None,
    description=description,
)

tester: JIDSlurmOperator = JIDSlurmOperator(task_id="Tester", dag=dag)
binary_tester: JIDSlurmOperator = JIDSlurmOperator(task_id="BinaryTester", dag=dag)
socket_tester: JIDSlurmOperator = JIDSlurmOperator(task_id="SocketTester", dag=dag)

tester >> binary_tester
tester >> socket_tester
