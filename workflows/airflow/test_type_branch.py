"""DAG for testing branching based on provision of a `run_type.`

If a specific `run_type` is passed along in the dag run conf, it will change
the branch it takes.

This tests functionality that allows workflows to branch based on run_types
that can be provided when asking the DAQ to record a run.
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


@task.branch(task_id="BranchTester")
def test_branch_func(**context) -> str:
    if "dag_run" in context:
        conf: Dict[str, Any] = context["dag_run"].conf
        if "run_type" in conf and conf["run_type"] == "TEST_ERROR":
            return "BinaryErrTester"
    return "BinaryTester"


branch_tester = test_branch_func()
tester: JIDSlurmOperator = JIDSlurmOperator(max_cores=2, task_id="Tester", dag=dag)
binary_tester: JIDSlurmOperator = JIDSlurmOperator(
    max_cores=5, task_id="BinaryTester", dag=dag
)
binary_err_tester: JIDSlurmOperator = JIDSlurmOperator(
    max_cores=5, task_id="BinaryErrTester", dag=dag
)
socket_tester: JIDSlurmOperator = JIDSlurmOperator(
    max_cores=2, task_id="SocketTester", dag=dag, trigger_rule="none_failed"
)
write_tester: JIDSlurmOperator = JIDSlurmOperator(
    max_cores=2, task_id="WriteTester", dag=dag, trigger_rule="none_failed"
)
read_tester: JIDSlurmOperator = JIDSlurmOperator(
    max_cores=2, task_id="ReadTester", dag=dag, trigger_rule="none_failed"
)

# If we get binary_err_tester rest of workflow won't run
tester >> branch_tester >> [binary_tester, binary_err_tester] >> socket_tester >> write_tester >> read_tester
