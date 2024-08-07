"""Test Airflow Dynamic DAG.

Generates a DAG based no configuration passed via the Airflow context.
This workflow expects that there is a "workflow" key passed along within
the Airflow context. The first task in the workflow unpacks this and stores
it in a variable so a subsequent task group can unpack the individual steps
of the true user DAG.

See the launch_airflow.py script in lute/launch_scripts for the workflow
definition and how it is parsed and passed along.
"""

from datetime import datetime
import os
import uuid
from typing import Optional, Any, Dict, List

from airflow.decorators import dag, task, task_group
from airflow.models import Variable

from lute.operators.jidoperators import JIDSlurmOperator


dag_id: str = f"lute_{os.path.splitext(os.path.basename(__file__))[0]}"


def create_links(
    wf_dict: Dict[str, Any],
    op: Optional[JIDSlurmOperator] = None,
    task_list: List[JIDSlurmOperator] = [],
) -> JIDSlurmOperator:
    print(wf_dict)
    # return JIDSlurmOperator(task_id="BinaryTester", max_cores=5)
    new_op: JIDSlurmOperator = JIDSlurmOperator(
        task_id=wf_dict["task_name"], max_cores=2
    )
    task_list.append(new_op)
    if wf_dict["next"] == []:
        return new_op
    else:
        child_tasks: List[JIDSlurmOperator] = []
        for task in wf_dict["next"]:
            child_tasks.append(create_links(task, new_op, task_list))
        # new_op.set_downstream(child_tasks)
        new_op >> child_tasks
        return new_op


@dag(dag_id=dag_id, start_date=datetime(1970, 1, 1), schedule_interval=None)
def test_dynamic():
    @task
    def retrieve_workflow(**context):
        if "dag_run" in context:
            print(context["dag_run"].conf["workflow"])
            wf: Dict[str, Any] = context["dag_run"].conf["workflow"]
            Variable.set(key="user_workflow", value=wf, serialize_json=True)
            return wf
        return None

    @task_group(group_id="user_workflow")
    def user_workflow():
        wf_dict: Optional[Dict[str, Any]] = Variable.get(
            "user_workflow", default_var=None, deserialize_json=True
        )
        if wf_dict is not None:
            print(wf_dict)
            task_list: List[JIDSlurmOperator] = []
            first_task: JIDSlurmOperator = create_links(wf_dict, task_list=task_list)
        else:
            tester: JIDSlurmOperator = JIDSlurmOperator(max_cores=2, task_id="Tester")

    retrieve_workflow() >> user_workflow()


test_dynamic()
