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
import time
import sys
from typing import Optional, Any, Dict, List

from airflow import configuration
from airflow.decorators import dag, task, task_group
from airflow.utils.trigger_rule import TriggerRule
from airflow.models import Variable, DagBag, TaskInstance
from airflow.models.taskmixin import DAGNode

from lute.operators.jidoperators import JIDSlurmOperator


dag_id: str = f"lute_{os.path.splitext(os.path.basename(__file__))[0]}"


def create_links(
    wf_dict: Dict[str, Any],
    op: Optional[JIDSlurmOperator] = None,
    task_list: List[JIDSlurmOperator] = [],
) -> JIDSlurmOperator:
    slurm_params: str = wf_dict.get("slurm_params", "")
    new_op: JIDSlurmOperator = JIDSlurmOperator(
        task_id=wf_dict["task_name"], custom_slurm_params=slurm_params
    )
    task_list.append(new_op)
    if wf_dict["next"] == []:
        return new_op
    else:
        child_tasks: List[JIDSlurmOperator] = []
        for task in wf_dict["next"]:
            child_tasks.append(create_links(task, new_op, task_list))
        new_op >> child_tasks
        return new_op


@dag(dag_id=dag_id, start_date=datetime(1970, 1, 1), schedule_interval=None)
def test_dynamic():
    @task
    def retrieve_workflow(**context):
        if "dag_run" in context:
            wf: Dict[str, Any] = context["dag_run"].conf["workflow"]
            Variable.set(key="user_workflow", value=wf, serialize_json=True)
            time.sleep(3)  # Make sure var gets set
            return wf
        return None

    @task_group(group_id="user_workflow")
    def user_workflow():
        wf_dict: Optional[Dict[str, Any]] = Variable.get(
            "user_workflow", default_var=None, deserialize_json=True
        )
        if wf_dict is not None:
            task_list: List[JIDSlurmOperator] = []
            _: JIDSlurmOperator = create_links(wf_dict, task_list=task_list)

    @task(trigger_rule=TriggerRule.ALL_DONE)
    def delete_workflow(**context):
        folder: Any = configuration.get("core", "DAGS_FOLDER")
        dagbag: DagBag = DagBag(folder)
        dag_ref: Any = dagbag.dags[dag_id]

        tg: Any = dag_ref.task_group.get_child_by_label("user_workflow")
        execution_date: str = context.get("logical_date")

        # Collect the TaskGroup state now by looking at state of Tasks in the group
        # Otherwise this information gets hidden when we delete the user_workflow
        # This Task (`delete_workflow`) always succeeds, so if we don't collect
        # the information the workflow is marked as successful even when it fails
        task: "DAGNode"
        ti: TaskInstance
        user_wf_state: str = "success"
        for task in tg.children.values():
            try:
                ti = TaskInstance(task, execution_date)
                if ti.current_state() != "success":
                    print(
                        f"{task.task_id} was marked {ti.current_state()}. "
                        "Marking task group the same."
                    )
                    user_wf_state = ti.current_state()
            except Exception as err:
                print(err)
        Variable.delete(key="user_workflow")
        if user_wf_state != "success":
            print(f"User workflow does not report success: {user_wf_state}")
            sys.exit(-1)

    retrieve_workflow() >> user_workflow() >> delete_workflow()


test_dynamic()
