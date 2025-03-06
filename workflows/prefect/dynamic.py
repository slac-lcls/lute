from typing import Any, Dict, List

from prefect import flow
from prefect.futures import wait, PrefectFuture
from prefect.task_runners import ThreadPoolTaskRunner

from flow_dataclasses import FlowConf
from tasks.jidtasks import run_managed_task

flow_name: str = f"lute_dynamic"


def create_workflow(
    wf_dict: Dict[str, Any],
    flow_conf: FlowConf,
    wait_for: List[PrefectFuture] = [],
    all_futures: List[PrefectFuture] = [],
) -> None:
    slurm_params: str = wf_dict.get("slurm_params", "")
    future: PrefectFuture = run_managed_task.with_options(
        name=wf_dict["task_name"]
    ).submit(
        lute_task_id=wf_dict["task_name"],
        conf=flow_conf,
        custom_slurm_params=slurm_params,
        wait_for=wait_for,
    )
    all_futures.append(future)
    if wf_dict["next"] == []:
        return
    else:
        for task in wf_dict["next"]:
            create_workflow(task, flow_conf, [future], all_futures)
    return


@flow(name=flow_name, task_runner=ThreadPoolTaskRunner(max_workers=8), log_prints=True)
def dynamic_flow(flow_conf: FlowConf) -> None:

    wf_dict: Dict[str, str] = flow_conf.get("workflow")

    wait_for: List[PrefectFuture] = []
    all_futures: List[PrefectFuture] = []
    create_workflow(wf_dict, flow_conf, wait_for, all_futures)
    wait(all_futures)


if __name__ == "__main__":
    ...
