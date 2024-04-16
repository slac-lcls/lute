#!/sdf/group/lcls/ds/ana/sw/conda1/inst/envs/ana-4.0.60-py3/bin/python

"""Script submitted by Automated Run Processor (ARP) to trigger an Airflow DAG.

This script is submitted by the ARP to the batch nodes. It triggers Airflow to
begin running the tasks of the specified directed acyclic graph (DAG).
"""

__author__ = "Gabriel Dorlhiac"

import sys
import os
import uuid
import getpass
import datetime
import logging
import argparse
import time
from typing import Dict, Union, List, Optional, Any

import requests
from requests.auth import HTTPBasicAuth

# Requests, urllib have lots of debug statements. Only set level for this logger
logger: logging.Logger = logging.getLogger("Launch_Airflow")
handler: logging.Handler = logging.StreamHandler()
formatter: logging.Formatter = logging.Formatter(logging.BASIC_FORMAT)
handler.setFormatter(formatter)
logger.addHandler(handler)

if __debug__:
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.INFO)


def _retrieve_pw(instance: str = "prod") -> str:
    path: str = "/sdf/group/lcls/ds/tools/lute/airflow_{instance}.txt"
    if instance == "prod" or instance == "test":
        path = path.format(instance=instance)
    else:
        raise ValueError('`instance` must be either "test" or "prod"!')

    with open(path, "r") as f:
        pw: str = f.readline().strip()
    return pw


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="trigger_airflow_lute_dag",
        description="Trigger Airflow to begin executing a LUTE DAG.",
        epilog="Refer to https://github.com/slac-lcls/lute for more information.",
    )
    parser.add_argument("-c", "--config", type=str, help="Path to config YAML file.")
    parser.add_argument("-d", "--debug", help="Run in debug mode.", action="store_true")
    parser.add_argument(
        "--test", help="Use test Airflow instance.", action="store_true"
    )
    parser.add_argument(
        "-w", "--workflow", type=str, help="Workflow to run.", default="test"
    )

    args: argparse.Namespace
    extra_args: List[str]  # Should contain all SLURM arguments!
    args, extra_args = parser.parse_known_args()
    airflow_instance: str
    instance_str: str
    if args.test:
        airflow_instance = "http://172.24.5.190:8080/"
        instance_str = "test"
    else:
        airflow_instance = "http://172.24.5.247:8080/"
        instance_str = "prod"

    airflow_api_endpoints: Dict[str, str] = {
        "health": "api/v1/health",
        "run_dag": f"api/v1/dags/lute_{args.workflow}/dagRuns",
        "get_tasks": f"api/v1/dags/lute_{args.workflow}/tasks",
        "get_xcom": (  # Need to format dag_run_id, task_id, xcom_key
            f"api/v1/dags/lute_{args.workflow}/dagRuns/{{dag_run_id}}/taskInstances"
            f"/{{task_id}}/xcomEntries/{{xcom_key}}"
        ),
    }

    pw: str = _retrieve_pw(instance_str)
    auth: HTTPBasicAuth = HTTPBasicAuth("btx", pw)
    resp: requests.models.Response = requests.get(
        f"{airflow_instance}/{airflow_api_endpoints['health']}",
        auth=auth,
    )
    resp.raise_for_status()

    params: Dict[str, Union[str, int, List[str]]] = {
        "config_file": args.config,
        "debug": args.debug,
    }

    # Experiment, run #, and ARP env variables come from ARP submission only
    dag_run_data: Dict[str, Union[str, Dict[str, Union[str, int, List[str]]]]] = {
        "dag_run_id": str(uuid.uuid4()),
        "conf": {
            "experiment": os.environ.get("EXPERIMENT"),
            "run_id": f"{os.environ.get('RUN_NUM')}_{datetime.datetime.utcnow().isoformat()}",
            "JID_UPDATE_COUNTERS": os.environ.get("JID_UPDATE_COUNTERS"),
            "ARP_ROOT_JOB_ID": os.environ.get("ARP_JOB_ID"),
            "ARP_LOCATION": os.environ.get("ARP_LOCATION", "S3DF"),
            "Authorization": os.environ.get("Authorization"),
            "user": getpass.getuser(),
            "lute_location": os.path.abspath(f"{os.path.dirname(__file__)}/.."),
            "lute_params": params,
            "slurm_params": extra_args,
        },
    }

    resp = requests.post(
        f"{airflow_instance}/{airflow_api_endpoints['run_dag']}",
        json=dag_run_data,
        auth=auth,
    )
    resp.raise_for_status()
    dag_run_id: str = dag_run_data["dag_run_id"]
    logger.info(f"Submitted DAG (Workflow): {args.workflow}\nDAG_RUN_ID: {dag_run_id}")
    dag_state: str = resp.json()["state"]
    logger.info(f"DAG is {dag_state}")

    # Get Task information
    resp = requests.get(
        f"{airflow_instance}/{airflow_api_endpoints['get_tasks']}",
        auth=auth,
    )
    resp.raise_for_status()
    task_ids: List[str] = [task["task_id"] for task in resp.json()["tasks"]]
    task_id_str: str = ",\n\t- ".join(tid for tid in task_ids)
    logger.info(
        f"Contains Managed Tasks (alphabetical, not execution order):\n\t- {task_id_str}"
    )

    # Enter loop for checking status
    time.sleep(1)
    # Same as run_dag endpoint, but needs to include the dag_run_id on the end
    url: str = f"{airflow_instance}/{airflow_api_endpoints['run_dag']}/{dag_run_id}"
    # Pulling logs for each Task via XCom
    xcom_key: str = "log"
    completed_tasks: Dict[str, str] = {}  # Remember exit status of each Task
    logged_running: List[str] = []  # Keep track to only print "running" once
    while True:
        time.sleep(1)
        # DAG Status
        resp = requests.get(url, auth=auth)
        resp.raise_for_status()
        dag_state = resp.json()["state"]
        # Check Task instances
        task_url: str = f"{url}/taskInstances"
        resp = requests.get(task_url, auth=auth)
        resp.raise_for_status()
        instance_information: Dict[str, Any] = resp.json()["task_instances"]
        for inst in instance_information:
            task_id: str = inst["task_id"]
            task_state: Optional[str] = inst["state"]
            if task_id not in completed_tasks and task_state:
                if task_id not in logged_running:
                    # Should be "running" by first time it reaches here.
                    # Or e.g. "upstream_failed"...
                    logger.info(f"{task_id} state: {task_state}")
                    logged_running.append(task_id)

                if task_state in ("success", "failed", "upstream_failed"):
                    # Only pushed to XCOM at the end of each Task
                    xcom_url: str = (
                        f"{airflow_instance}/{airflow_api_endpoints['get_xcom']}"
                    )
                    xcom_url = xcom_url.format(
                        dag_run_id=dag_run_id,
                        task_id=task_id,
                        xcom_key=xcom_key,
                    )
                    resp = requests.get(xcom_url, auth=auth)
                    resp.raise_for_status()
                    logs: str = resp.json()["value"]  # Only want to print once.
                    logger.info(f"Providing logs for {task_id}")
                    print("-" * 50, flush=True)
                    print(logs, flush=True)
                    print("-" * 50, flush=True)
                    logger.info(f"End of logs for {task_id}")
                    completed_tasks[task_id] = task_state

        if dag_state in ("queued", "running"):
            continue
        logger.info(f"DAG exited: {dag_state}")
        break

    if dag_state == "failed":
        sys.exit(1)
    else:
        sys.exit(0)
