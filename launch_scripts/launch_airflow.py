#!/sdf/group/lcls/ds/ana/sw/conda1/inst/envs/ana-4.0.62-py3/bin/python

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
from requests.exceptions import HTTPError

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


def _retrieve_pw(instance: str = "prod", is_admin: bool = False) -> str:
    user_type: str
    if is_admin:
        logger.debug("Running as operator.")
        user_type = "admin"
    else:
        logger.debug("Running as user.")
        user_type = "user"
    path: str = "/sdf/group/lcls/ds/tools/lute/airflow_{instance}_{user_type}.txt"
    if instance == "prod" or instance == "test":
        path = path.format(instance=instance, user_type=user_type)
    else:
        raise ValueError('`instance` must be either "test" or "prod"!')
    with open(path, "r") as f:
        pw: str = f.readline().strip()
    return pw


def _request_arp_token(exp: str, lifetime: int = 300) -> str:
    """Request an ARP token via Kerberos endpoint.

    A token is required for job submission.

    Args:
        exp (str): The experiment to request the token for. All tokens are
            scoped to a single experiment.

        lifetime (int): The lifetime, in minutes, of the token. After the token
            expires, it can no longer be used for job submission. The maximum
            time you can request is 480 minutes (i.e. 8 hours). NOTE: since this
            token is used for the entirety of a workflow, it must have a lifetime
            equal or longer than the duration of the workflow's execution time.
    """
    from kerberos import GSSError
    from krtc import KerberosTicket

    try:
        krbheaders: Dict[str, str] = KerberosTicket(
            "HTTP@pswww.slac.stanford.edu"
        ).getAuthHeaders()
    except GSSError:
        logger.info(
            "Cannot proceed without credentials. Try running `kinit` from the command-line."
        )
        raise
    base_url: str = "https://pswww.slac.stanford.edu/ws-kerb/lgbk/lgbk"
    token_endpoint: str = (
        f"{base_url}/{exp}/ws/generate_arp_token?token_lifetime={lifetime}"
    )
    resp: requests.models.Response = requests.get(token_endpoint, headers=krbheaders)
    resp.raise_for_status()
    token: str = resp.json()["value"]
    formatted_token: str = f"Bearer {token}"
    return formatted_token


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="trigger_airflow_lute_dag",
        description="Trigger Airflow to begin executing a LUTE DAG.",
        epilog="Refer to https://github.com/slac-lcls/lute for more information.",
    )
    parser.add_argument(
        "-a", "--admin", help="Run as admin. Requires permissions.", action="store_true"
    )
    parser.add_argument("-c", "--config", type=str, help="Path to config YAML file.")
    parser.add_argument("-d", "--debug", help="Run in debug mode.", action="store_true")
    parser.add_argument(
        "--test", help="Use test Airflow instance.", action="store_true"
    )
    parser.add_argument(
        "-w", "--workflow", type=str, help="Workflow to run.", default="test"
    )
    parser.add_argument(
        "-W",
        "--workflow_defn",
        type=str,
        help="Path to a YAML file with workflow.",
        default="",
    )
    # Optional arguments for when running from command-line
    parser.add_argument(
        "-e",
        "--experiment",
        type=str,
        help="Provide an experiment if not running with ARP.",
        required=False,
    )
    parser.add_argument(
        "-r",
        "--run",
        type=str,
        help="Provide a run number if not running with ARP.",
        required=False,
    )

    args: argparse.Namespace
    extra_args: List[str]  # Should contain all SLURM arguments!
    args, extra_args = parser.parse_known_args()
    # Check if was submitted from ARP - look for token
    use_kerberos: bool = (
        True  # Always copy kerberos ticket so non-active experiments can work.
    )
    cache_file: Optional[str] = os.getenv("KRB5CCNAME")
    if os.getenv("Authorization") is None:
        if cache_file is None:
            logger.info("No Kerberos cache. Try running `kinit` and resubmitting.")
            sys.exit(-1)

        if args.experiment is None or args.run is None:
            logger.info(
                (
                    "You must provide a `-e ${EXPERIMENT}` and `-r ${RUN_NUM}` "
                    "if not running with the ARP!\n"
                    "If you submitted this from the eLog and are seeing this error "
                    "please contact the maintainers."
                )
            )
            sys.exit(-1)
        os.environ["EXPERIMENT"] = args.experiment
        os.environ["RUN_NUM"] = args.run

        os.environ["Authorization"] = _request_arp_token(args.experiment)
        os.environ["ARP_JOB_ID"] = str(uuid.uuid4())

    wf_name: str
    use_custom_defn: bool
    if args.workflow_defn:
        wf_name = "test_dynamic"
        use_custom_defn = True
        logger.info("Will attempt running custom DAG")
    else:
        wf_name = args.workflow
        use_custom_defn = False

    airflow_instance: str
    instance_str: str
    if args.test:
        airflow_instance = "http://172.24.5.190:8080"
        instance_str = "test"
    else:
        airflow_instance = "http://172.24.5.247:8080"
        instance_str = "prod"

    airflow_api_endpoints: Dict[str, str] = {
        "health": "api/v1/health",
        "run_dag": f"api/v1/dags/lute_{wf_name}/dagRuns",
        "get_tasks": f"api/v1/dags/lute_{wf_name}/tasks",
        "get_xcom": (  # Need to format dag_run_id, task_id, xcom_key
            f"api/v1/dags/lute_{wf_name}/dagRuns/{{dag_run_id}}/taskInstances"
            f"/{{task_id}}/xcomEntries/{{xcom_key}}"
        ),
        # Only for User-Specified workflows
        "mod_dag": f"api/v1/dags/lute_{wf_name}",  # Delete, pause/unpause, etc.
        "create_defn": "api/v1/variables",
        "update_defn": "api/v1/variables/user_workflow",
        "parse_file": "api/v1/parseDagFile/{file_token}",
    }

    pw: str = _retrieve_pw(instance_str, is_admin=args.admin)
    user_name: str = "btx" if args.admin else "lcls_user"
    auth: HTTPBasicAuth = HTTPBasicAuth(user_name, pw)
    resp: requests.models.Response = requests.get(
        f"{airflow_instance}/{airflow_api_endpoints['health']}",
        auth=auth,
    )
    resp.raise_for_status()

    params: Dict[str, Union[str, int, List[str]]] = {
        "config_file": args.config,
        "debug": args.debug,
    }

    wf_defn: Dict[str, Any] = {}
    if use_custom_defn:
        import yaml
        import json

        if not os.path.exists(args.workflow_defn):
            logger.error("Workflow definition path does not exist! Exiting!")
            sys.exit(-1)
        with open(args.workflow_defn, "r") as f:
            wf_defn = yaml.load(f, yaml.FullLoader)

        # Update user workflow definition in Airflow
        new_workflow: Dict[str, str] = {
            "key": "user_workflow",
            "value": json.dumps(wf_defn),
        }
        resp = requests.patch(
            f"{airflow_instance}/{airflow_api_endpoints['update_defn']}",
            json=new_workflow,
            auth=auth,
        )
        try:
            resp.raise_for_status()
        except requests.exceptions.HTTPError as err:
            if err.response.status_code == 404:
                # Workflow definition not found so previous DAG completed properly
                resp = requests.post(
                    f"{airflow_instance}/{airflow_api_endpoints['create_defn']}",
                    json=new_workflow,
                    auth=auth,
                )
                resp.raise_for_status()
            else:
                raise
        logger.debug("Sent new workflow definition.")
        resp = requests.get(
            f"{airflow_instance}/{airflow_api_endpoints['mod_dag']}",
            auth=auth,
        )
        resp.raise_for_status()
        file_token: str = resp.json()["file_token"]
        f_endpoint: str = airflow_api_endpoints["parse_file"].format(
            file_token=file_token
        )
        resp = requests.put(f"{airflow_instance}/{f_endpoint}", auth=auth)
        resp.raise_for_status()
        logger.debug("Re-parsed DAG for setup with new workflow.")

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
            "workflow": wf_defn,  # Only used for custom defined workflows.
        },
    }

    resp = requests.post(
        f"{airflow_instance}/{airflow_api_endpoints['run_dag']}",
        json=dag_run_data,
        auth=auth,
    )
    resp.raise_for_status()
    dag_run_id: str = dag_run_data["dag_run_id"]
    logger.info(f"Submitted DAG (Workflow): {wf_name}\nDAG_RUN_ID: {dag_run_id}")
    dag_state: str = resp.json()["state"]
    logger.info(f"DAG is {dag_state}")

    # Get Task information
    task_ids: List[str]
    if not use_custom_defn:
        resp = requests.get(
            f"{airflow_instance}/{airflow_api_endpoints['get_tasks']}",
            auth=auth,
        )
        resp.raise_for_status()
        task_ids = [task["task_id"] for task in resp.json()["tasks"]]
    else:
        # Airflow shouldn't have list of Tasks yet so we parse manually
        task_ids = ["retrieve_workflow"]

        def get_names(wf_defn: Dict[str, Any], names: List[str]) -> None:
            names.append(f"user_workflow.{wf_defn['task_name']}")
            for wf_new in wf_defn["next"]:
                get_names(wf_new, names)

        get_names(wf_defn, task_ids)
        task_ids = sorted(task_ids)
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
            if task_id not in completed_tasks and task_state not in (None, "scheduled"):
                if task_id not in logged_running:
                    # Should be "running" by first time it reaches here.
                    # Or e.g. "upstream_failed"... Setup to skip "scheduled"
                    logger.info(f"{task_id} state: {task_state}")
                    logged_running.append(task_id)

                if task_state in ("success", "failed"):
                    # Only pushed to XCOM at the end of each Task
                    xcom_url: str = (
                        f"{airflow_instance}/{airflow_api_endpoints['get_xcom']}"
                    )
                    xcom_url = xcom_url.format(
                        dag_run_id=dag_run_id,
                        task_id=task_id,
                        xcom_key=xcom_key,
                    )
                    try:
                        resp = requests.get(xcom_url, auth=auth)
                        resp.raise_for_status()
                        logs: str = resp.json()["value"]  # Only want to print once.
                        logger.info(f"Providing logs for {task_id}")
                        print("-" * 50, flush=True)
                        print(logs, flush=True)
                        print("-" * 50, flush=True)
                    except HTTPError:
                        # retrieve_workflow has no logs...
                        logger.info(f"No logs for {task_id}.")
                    logger.info(f"End of logs for {task_id}")
                    completed_tasks[task_id] = task_state

                elif task_state in ("upstream_failed"):
                    # upstream_failed never launches so has no log
                    completed_tasks[task_id] = task_state

        if dag_state in ("queued", "running"):
            continue
        logger.info(f"DAG exited: {dag_state}")
        break

    if use_kerberos:
        # We had to do some funny business to get Kerberos credentials...
        # Cleanup now that we're done
        logger.debug("Removing duplicate Kerberos credentials.")
        # This should be defined if we get here
        # Format is FILE:/.../...
        if cache_file is not None:
            os.remove(cache_file[5:])
            os.rmdir(f"{os.path.expanduser('~')}/.tmp_cache")

    if dag_state == "failed":
        sys.exit(1)
    else:
        sys.exit(0)
