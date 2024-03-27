#!/sdf/group/lcls/ds/ana/sw/conda1/inst/envs/ana-4.0.59-py3/bin/python

"""Script submitted by Automated Run Processor (ARP) to trigger an Airflow DAG.

This script is submitted by the ARP to the batch nodes. It triggers Airflow to
begin running the tasks of the specified directed acyclic graph (DAG).
"""

__author__ = "Gabriel Dorlhiac"

import os
import uuid
import getpass
import datetime
import logging
import argparse
from typing import Dict, Union, List

import requests
from requests.auth import HTTPBasicAuth

if __debug__:
    logging.basicConfig(level=logging.DEBUG)
else:
    logging.basicConfig(level=logging.INFO)

logger: logging.Logger = logging.getLogger(__name__)


def _retrieve_pw(instance: str = "prod") -> str:
    path: str = "/sdf/group/lcls/ds/tools/lute/airflow_{instance}.txt"
    if instance == "prod" or instance == "test":
        path = path.format(instance)
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
    }

    resp: requests.models.Response = requests.get(
        f"{airflow_instance}/{airflow_api_endpoints['health']}",
        auth=HTTPBasicAuth("btx", _retrieve_pw(instance_str)),
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
            "run_id": f"{os.environ.get('RUN_NUM')}{datetime.datetime.utcnow().isoformat()}",
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

    resp: requests.models.Response = requests.post(
        f"{airflow_instance}/{airflow_api_endpoints['run_dag']}",
        json=dag_run_data,
        auth=HTTPBasicAuth("btx", _retrieve_pw(instance_str)),
    )
    resp.raise_for_status()
    logger.info(resp.text)
