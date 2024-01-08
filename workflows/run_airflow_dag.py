#!/sdf/group/lcls/ds/ana/sw/conda1/inst/envs/ana-4.0.47-py3/bin/python

"""Script submitted by Automated Run Processor (ARP) to trigger Airflow DAG.

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
import requests
from typing import Dict, Union

if __debug__:
    logging.basicConfig(level=logging.DEBUG)
else:
    logging.basicConfig(level=logging.INFO)

logger: logging.Logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="trigger_airflow_lute_dag",
        description="Trigger Airflow to begin executing a LUTE DAG.",
        epilog="Refer to https://github.com/slac-lcls/lute for more information.",
    )
    parser.add_argument(
        "-a",
        "--account",
        type=str,
        help="SLURM account.",
        default=f"lcls:{os.environ.get('EXPERIMENT', '')}",
    )
    parser.add_argument("-c", "--config", type=str, help="Path to config YAML file.")
    parser.add_argument(
        "-d", "--debug", type=str, help="Run in debug mode.", action="store_true"
    )
    parser.add_argument(
        "-l",
        "--lute_path",
        type=str,
        help="Path to the LUTE installation to use.",
        default="/sdf/group/lcls/ds/tools/lute",
    )
    parser.add_argument(
        "-n",
        "--ncores",
        type=int,
        help="Number of cores. Add an extra core for the Executor.",
        default=65,
    )
    parser.add_argument(
        "-q", "--queue", type=str, help="SLURM queue.", default="milano"
    )
    parser.add_argument(
        "-r", "--reservation", type=str, help="SLURM reservation", default=""
    )
    parser.add_argument(
        "-w", "--workflow", type=str, help="Workflow to run.", default="test"
    )

    args: argparse.Namespace = parser.parse_args()
    airflow_instance: str = "http://172.24.5.247:8080/"

    airflow_api_endpoints: Dict[str, str] = {
        "health": "api/v1/health",
        "run_dag": f"api/v1/dags/lute_{args.workflow}/dagRuns",
    }

    resp: requests.models.Response = requests.get(
        f"{airflow_instance}/{airflow_api_endpoints['health']}",
        auth=HTTPBasicAuth(),  # NEED AUTH SOLUTION
    )
    resp.raise_for_status()

    params: Dict[str, Union[str, int]] = {
        "config_file": args.config,
        # "dag": f"lute_{args.workflow}",
        "queue": args.queue,
        "ncores": args.ncores,
        "experiment": os.environ["EXPERIMENT"],
        "run_num": os.environ["RUN_NUM"],
        "account": args.account,
    }

    dag_run_data: Dict[str, Union[str, Dict[str, Union[str, int]]]] = {
        "dag_run_id": str(uuid.uuid4()),
        "conf": {
            "experiment": os.environ["EXPERIMENT"],
            "run_id": f"{os.environ['RUN_NUM']}{datetime.datetime.utcnow().isoformat()}",
            "JID_UPDATE_COUNTERS": os.environ["JID_UPDATE_COUNTERS"],
            "ARP_ROOT_JOB_ID": os.environ["ARP_JOB_ID"],
            "ARP_LOCATION": "S3DF",  # os.environ["ARP_LOCATION"],
            "Authorization": os.environ["Authorization"],
            "user": getpass.getuser(),
            "parameters": params,
        },
    }

    resp: requests.models.Response = requests.post(
        f"{airflow_instance}/{airflow_api_endpoints['run_dag']}",
        json=dag_run_data,
        auth=HTTPBasicAuth(),  # NEED AUTH SOLUTION
    )
    resp.raise_for_status()
    logger.info(resp.text)
