"""Script submitted by Automated Run Processor (ARP) to trigger a Prefect flow.

This script is submitted by the ARP to the batch nodes. It triggers Prefect to
begin running the tasks of the specified deployment of a flow.
"""

__author__ = "Gabriel Dorlhiac"

import argparse
import collections
import datetime
import getpass
import logging
import os
import sys
import time
import uuid
import yaml
from typing import Any, Dict, List, Literal, Optional, Tuple
from typing_extensions import TypedDict

import requests
from requests.auth import HTTPBasicAuth

# Requests, urllib have lots of debug statements. Only set level for this logger
logger: logging.Logger = logging.getLogger("Launch_Prefect")
handler: logging.Handler = logging.StreamHandler()
formatter: logging.Formatter = logging.Formatter(logging.BASIC_FORMAT)
handler.setFormatter(formatter)
logger.addHandler(handler)

if __debug__:
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.INFO)


class LuteParams(TypedDict):
    config_file: str
    debug: bool


class FlowConf(TypedDict):
    experiment: str
    run_id: str
    JID_UPDATE_COUNTERS: Optional[str]
    ARP_ROOT_JOB_ID: str
    ARP_LOCATION: str
    Authorization: str
    user: str
    lute_location: str
    executable_subdir: str
    kerb_file: Optional[str]
    lute_params: LuteParams
    slurm_params: List[str]
    workflow: Dict[str, Any]
    run_type: Optional[str]
    is_daq2: Optional[bool]


class FlowRequestDict(TypedDict):
    parameters: Dict[Literal["flow_conf"], FlowConf]


def _retrieve_creds_and_url(instance: str = "experimental") -> Tuple[str, str, str]:
    path: str = "/sdf/group/lcls/ds/tools/lute/prefect_{instance}.txt"
    if instance == "experimental":
        path = path.format(instance=instance)
    else:
        raise ValueError('`instance` must be "experimental"')
    with open(path, "r") as f:
        user_pw: str = f.readline().strip()
        url: str = f.readline().strip()
    user: str
    pw: str
    user, pw = user_pw.split(":")
    return user, pw, url


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
    from kerberos import GSSError  # type: ignore
    from krtc import KerberosTicket  # type: ignore

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


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="trigger_prefect_lute_flow",
        description="Trigger Prefect to begin executing a LUTE flow.",
        epilog="Refer to https://github.com/slac-lcls/lute for more information.",
    )
    parser.add_argument("-c", "--config", type=str, help="Path to config YAML file.")
    parser.add_argument("-d", "--debug", help="Run in debug mode.", action="store_true")
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
    cache_file: Optional[str] = os.getenv("KRB5CCNAME")
    if (
        os.getenv("Authorization") is None
        or os.getenv("EXPERIMENT") is None
        or os.getenv("RUN_NUM") is None
    ):
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
        os.environ["ARP_JOB_ID"] = uuid.uuid4().hex[:24]

    user: str
    pw: str
    PREFECT_API_URL: str
    user, pw, PREFECT_API_URL = _retrieve_creds_and_url()
    auth: HTTPBasicAuth = HTTPBasicAuth(user, pw)

    flow_name: str = "lute_dynamic"
    deployment_name: str = "dev"

    csrf_endpoint: str = f"{PREFECT_API_URL}/csrf-token"
    name_endpoint: str = (
        f"{PREFECT_API_URL}/deployments/name/{flow_name}/{deployment_name}"
    )

    if not os.path.exists(args.workflow_defn):
        logger.error("Workflow definition path does not exist! Exiting!")
        sys.exit(-1)

    wf_defn: Dict[str, Any]
    with open(args.workflow_defn, "r") as f:
        wf_defn = yaml.load(stream=f, Loader=yaml.FullLoader)

    # Experiment, run #, and ARP env variables come from ARP submission only
    # We override above or exit if we cannot, so we cast here
    experiment: Optional[str] = os.getenv("EXPERIMENT")
    run_num: Optional[str] = os.getenv("RUN_NUM")
    arp_job_id: Optional[str] = os.getenv("ARP_JOB_ID")
    jid_authorization: Optional[str] = os.getenv("Authorization")
    assert isinstance(experiment, str)
    assert isinstance(run_num, str)
    assert isinstance(arp_job_id, str)
    assert isinstance(jid_authorization, str)

    elog_auth: Dict[str, str] = {
        "Authorization": jid_authorization,
    }
    base_url: str = "https://pswww.slac.stanford.edu/ws-jwt/lgbk/lgbk"
    run_doc_endpoint: str = f"{experiment}/ws/runs/{run_num}"
    run_doc_url: str = f"{base_url}/{run_doc_endpoint}"
    resp = requests.get(run_doc_url, headers=elog_auth)

    run_type: str
    is_daq2: Optional[bool] = None
    if resp.status_code != 200:
        logger.warning(
            "Unable to retrieve run document! No `run_type` information will be used! "
            "No information about psana1/psana2 can be retrieved. "
            "Workflow may be able to continue but this could point to issues with "
            "API access that lead to problems downstream."
        )
        run_type = "UNKNOWN"
    else:
        if args.type != "":
            run_type = args.type
        else:
            # If API request succeeds `type` should always be defined
            run_type = resp.json()["value"]["type"]
        # Try checking for "psana1" vs "psana2" by searching for "drp" in detector names
        param_keys: collections.abc.KeysView = resp.json()["value"]["params"].keys()
        for key in param_keys:
            if "/drp/" in key:
                # Detectors in LCLS2 DAQ are sent to eLog as "DAQ Detectors/drp/<name>"
                # In LCLS1 they are sent as "DAQ Detector/<name>"
                is_daq2 = True
        else:
            is_daq2 = False

    params: LuteParams = {"config_file": args.config, "debug": args.debug}

    conf: FlowConf = {
        "experiment": experiment,
        "run_id": f"{run_num}_{datetime.datetime.utcnow().isoformat()}",
        "JID_UPDATE_COUNTERS": os.getenv("JID_UPDATE_COUNTERS"),
        "ARP_ROOT_JOB_ID": arp_job_id,
        "ARP_LOCATION": os.getenv("ARP_LOCATION", "S3DF"),
        "Authorization": jid_authorization,
        "user": getpass.getuser(),
        "lute_location": os.path.abspath(f"{os.path.dirname(__file__)}/.."),
        "executable_subdir": os.path.abspath(os.path.dirname(__file__)).split("/")[-1],
        "kerb_file": cache_file,
        "lute_params": params,
        "slurm_params": extra_args,
        "workflow": wf_defn,
        "run_type": run_type,
        "is_daq2": is_daq2,  # True if LCLS2 DAQ, False if LCLS1 DAQ, None if undetermined
    }

    # Get CSRF
    ##############################################
    resp = requests.get(csrf_endpoint, auth=auth, params={"client": user})

    token: str = resp.json()["token"]
    client: str = resp.json()["client"]

    # Get ID from name
    ##############################################
    resp = requests.get(name_endpoint, auth=auth)

    deployment_id: str = resp.json()["id"]

    # Launch flow_run
    ##############################################
    launch_endpoint: str = (
        f"{PREFECT_API_URL}/deployments/{deployment_id}/create_flow_run"
    )

    data: FlowRequestDict = {"parameters": {"flow_conf": conf}}
    headers: Dict[str, str] = {
        "Prefect-Csrf-Token": token,
        "Prefect-Csrf-Client": client,
    }

    resp = requests.post(launch_endpoint, headers=headers, json=data, auth=auth)

    flow_run_id: str = resp.json()["id"]

    # Run loop, gather logs, etc.
    ##############################################
    flow_run_state_endpoint: str = f"{PREFECT_API_URL}/flow_runs/{flow_run_id}"
    log_endpoint: str = f"{PREFECT_API_URL}/logs/filter"

    log_payload: Dict[str, Any] = {
        "logs": {
            "level": {
                "ge_": 20,
            },
            "flow_run_id": {"any_": [flow_run_id]},
        },
        "sort": "TIMESTAMP_ASC",
    }

    resp = requests.get(flow_run_state_endpoint, auth=auth)
    state: str = resp.json()["state_type"]

    last_log_idx: int = 0
    while state in ("SCHEDULED", "PENDING", "RUNNING"):
        time.sleep(5)
        # Retrieve logs for flow run, printing new ones... Bit wasteful...
        resp = requests.post(log_endpoint, headers=headers, auth=auth, json=log_payload)
        current_len_logs: int = len(resp.json())
        log_dict: Dict[str, Any]
        for log_dict in resp.json()[last_log_idx:current_len_logs]:
            print(log_dict["message"])
        last_log_idx = current_len_logs

        resp = requests.get(flow_run_state_endpoint, auth=auth)
        state = resp.json()["state_type"]


if __name__ == "__main__":
    main()
