"""Launch a LUTE workflow using Maestro as the workflow manager."""

__all__ = []
__author__ = "Gabriel Dorlhiac"

import argparse
import collections
import logging
import os
import socket
import sys
from typing import Dict, List, Optional

import requests

from maestro._maestro import _maestro
from maestro.parser import load_lute_dag

logger: logging.Logger = logging.getLogger("PyMaestro")
handler: logging.Handler = logging.StreamHandler()
formatter: logging.Formatter = logging.Formatter(logging.BASIC_FORMAT)
handler.setFormatter(formatter)
logger.addHandler(handler)


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


def get_concurrent_job_steps(wf: List[_maestro.JobStep]) -> int:
    """Return the maximum number of concurrent JobSteps.

    This can be used to determine how many threads to add to the threadpool for the
    workflow manager.

    NOTE: This is a very basic calculation - if you have complicated branch structures
    it may undershoot the number of concurrent jobs. For safety you can add one to
    the returned value - this will likely cover 99% of all workflow cases.

    Args:
        wf (List[_maestro.JobStep]): The workflow.

    Returns:
        max_concurrent_jobs (int): The maximum number of jobs found to run in
            parallel at any given time.
    """
    num_concurrent_steps: int = len(wf)
    for step in wf:
        next_concurrent_steps: int = get_concurrent_job_steps(step.next)
        num_concurrent_steps = max(num_concurrent_steps, next_concurrent_steps)

    return num_concurrent_steps


def main():
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        prog="launch_maestro",
        description="A light-weight workflow manager which executes LUTE Managed Tasks.",
        epilog="Refer to https://github.com/slac-lcls/lute for more information.",
    )
    # We pop out the optional args for changing the order of the help message.
    # We'll add it back later
    optional_args: argparse._ArgumentGroup = parser._action_groups.pop()

    # Required arguments
    required_args: argparse._ArgumentGroup = parser.add_argument_group(
        "required arguments"
    )
    required_args.add_argument(
        "-c", "--config", type=str, help="Path to config YAML file.", required=True
    )
    required_args.add_argument(
        "-W",
        "--workflow_defn",
        type=str,
        help="Path to a YAML file with workflow.",
        required=True,
    )

    # Arguments required for when running from command-line
    non_arp_required_args: argparse._ArgumentGroup = parser.add_argument_group(
        "required arguments when running without the ARP"
    )
    non_arp_required_args.add_argument(
        "-e",
        "--experiment",
        type=str,
        help="Provide an experiment if not running with ARP.",
        required=False,
    )
    non_arp_required_args.add_argument(
        "-r",
        "--run",
        type=str,
        help="Provide a run number if not running with ARP.",
        required=False,
    )

    # Optional Arguments
    optional_args.add_argument(
        "-d", "--debug", help="Run in debug mode.", action="store_true"
    )
    optional_args.add_argument(
        "--num_server_threads",
        type=int,
        help="Number of threads to use for the HTTP server.",
        default=2,
    )
    optional_args.add_argument(
        "--unbuffered",
        help=(
            "Flush logs immediately. Warning: This can make output confusing "
            "when running multiple managed Tasks are running in parallel."
        ),
        action="store_true",
    )
    parser._action_groups.append(optional_args)

    args: argparse.Namespace
    extra_args: List[str]  # Should contain all SLURM arguments!
    args, extra_args = parser.parse_known_args()

    # Do we use any APIs now that need kerberos ticket if not using JID?
    # Maybe can get rid of this for the SLURM only submission?
    cache_file: Optional[str] = os.getenv("KRB5CCNAME")
    if os.getenv("EXPERIMENT") is None or os.getenv("RUN_NUM") is None:
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
                break
        else:
            is_daq2 = False

    lute_location: str = os.path.abspath(f"{os.path.dirname(__file__)}/..")

    wf_defn: List[_maestro.JobStep] = load_lute_dag(
        workflow_path=args.workflow_defn,
        lute_location=lute_location,
        config_file=args.config,
        debug=args.debug,
        default_slurm_params=" ".join(extra_args),
    )

    num_concurrent_steps: int = get_concurrent_job_steps(wf_defn)
    manager_host: str = socket.gethostname()
    manager_port: int = 41239
    os.environ["LUTE_MANAGER_URL"] = f"{manager_host}:{manager_port}"
    # fmt: off
    manager_params: _maestro.ManagerParameters = _maestro.ManagerParameters(
        num_concurrent_steps,                # Manager threads
        args.num_server_threads,             # Server threads
        args.unbuffered,                     # Unbuffered logs
        "0.0.0.0",                           # Server IP
        manager_port,                        # Server port
        _maestro.LauncherType.SlurmLauncher, # Launch mechanism
        is_daq2,                             # Is daq2?
        run_type,                            # Run type
    )
    # fmt: on

    _maestro.run_workflow(wf_defn, manager_params)


if __name__ == "__main__":
    main()
