#!/sdf/group/lcls/ds/ana/sw/conda1/inst/envs/ana-4.0.62-py3/bin/python

"""Script submitted by Automated Run Processor (ARP) to trigger a SLURM-job workflow.

This script is submitted by the ARP to the batch nodes. It runs a batch job which itself
submits the individual workflow job steps.
"""

__author__ = "Gabriel Dorlhiac"

import argparse
import logging
import os
import re
import subprocess
import sys
import yaml
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Literal, Optional, Tuple, Union, overload
from typing_extensions import TypedDict

logger: logging.Logger = logging.getLogger("Launch_SLURM_Workflow")
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


class MissingParametersException(Exception): ...


@overload
def _run_subprocess_log(cmd: List[str], return_output: Literal[True]) -> str: ...


@overload
def _run_subprocess_log(cmd: List[str], return_output: Literal[False]) -> None: ...


@overload
def _run_subprocess_log(cmd: List[str]) -> None: ...


def _run_subprocess_log(cmd: List[str], return_output: bool = False) -> Optional[str]:
    """Run a subprocess with logging."""
    global logger

    out: str
    err: str
    out, err = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True
    ).communicate()
    # if out:
    #    logger.info(out)
    # if err:
    #    logger.info(err)

    if return_output:
        return out
    return None


def get_slurm_logfile_path(jobid: str) -> str:
    # scontrol show job <JobID> | grep StdOut | cut -d "=" -f2
    # may only work for job that is currently running
    # Would maybe run this first command and save the log file path
    scontrol_cmd: List[str] = [
        "scontrol",
        "show",
        "job",
        jobid,
    ]
    scontrol_out: str = _run_subprocess_log(scontrol_cmd, return_output=True)
    pattern: str = r"StdOut=.*"
    logfile_path: str = (
        re.findall(pattern, scontrol_out)[0].replace("%J", jobid).split("=")[1]
    )
    return logfile_path


def get_slurm_log(logfile_path: str) -> str:
    read_logfile_cmd: List[str] = ["cat", logfile_path]
    logfile: str = _run_subprocess_log(read_logfile_cmd, return_output=True)

    return logfile


def get_slurm_job_status(jobid: str) -> str:
    status_cmd: List[str] = [
        "sacct",
        "-j",
        jobid,
        "-o",
        "State",
    ]
    status_cmd_out: str = _run_subprocess_log(status_cmd, return_output=True)
    status: str = status_cmd_out.split("\n")[2].strip().replace("+", "")
    return status


def prepare_launch_command(
    task_name: str,
    lute_params: LuteParams,
    slurm_params: str,
    kerb_file: Optional[str] = None,
) -> str:
    if lute_params == {}:
        logger.critical("Empty LUTE parameter dictionary! Need configuration YAML!")
        raise MissingParametersException

    config_path: str = lute_params["config_file"]
    lute_param_str: str
    if lute_params["debug"]:
        lute_param_str = f"--taskname {task_name} --config {config_path} --debug"
    else:
        lute_param_str = f"--taskname {task_name} --config {config_path}"

    if kerb_file is not None:
        lute_param_str = f"{lute_param_str} -K {kerb_file}"

    parameter_str: str = f"{lute_param_str} {slurm_params}"
    return parameter_str


def launch_lute_task(
    lute_location: str,
    task_name: str,
    lute_params: LuteParams,
    slurm_params: str,
    kerb_file: Optional[str] = None,
    wait_for: Optional[Future] = None,
) -> Tuple[str, str, str]:
    if wait_for is not None:
        while not wait_for.done():
            ...

        prev_task: str
        prev_status: str
        prev_task, prev_status, _ = wait_for.result()
        if prev_status in ("FAILED", "UPSTREAM_FAILED"):
            log: str = (
                f"---- UPSTREAM {prev_task} FAILED, NOT LAUNCHING {task_name} ----"
            )
            status: str = "UPSTREAM_FAILED"
            return task_name, status, log

    parameter_str: str = prepare_launch_command(
        task_name=task_name,
        lute_params=lute_params,
        slurm_params=slurm_params,
        kerb_file=kerb_file,
    )
    executable: str = f"{lute_location}/launch_scripts/submit_slurm.sh"
    launch_cmd: List[str] = [executable]
    launch_cmd.extend(parameter_str.split())

    logger.info(f"Submitting {task_name}")
    out: str = _run_subprocess_log(launch_cmd, return_output=True)

    # grab jobid from out
    pattern: str = r"Submitted batch job [0-9]{0,100}"
    jobid_full_str: str = re.findall(pattern, out)[0]

    jobid: str = jobid_full_str.split()[-1]  # out.split()[-1]

    logfile_path: str = get_slurm_logfile_path(jobid=jobid)
    # Loop on task here?
    logfile: str = f"---- NO LOGS FOR JOB {jobid} RUNNING {task_name} ----"
    while (status := get_slurm_job_status(jobid=jobid)) not in (
        "FAILED",
        "COMPLETED",
        "CANCELLED",
    ):
        logfile = get_slurm_log(logfile_path=logfile_path)

    return task_name, status, logfile


def create_workflow(
    executor: ThreadPoolExecutor,
    wait_for: Optional[Future],
    all_futures: List[Future],
    wf_dict: Union[Dict[str, Any], List[Dict[str, Any]]],
    lute_location: str,
    lute_params: LuteParams,
    kerb_file: Optional[str] = None,
) -> None:
    if isinstance(wf_dict, list):
        for task_dict in wf_dict:
            slurm_params: str = task_dict.get("slurm_params", "")
            future: Future = executor.submit(
                launch_lute_task,
                lute_location,
                task_dict["task_name"],
                lute_params,
                slurm_params,
                kerb_file,
                wait_for,
            )
            all_futures.append(future)
            if task_dict["next"] == []:
                return None
            else:
                for task in task_dict["next"]:
                    create_workflow(
                        executor,
                        wait_for=future,
                        all_futures=all_futures,
                        wf_dict=task,
                        lute_location=lute_location,
                        lute_params=lute_params,
                        kerb_file=kerb_file,
                    )
    else:
        slurm_params: str = wf_dict.get("slurm_params", "")
        future: Future = executor.submit(
            launch_lute_task,
            lute_location,
            wf_dict["task_name"],
            lute_params,
            slurm_params,
            kerb_file,
            wait_for,
        )
        all_futures.append(future)
        if wf_dict["next"] == []:
            return None
        else:
            for task in wf_dict["next"]:
                create_workflow(
                    executor,
                    wait_for=future,
                    all_futures=all_futures,
                    wf_dict=task,
                    lute_location=lute_location,
                    lute_params=lute_params,
                    kerb_file=kerb_file,
                )
    return None


def count_tasks_and_print_wf(
    wf: Union[List[Dict[str, Any]], Dict[str, Any]], wf_str: str, task_count: int = 0
) -> Tuple[str, int]:
    if isinstance(wf, list):
        return "", 1
    else:
        task_name: str = wf["task_name"]
        new_str: str
        if wf_str != "":
            new_str = f"{wf_str} >> {task_name}"
        else:
            new_str = task_name

        if wf["next"] == []:
            return f"{new_str}", 1
        else:
            task_count = 1
            full_branched_str: str = ""
            for task in wf["next"]:
                branch_str: str
                branch_task_count: int
                branch_str, branch_task_count = count_tasks_and_print_wf(
                    task, new_str, task_count
                )
                full_branched_str += f"{branch_str}\n"
                task_count += branch_task_count

            return full_branched_str, task_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="trigger_slurm_lute_workflow",
        description="Run a batch job which executes a LUTE workflow.",
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

    use_kerberos: bool = (
        True  # Always copy kerberos ticket so non-active experiments can work.
    )
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

    wf_defn: Dict[str, Any]
    with open(args.workflow_defn, "r") as f:
        wf_defn = yaml.load(stream=f, Loader=yaml.FullLoader)

    lute_location: str = os.path.abspath(f"{os.path.dirname(__file__)}/..")

    lute_params: LuteParams = {"config_file": args.config, "debug": args.debug}

    wf_repr: str
    task_count: int
    wf_repr, task_count = count_tasks_and_print_wf(wf_defn, "", 0)

    logger.info(f"Running the following workflow with {task_count} Managed Tasks:")
    print(wf_repr)
    with ThreadPoolExecutor(max_workers=task_count) as executor:
        all_futures: List[Future] = []
        # Recursively submit work to the ThreadPoolExecutor. The individual functions
        # submitted to the ThreadPoolExecutor wait on previous futures if
        # appropriate
        create_workflow(
            executor=executor,
            wait_for=None,
            all_futures=all_futures,
            wf_dict=wf_defn,
            lute_location=lute_location,
            lute_params=lute_params,
            kerb_file=cache_file,
        )

        n_failed: int = 0
        for future in as_completed(all_futures):  # as_completed from concurrent.futures
            task_name: str
            status: str
            logfile: str
            task_name, status, logfile = future.result()
            logger.info(f"Providing logs for {task_name}")
            print("-" * 50, flush=True)
            print(logfile, flush=True)
            print("-" * 50, flush=True)
            if status in ("FAILED", "UPSTREAM_FAILED"):
                # Need smarter way to determine workflow failure
                # For now count any step failing as the entire workflow failing
                n_failed += 1

        if n_failed > 0:
            logger.info("Workflow exited: FAILED")
        else:
            logger.info("Workflow exited: COMPLETED")
