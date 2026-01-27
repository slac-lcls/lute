"""Launch a LUTE workflow using Maestro as the workflow manager."""

__all__ = []
__author__ = "Gabriel Dorlhiac"

import logging
import os
import socket
import sys
from typing import List

from lute.execution.launch import (
    get_base_launch_parser,
    setup_launch_env,
    retrieve_run_info,
)
from maestro._maestro import _maestro
from maestro.parser import load_lute_dag

logger: logging.Logger = logging.getLogger("PyMaestro")
handler: logging.Handler = logging.StreamHandler()
formatter: logging.Formatter = logging.Formatter(logging.BASIC_FORMAT)
handler.setFormatter(formatter)
logger.addHandler(handler)


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
    parser = get_base_launch_parser(
        "A light-weight workflow manager which executes LUTE Managed Tasks."
    )
    parser.add_argument(
        "--num_server_threads",
        type=int,
        help="Number of threads to use for the HTTP server.",
        default=2,
    )
    parser.add_argument(
        "--unbuffered",
        help=(
            "Flush logs immediately. Warning: This can make output confusing "
            "when running multiple managed Tasks are running in parallel."
        ),
        action="store_true",
    )

    args, extra_args = parser.parse_known_args()

    launch_info = setup_launch_env(args)
    experiment = launch_info["experiment"]
    run_num = launch_info["run_num"]
    jid_authorization = launch_info["authorization"]

    run_type, is_daq2 = retrieve_run_info(
        experiment, run_num, jid_authorization, args.type
    )

    bin_dir: str = os.path.dirname(os.path.realpath(sys.argv[0]))
    lute_location: str = os.path.abspath(f"{bin_dir}/..")

    wf_defn: List[_maestro.JobStep] = load_lute_dag(
        workflow_path=args.workflow_defn,
        lute_location=lute_location,
        executable_subdir=bin_dir.split("/")[-1],
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
        num_concurrent_steps,                    # Manager threads
        args.num_server_threads,                 # Server threads
        args.unbuffered,                         # Unbuffered logs
        "0.0.0.0",                               # Server IP
        manager_port,                            # Server port
        _maestro.LauncherType.SlurmLauncherType, # Launch mechanism
        is_daq2,                                 # Is daq2?
        run_type,                                # Run type
    )
    # fmt: on

    _maestro.run_workflow(wf_defn, manager_params)


if __name__ == "__main__":
    main()
