"""Launch a LUTE workflow using Maestro as the workflow manager."""

__all__ = []
__author__ = "Gabriel Dorlhiac"

import logging
import os
import random
import socket
import sys
from typing import List, Dict, Optional

from lute.execution.launch import (
    get_base_launch_parser,
    setup_launch_env,
    retrieve_run_info,
    get_concurrent_job_steps,
)
from maestro._maestro import _maestro
from maestro.parser import load_lute_dag

logger: logging.Logger = logging.getLogger("PyMaestro")
handler: logging.Handler = logging.StreamHandler()
formatter: logging.Formatter = logging.Formatter(logging.BASIC_FORMAT)
handler.setFormatter(formatter)
logger.addHandler(handler)


def find_random_port(
    min_port: int = 41923, max_port: int = 64324, max_tries: int = 20
) -> Optional[int]:
    ports: List[int] = random.choices(range(min_port, max_port), k=max_tries)

    for port in ports:
        sock: socket.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        try:
            sock.bind(("", port))
            sock.close()
            del sock
            return port
        except Exception:
            continue
    return None


def main():
    parser = get_base_launch_parser(
        "A light-weight workflow manager which executes LUTE Managed Tasks."
    )
    parser.add_argument(
        "--env_policy",
        type=int,
        help=(
            "Select policy for what maestro provides to the managed Task environment.\n"
            " - 0 == REPLACE: LaunchContext fully determines env.\n"
            " - 1 == MERGE: LaunchContext env merged with the maestro env.\n"
            " - 2 == IGNORE: LaunchContext env ignored. Task receives maestro env.\n"
            "Default: 1 (MERGE)."
        ),
        default=1,
        choices=[0, 1, 2],
    )
    parser.add_argument(
        "--launcher_type",
        type=int,
        help="Select process Launcher to use: 1 == Python, 2 == SLURM. Default: 2.",
        default=2,
        choices=[1, 2],
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

    launcher_type: _maestro.LauncherType
    if args.launcher_type == 1:
        launcher_type = _maestro.LauncherType.PythonLauncherType
    elif args.launcher_type == 2:
        launcher_type = _maestro.LauncherType.SlurmLauncherType
    else:
        raise RuntimeError(
            f"Unknonw LauncherType requested (`{args.launcher_type}`)! Select 1 or 2!"
        )

    # All environment modification is done through LaunchContext below
    # so add_to_env is False
    launch_env = setup_launch_env(args, add_to_env=False)
    experiment = launch_env["experiment"]
    run_num = launch_env["run_num"]
    jid_authorization = launch_env["authorization"]

    run_type, is_daq2 = retrieve_run_info(
        experiment, run_num, jid_authorization, args.type
    )
    branch_conditions: Dict[str, bool] = {
        "daq2": is_daq2 if is_daq2 is not None else True,
    }

    bin_dir: str = os.path.dirname(os.path.realpath(sys.argv[0]))
    lute_location: str = os.path.abspath(f"{bin_dir}/..")

    wf_defn: List[_maestro.JobStep] = load_lute_dag(
        workflow_path=args.workflow_defn,
        lute_location=lute_location,
        executable_subdir=bin_dir.split("/")[-1],
        config_file=args.config,
        debug=args.debug,
        default_slurm_params=" ".join(extra_args),
        branch_conditions=branch_conditions,
        run_type=run_type,
    )

    num_concurrent_steps: int = get_concurrent_job_steps(wf_defn)
    manager_host: str = socket.gethostname()
    manager_port: Optional[int] = find_random_port()
    if manager_port is None:
        raise RuntimeError("Cannot bind a port!")

    executor_env: Dict[str, str] = {
        "LUTE_MANAGER_URL": f"{manager_host}:{manager_port}"
    }
    for k, v in launch_env.items():
        if isinstance(v, str):
            executor_env[k] = v
        else:
            # Convert a potentially None kerb_file to empty string
            executor_env[k] = ""

    # Setup extra arguments
    extra_launch_args: List[str] = []
    if is_daq2:
        extra_launch_args.append("--psana2")
    else:
        extra_launch_args.append("--psana1")

    launch_labels: Dict[str, str] = {
        "run_type": run_type,
    }

    env_policy: _maestro.LaunchEnvPolicy
    if args.env_policy == 0:
        env_policy = _maestro.LaunchEnvPolicy.CTX_REPLACE
    elif args.env_policy == 1:
        env_policy = _maestro.LaunchEnvPolicy.CTX_MERGE
    elif args.env_policy == 2:
        env_policy = _maestro.LaunchEnvPolicy.CTX_IGNORE
    else:
        raise RuntimeError("Environment Policy (--env_policy) must be 0, 1, or 2!")

    launch_ctx: _maestro.LaunchContext = _maestro.LaunchContext(
        workflow_id="",  # Left empty, will be generated
        env_policy=env_policy,
        env=executor_env,
        extra_launch_args=extra_launch_args,
        labels=launch_labels,
    )

    manager_params: _maestro.ManagerParameters = _maestro.ManagerParameters(
        num_manager_threads=num_concurrent_steps,
        num_server_threads=args.num_server_threads,
        unbuffered_logs=args.unbuffered,
        host="0.0.0.0",
        port=manager_port,
        launch_type=launcher_type,
        context=launch_ctx,
        run_type=run_type,
    )

    status: str = _maestro.run_workflow(wf_defn, manager_params)
    if status == "FAILED":
        raise RuntimeError("Workflow Failed!")


if __name__ == "__main__":
    main()
