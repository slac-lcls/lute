import argparse
import logging
import os
import re
import socket
import subprocess
import sys
from typing import List, Optional

from lute.execution.executor import BaseExecutor, Executor
from lute import managed_tasks

if __debug__:
    logging.basicConfig(level=logging.DEBUG)
else:
    logging.basicConfig(level=logging.INFO)

logger: logging.Logger = logging.getLogger(__name__)

def setup_mpi_hostfile() -> None:
    """Prepare a hostfile for MPI if running in a SLURM job.

    This function creates a hostfile and sets an environment variable to the path
    where it was created. The hostfile can be used by Tasks that use MPI to determine
    available resources in a SLURM allocation. Depending on environment configuration
    and/or MPI version, the automated MPI mechanism may not work.
    """
    nodelist: str = os.getenv("SLURM_JOB_NODELIST", "")
    if nodelist:
        result: subprocess.CompletedProcess = subprocess.run(
            ["scontrol", "show", "hostnames", nodelist],
            capture_output=True,
            text=True,
            check=True,
        )
        nodes: List[str] = result.stdout.splitlines()
        tpn_str: str = os.getenv("SLURM_TASKS_PER_NODE", "")
        tpn_list: List[int] = []
        for part in tpn_str.split(","):
            m: Optional[re.Match] = re.match(r"(\d+)\(x(\d+)\)", part)
            if m:
                tasks: int = int(m.group(1))
                count: int = int(m.group(2))
                tpn_list.extend([tasks] * count)
            else:
                try:
                    tpn_list.append(int(part))
                except ValueError:
                    pass
        job_id: Optional[str] = os.getenv("SLURM_JOB_ID")
        assert isinstance(job_id, str)
        hostfile_path: str = os.path.abspath(f"lute_hostfile_{job_id}.hosts")
        executor_host: str = socket.gethostname()
        with open(hostfile_path, "w") as f:
            node: str
            tpn: int
            for node, tpn in zip(nodes, tpn_list):
                n_slots: int
                if node == executor_host:
                    n_slots = tpn - 1
                else:
                    n_slots = tpn
                f.write(f"{node} slots={n_slots}\n")

        # Task layer will look for this environment variable
        os.environ["LUTE_MPI_HOSTFILE_PATH"] = hostfile_path


def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        prog="run_managed_task",
        description="Run a LUTE managed task.",
        epilog="Refer to https://github.com/slac-lcls/lute for more information.",
    )
    parser.add_argument(
        "-c", "--config", type=str, help="Path to config file with Task parameters."
    )
    parser.add_argument(
        "-t",
        "--taskname",
        type=str,
        help="Name of the Managed Task to run.",
        default="test",
    )

    args: argparse.Namespace = parser.parse_args()
    config: str = args.config
    task_name: str = args.taskname

    # Environment variables need to be set before importing Executors
    os.environ["LUTE_CONFIGPATH"] = config

    # Prepare hostfile in case using MPI
    setup_mpi_hostfile()

    if hasattr(managed_tasks, task_name):
        managed_task: Executor = getattr(managed_tasks, task_name)
    else:
        import difflib

        logger.error(f"{task_name} unrecognized!")
        valid_names: List[str] = [
            name
            for name in dir(managed_tasks)
            if isinstance(getattr(managed_tasks, name), BaseExecutor)
        ]
        # List below may be empty...
        possible_options: List[str] = difflib.get_close_matches(
            task_name, valid_names, n=2, cutoff=0.1
        )
        if possible_options:
            logger.info(f"Perhaps you meant: {possible_options}?")
            logger.info(f"All possible options are: {valid_names}")
        else:
            logger.info(
                f"Could not infer a close match for the managed Task name. Possible options are: {valid_names}"
            )
        sys.exit(-1)

    managed_task._m_task_name = task_name
    managed_task.execute_task()


if __name__ == "__main__":
    main()
