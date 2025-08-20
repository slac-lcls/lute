import argparse
import logging
import os
import signal
import sys
import types
from typing import Type, Optional, Dict, Any

from lute.tasks.task import Task, ThirdPartyTask
from lute.execution.ipc import Message
from lute.io.config import parse_config
from lute.io.models.base import TaskParameters, ThirdPartyParameters
from lute.io.db import record_parameters_db
from lute.io.parameters import RowIds


def get_task() -> Optional[Task]:
    """Return the current Task."""
    objects: Dict[str, Any] = globals()
    for _, obj in objects.items():
        if isinstance(obj, Task):
            return obj
    return None


def timeout_handler(signum: int, frame: Optional[types.FrameType]) -> Any:
    """Log and exit gracefully on Task timeout."""
    task: Optional[Task] = get_task()
    if task:
        msg: Message = Message(contents="Timed out.", signal="TASK_FAILED")
        task._report_to_executor(msg)
        task.clean_up_timeout()
        sys.exit(-1)


def setup_env() -> bool:
    """Setup a new Task environment for first-party Tasks.

    Returns:
        setup_new_env (bool): Returns True if a new environment was requested.
    """
    setup_new_env: bool = False
    new_env: Dict[str, str] = {}
    for key, value in os.environ.items():
        if "LUTE_TENV_" in key:
            # Set if using a custom environment
            setup_new_env = True
            new_key: str = key[10:]
            new_env[new_key] = value
    if setup_new_env:
        os.environ.update(new_env)
        sys.modules.pop("pydantic")
    return setup_new_env


signal.signal(signal.SIGALRM, timeout_handler)

if __debug__:
    logging.basicConfig(level=logging.DEBUG)
else:
    logging.basicConfig(level=logging.INFO)

logger: logging.Logger = logging.getLogger(__name__)

parser: argparse.ArgumentParser = argparse.ArgumentParser(
    prog="run_subprocess_task",
    description="Analysis Task run as a subprocess managed by a LUTE Executor.",
    epilog="Refer to https://github.com/slac-lcls/lute for more information.",
)
parser.add_argument(
    "-c", "--config", type=str, help="Path to config file with Task parameters."
)
parser.add_argument(
    "-t", "--taskname", type=str, help="Name of the Task to run.", default="test"
)

args: argparse.Namespace = parser.parse_args()
config: str = args.config
task_name: str = args.taskname
task_parameters: TaskParameters = parse_config(task_name=task_name, config_path=config)

# For now, we will only use the exec with first-party Task's that require a new env.
TaskType: Type[Task]
if isinstance(task_parameters, ThirdPartyParameters) or not setup_env():
    is_third_party = True
    if isinstance(task_parameters, ThirdPartyParameters):
        TaskType = ThirdPartyTask
    else:
        from lute.tasks import import_task, TaskNotFoundError

        try:
            TaskType = import_task(task_name=task_name)
        except TaskNotFoundError:
            logger.debug(
                (
                    f"Task {task_name} not found! Things to double check:"
                    "\t - The spelling of the Task name."
                    "\t - Has the Task been registered in lute.tasks.import_task."
                )
            )
            sys.exit(-1)
    task: Task = TaskType(params=task_parameters)
    task.run()
else:
    from lute.execution.subprocess_utils import exec_script_template

    # We are a first-party Task that needs a new environment
    # Record the parameters - but only once if using MPI
    use_mpi: bool = False
    rank: int = 0
    try:
        from mpi4py import MPI

        comm: MPI.Intracomm = MPI.COMM_WORLD
        size: int = comm.Get_size()
        rank = comm.Get_rank()
        if size > 1:
            use_mpi = True
            print(f"Running in a MPI world of size: {size}", flush=True)
    except ModuleNotFoundError:
        print("mpi4py not found. Assuming this is not an MPI-based `Task`", flush=True)
    row_ids: Optional[RowIds]
    if use_mpi:
        if rank == 0:
            row_ids = record_parameters_db(task_parameters)
        comm.Barrier()
    else:
        row_ids = record_parameters_db(task_parameters)
    work_dir: str = task_parameters.lute_config.work_dir

    new_python: str = f"{os.environ.get('CONDA_PREFIX')}/bin/python"

    exec_script: str = exec_script_template.format(
        work_dir=work_dir,
        task_name=task_name,
        row_ids=row_ids,
    )
    os.execlp(new_python, new_python, "-c", exec_script)
