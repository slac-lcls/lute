import sys
import argparse
import logging
import signal
import types
from typing import Type, Optional, Dict, Any

from lute.tasks.task import Task, ThirdPartyTask
from lute.execution.ipc import Message
from lute.io.config import *
from lute.io.models.base import TaskParameters, BaseBinaryParameters


def get_task() -> Optional[Task]:
    """Return the current Task."""
    objects: Dict[str, Any] = globals()
    for _, obj in objects.items():
        if isinstance(obj, Task):
            return obj
    return None


def timeout_handler(signum: int, frame: types.FrameType) -> None:
    """Log and exit gracefully on Task timeout."""
    task: Optional[Task] = get_task()
    if task:
        msg: Message = Message(contents="Timed out.", signal="TASK_FAILED")
        task._report_to_executor(msg)
        task.clean_up_timeout()
        sys.exit(-1)


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

# Hack to avoid importing modules with conflicting dependencie
TaskType: Type[Task]
if isinstance(task_parameters, BaseBinaryParameters):
    TaskType = ThirdPartyTask
else:
    from lute.tasks import import_task, TaskNotFoundError

    try:
        TaskType = import_task(task_name)
    except TaskNotFoundError as err:
        logger.debug(
            (
                f"Task {task_name} not found! Things to double check:\n"
                "\t - The spelling of the Task name.\n"
                "\t - Has the Task been registered in lute.tasks.import_task."
            )
        )
        sys.exit(-1)

task: Task = TaskType(params=task_parameters)
task.run()
