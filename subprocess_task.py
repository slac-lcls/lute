import sys
import os
import argparse
import logging
import signal
import types
import importlib.util
from typing import Type, Optional, Dict, Any

from lute.tasks.task import Task, BinaryTask
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
module_with_task: Optional[str] = None
lute_path: str = os.getenv("LUTE_PATH", os.path.dirname(__file__))
if isinstance(task_parameters, BaseBinaryParameters):
    TaskType = BinaryTask
else:
    for module_name in os.listdir(f"{lute_path}/lute/tasks"):
        if module_name.endswith(".py") and module_name not in [
            "dataclasses.py",
            "task.py",
            "__init__.py",
        ]:
            with open(f"{lute_path}/lute/tasks/{module_name}", "r") as f:
                txt: str = f.read()
                if f"class {task_name}(Task):" in txt:
                    module_with_task = module_name[:-3]
                    del txt
                    break
    else:
        logger.debug(
            f"Task {task_name} not found while scanning directory: `{lute_path}/lute/tasks`."
        )
        sys.exit(-1)

# If we got this far we should have a module or are BinaryTask
if module_with_task is not None:
    spec: importlib.machinery.ModuleSpec = importlib.util.spec_from_file_location(
        module_with_task, f"{lute_path}/lute/tasks/{module_with_task}.py"
    )
    task_module: types.ModuleType = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(task_module)
    TaskType: Type[Task] = getattr(task_module, f"{task_name}")

task: Task = TaskType(params=task_parameters)
task.run()
