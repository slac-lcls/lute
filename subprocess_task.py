import sys
import argparse
import logging
from typing import Type

from lute.io.config import *
from lute import tasks

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

try:
    if hasattr(task_parameters, "executable"):
        TaskType: Type[tasks.Task] = getattr(tasks, "BinaryTask")
    else:
        TaskType: Type[tasks.Task] = getattr(tasks, f"{task_name}")
except AttributeError:
    logger.debug(f"Task {task_name} unrecognized! Exiting")
    sys.exit(-1)

task: tasks.Task = TaskType(params=task_parameters)
task.run()
