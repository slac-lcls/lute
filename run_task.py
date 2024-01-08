import sys
import argparse
import logging
import os

import lute.tasks as tasks
from lute.io.config import *
from lute.execution.executor import *

if __debug__:
    logging.basicConfig(level=logging.DEBUG)
else:
    logging.basicConfig(level=logging.INFO)

logger: logging.Logger = logging.getLogger(__name__)

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

from lute import managed_tasks

if hasattr(managed_tasks, task_name):
    managed_task: Executor = getattr(managed_tasks, task_name)
else:
    logger.debug(f"{task_name} unrecognized!")
    sys.exit(-1)

managed_task.execute_task()
