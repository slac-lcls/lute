import sys
import argparse
import logging
import os
from typing import List

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
    import difflib

    from lute.execution.executor import BaseExecutor

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

managed_task.execute_task()
