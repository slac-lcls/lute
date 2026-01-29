import sys
import argparse
import logging
import os
import re
from typing import List, Optional

from lute.execution.executor import BaseExecutor
from lute import managed_tasks

if __debug__:
    logging.basicConfig(level=logging.DEBUG)
else:
    logging.basicConfig(level=logging.INFO)

logger: logging.Logger = logging.getLogger(__name__)


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

    # Try to find the *managed* Task - In case of generated parameter combos, there
    # may be a suffix in the name, e.g. Tester_0, Tester_1, so check for this
    # if names don't match
    managed_task: BaseExecutor
    if hasattr(managed_tasks, task_name):
        managed_task = getattr(managed_tasks, task_name)
    else:
        # Check if this is a generated Task name with suffix (e.g., TaskName_0)
        # Try stripping the numeric suffix
        base_task_name: str = re.sub(r"_\d+$", "", task_name)
        suffix_match: Optional[re.Match[str]] = re.search(r"(_\d+)$", task_name)
        suffix: str = suffix_match.group(1) if suffix_match else ""

        if base_task_name != task_name and hasattr(managed_tasks, base_task_name):
            # Found the underlying managed Task
            logger.info(
                f"Parameter sweep managed Task detected: '{task_name}' -> "
                f"base task '{base_task_name}'"
            )
            managed_task = getattr(managed_tasks, base_task_name)

            # IMPORTANT: For config lookup, we need Task name + suffix
            # Full process is:
            #   1. Find the *managed* Task: TestRunner_0 -> TestRunner
            #   2. For this instance of the *managed* Task, change the name
            #      of the Task it runs so that config look up works later.
            #      TestRunner.task_name = Test -> TestRunner.task_name = Test_0
            if hasattr(managed_task, "task_name"):
                suffixed_task_name: str = f"{managed_task.task_name}{suffix}"
                logger.info(f"Resolved config section name: '{suffixed_task_name}'")
                managed_task.task_name = suffixed_task_name
        else:
            # Task not found even after stripping suffix
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

    # Set the name of the *managed* Task (Executor uses this for DB calls)
    managed_task._m_task_name = task_name
    managed_task.execute_task()


if __name__ == "__main__":
    main()
