"""Tasklet functions to be run before or after a Task.

"Tasklets" are simple processing steps which are intended to be run by an
Executor after the main Task has finished. These functions help provide a way to
handle some common operations which may need to be performed, such as examining
an output text file, or doing simple file conversions. These operations can be
also be imported for use in first-party Tasks. In particular for
`ThirdPartyTask`s, it is not easy to append operations to a Task which is why
the Executor mechanism is provided.

Functions:
    git_clone(repo: str, location: str, permissions: str): Clone a git repo.

    concat_files(location: str, in_files_glob: str, out_file: str): Concatenate
        a group of files into a single output file.

    grep(match_str: str, in_file: str) -> str | List[str]: grep for text in a
        specific file. Returns the results.

Usage:
    As tasklets are just functions they can be imported and used within Task
    code normally if needed.

    However, tasklets can also be managed through the Executor in a similar way
    to environment changes. E.g., to add a tasklet to an Executor instance one
    would:

    # First create Executor instance as normal to run the Task
    MyTaskRunner: Executor = Executor("RunMyTask")
    #MyTaskRunner.update_environment(...) # if needed
    MyTaskRunner.add_tasklet(
        tasklet, args, when="before", set_result=False, set_summary=False
    )

    A special substitution syntax can be used in args if specific values
    from a `TaskParameters` object will be needed to run the Tasklet:
        args=("{{ param_to_sub }}", ...)
"""

__all__ = ["grep", "git_clone"]
__author__ = "Gabriel Dorlhiac"

import logging
import os
import subprocess
from typing import Union, List


if __debug__:
    logging.basicConfig(level=logging.DEBUG)
else:
    logging.basicConfig(level=logging.INFO)

logger: logging.Logger = logging.getLogger(__name__)


def concat_files(location: str, in_files_glob: str, out_file: str) -> None:
    """Concatenate a series of files into a single output file.

    Args:
        location (str): Path to the files to concatenate.

        in_files_glob (str): A glob to match a series of files at the specified
            path. These will all be concatenated.

        out_file (str): Name of the concatenated output.
    """
    ...


def git_clone(repo: str, location: str, permissions: str) -> None:
    """Clone a git repository.

    Will not overwrite a directory of there is already a folder at the specified
    location.

    Args:
        repo (str): Name of the repository to clone. Should be specified as:
            "<user_or_organization>/<repository_name>"

        location (str): Path to the location to clone to.

        permissions (str): Permissions to set on the repository.
    """
    repo_only: str = repo.split("/")[1]
    if os.path.exists(f"location/{repo_only}"):
        logger.debug(
            f"Repository {repo} already exists at {location}. Will not overwrite."
        )
        return
    cmd: List[str] = ["git", "clone", f"https://github.com/{repo}.git", location]
    out: str
    out, _ = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True
    ).communicate()


def grep(match_str: str, in_file: str) -> Union[str, List[str], None]:
    """Grep for specific lines of text output.

    Args:
        match_str (str): String to search for.

        in_file (str): File to search.

    Returns:
        lines (str | List[str]): The matches. It may be an empty string if
            nothing is found.
    """
    cmd: List[str] = ["grep", match_str, in_file]
    out: str
    out, _ = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True
    ).communicate()

    lines: List[str] = out.split("\n")
    return lines[0] if len(lines) == 1 else lines
