"""Machinary for the IO of configuration YAML files and their validation.

Functions:
    parse_config(taskname: str, config_path: str) -> TaskParameters: Parse a
        configuration file and return a TaskParameters object of validated
        parameters for a specific Task. Raises an exception if the provided
        configuration does not match the expected model.

Exceptions:
    ValidationError: Error raised by pydantic during data validation. (From
        Pydantic)
"""

# flake8: noqa: F403,F405

__all__ = ["parse_config"]
__author__ = "Gabriel Dorlhiac"

import base64
import logging
import os
import re
import subprocess
import warnings
import zlib
from typing import Callable, List, Dict, Iterator, Any, Union, Optional

import pprint
import yaml

from lute.tasks.dataclasses import VersionSpecifier

if __debug__:
    logging.basicConfig(level=logging.DEBUG)
else:
    logging.basicConfig(level=logging.INFO)

logger: logging.Logger = logging.getLogger(__name__)


def _record_git_diff(location: str, diff_args: Optional[List[str]] = None) -> str:
    """Collect the diff of the specified repository and then compress and encode it.

    Args:
        location (str): Full path to the repository.

        diff_args (Optional[List[str]]): A list of arguments to pass to git diff.
            E.g. can specify to exclude folders, ignore permission changes etc.

    Returns:
        enc_str (str): The compressed, b64 encoded, diff as a string.
    """
    if not os.path.exists(location):
        logger.warning(f"Requested git diff of {location} which does not exist!")
        return ""

    cmd: List[str] = [
        "git",
        "-c",
        "core.fileMode=false",  # With this option, ignore changes in file mode (executable/not executable)
        "diff",
    ]
    if diff_args:
        cmd.extend(diff_args)

    try:
        out: bytes
        out, _ = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=location
        ).communicate()

        compressed: bytes = zlib.compress(out)
        encoded: bytes = base64.b64encode(s=compressed)

        return encoded.decode("utf-8")
    except Exception as e:
        logger.warning(f"Error getting git diff for {location}: {e}")
        return ""


def _record_git_commit_hash(location: str) -> str:
    """Collect the git commit hash of the specified repository.

    Args:
        location (str): Full path to the repository.

    Returns:
        commit_hash (str): The git commit hash as a string.
    """
    if not os.path.exists(location):
        logger.warning(f"Requested git commit hash of {location} which does not exist!")
        return ""
    cmd: List[str] = [
        "git",
        "rev-parse",
        "HEAD",
    ]
    try:
        out: bytes
        out, _ = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=location
        ).communicate()

        return out.decode("utf-8").strip()
    except Exception as e:
        logger.warning(f"Error getting git commit hash for {location}: {e}")
        return ""


def _is_odd(num: int) -> bool:
    return not ((num ^ 1) == (num + 1))


def record_version(
    version_specifier: int = 0,
    location: Optional[str] = None,
    diff_args: Optional[List[str]] = None,
) -> Dict[str, str]:
    """Record version information for an execution of a Task.

    Args:
        version_specifier (int): Bitwise OR combination of VersionSpecifier enums.

        location (Optional[str]): Location of the git repository. Defaults to the
            LUTE repository.

        diff_args (Optional[List[str]]): A list of arguments to pass to git diff.
            E.g. can specify to exclude folders, ignore permission changes etc.

    Returns:
        version_info (Dict[str, str]): A dictionary of version details.
    """

    if version_specifier == 0:
        # No version selection information has been provided
        return {}

    lute_dir: Optional[str] = os.getenv("LUTE_PATH")
    if lute_dir is None:
        lute_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    if location is None:
        logger.warning("No version location provided! Cannot record version info!")

    # A complete version specification may contain multiple parts
    version_info: Dict[str, str] = {}

    # LUTE_VERSION is enumerator 1, so check if odd
    if _is_odd(num=version_specifier):
        # Use actual version? Or rely on diff until more strict about versioning?
        # from lute import __version__ as lute_ver
        # version_info["lute-version"] = lute_ver
        version_info["lute-version"] = _record_git_commit_hash(location=lute_dir)

    # GIT_SHA is enumerator 2
    if bool(version_specifier & 0b10) and location is not None:
        version_info["git-sha"] = _record_git_commit_hash(location=location)

    # GIT_DIFF is enumerator 4
    if bool(version_specifier & 0b100) and location is not None:
        version_info["git-diff"] = _record_git_diff(
            location=location, diff_args=diff_args
        )

    return version_info
