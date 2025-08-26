"""Version agnostic database API module.

This module serves as the import point for functions defined for various versions
of the database API. Functions can be imported from this module instead of worrying
about handling imports from the various API sub-packages.
"""

import importlib
import logging
import os
from functools import lru_cache
from types import ModuleType
from typing import Callable

import lute.io._db.common_sqlite as common_sqlite
from lute.execution.logging import get_logger

if __debug__:
    logging.basicConfig(level=logging.DEBUG)
else:
    logging.basicConfig(level=logging.INFO)

logger: logging.Logger = get_logger(__name__, is_task=False)

LUTE_DB_CURRENT_SPEC_VERSION: int = 0x000002
LUTE_DB_DEFAULT_SPEC_VERSION: int = 0x000001

LUTE_DB_SPEC_VERSION: int = int(
    os.getenv("LUTE_DB_SPEC_VERSION", LUTE_DB_DEFAULT_SPEC_VERSION)
)


def lazy_import(func_name: str, api_version: int) -> Callable:
    """Return a lazily loaded version of the function."""

    def wrapper(*args, **kwargs):
        func: Callable = import_function(func_name=func_name, api_version=api_version)
        return func(*args, **kwargs)

    return wrapper


@lru_cache
def import_function(func_name: str, api_version: int) -> Callable:
    """Import a database function from the appropriate API version.

    Args:
        func_name (str): The name of the function to import.

        api_version (int): The API version. Currently either 0x000001 or 0x000002.

    Returns:
        func (Callable): The requested function.

    Raises:
        DatabaseError: Raised if the api_version is not supported.

        AttributeError: Raised if the function requested does not exist.
    """
    if api_version not in (1, 2):
        raise common_sqlite.DatabaseError(
            "Unrecognized database specification version! Set LUTE_DB_SPEC_VERSION appropriately! "
            "Supported versions: 0x000001 and 0x000002"
        )
    api_mod: ModuleType = importlib.import_module(f"lute.io._db.v{api_version}.api")
    try:
        func: Callable = getattr(api_mod, func_name)
    except AttributeError:
        logging.error(
            f"Attempting to retrieve database API non-existent function {func_name}!"
        )
        raise
    return func


record_analysis_db: Callable = lazy_import("record_analysis_db", LUTE_DB_SPEC_VERSION)
read_latest_db_entry: Callable = lazy_import(
    "read_latest_db_entry", LUTE_DB_SPEC_VERSION
)

if LUTE_DB_SPEC_VERSION == 0x000002:
    get_executions_summary: Callable = lazy_import(
        "get_executions_summary", LUTE_DB_SPEC_VERSION
    )
    get_task_parameters_summary: Callable = lazy_import(
        "get_task_parameters_summary", LUTE_DB_SPEC_VERSION
    )
