"""Tools for working with the LUTE parameter and configuration database.

The current implementation relies on a sqlite backend database. In the future
this may change - therefore relatively few high-level API function calls are
intended to be public. These abstract away the details of the database interface
and work exclusively on LUTE objects.

Functions:
    record_analysis_db(cfg: DescribedAnalysis) -> None: Writes the configuration
        to the backend database.

    read_latest_db_entry(db_dir: str, task_name: str, param: str) -> Any: Retrieve
        the most recent entry from a database for a specific Task.

Exceptions:
    DatabaseError: Generic exception raised for LUTE database errors.
"""

__all__ = [
    "record_analysis_db",
    "read_latest_db_entry",
]
__author__ = "Gabriel Dorlhiac"

import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

from lute.execution.logging import get_logger
from lute.io.models.base import TaskParameters
from lute.tasks.dataclasses import DescribedAnalysis

if __debug__:
    logging.basicConfig(level=logging.DEBUG)
else:
    logging.basicConfig(level=logging.INFO)

logger: logging.Logger = get_logger(__name__, is_task=False)


def record_analysis_db(cfg: DescribedAnalysis) -> None:
    """Write an DescribedAnalysis object to the database.

    The DescribedAnalysis object is maintained by the Executor and contains all
    information necessary to fully describe a single `Task` execution. The
    contained fields are split across multiple tables within the database as
    some of the information can be shared across multiple Tasks. Refer to
    `docs/design/database.md` for more information on the database specification.

    This function is meant to be called by the Executor at the end of Task
    execution, assuming the Task has not previously entered partial data into the
    database. See `record_parameters_db` and `update_task_entry_db` for how to
    handle the case where the Task and Executor both store some information.

    Args:
        cfg (DescribedAnalysis): The DescribedAnalysis completed by the Executor
            after Task completion.
    """
    import sqlite3
    from ._sqlite import create_tables, add_execution

    assert isinstance(cfg.task_parameters, TaskParameters)
    try:
        assert hasattr(cfg.task_parameters, "lute_config")
        work_dir: str = cfg.task_parameters.lute_config.work_dir
    except AttributeError:
        logger.error(
            (
                "Unable to access TaskParameters object. Likely wasn't created. "
                "Cannot store result."
            )
        )
        return
    assert hasattr(cfg.task_parameters, "lute_config")
    del cfg.task_parameters.lute_config.work_dir

    db_path: str = f"{work_dir}/lute.db"
    con: sqlite3.Connection = sqlite3.Connection(db_path)
    create_tables(con=con)
    with con:
        try:
            create_tables(con=con)
            add_execution(con=con, cfg=cfg)
        except sqlite3.OperationalError as err:
            logger.error(f"Database storage error: {err}")
    try:
        os.chmod(db_path, 0o664)
    except Exception:
        logger.error("Cannot setup permissions on database!")


def read_latest_db_entry(
    db_dir: str,
    task_name: str,
    param: str,
    valid_only: bool = True,
    for_run: Optional[Union[str, int]] = os.getenv("RUN"),
) -> Optional[Any]:
    """Read most recent value entered into the database for a Task parameter.

    (Will be updated for schema compliance as well as Task name.)

    Args:
        db_dir (str): Database location.

        task_name (str): The name of the Task to check the database for.

        param (str): The parameter name for the Task that we want to retrieve.

        valid_only (bool): Whether to consider only valid results or not. E.g.
            An input file may be useful even if the Task result is invalid
            (Failed). Default = True.

        for_run (Optional[str | int]): Only consider latest entries from the
            specific experiment run provided.

    Returns:
        val (Any): The most recently entered value for `param` of `task_name`
            that can be found in the database. Returns None if nothing found.
    """
    import sqlite3
    from ._sqlite import select_param_from_db

    db_path: str = f"{db_dir}/lute.db"
    con: sqlite3.Connection = sqlite3.Connection(db_path)
    with con:
        try:
            cond: Dict[str, str] = {}
            if valid_only:
                cond["valid_flag"] = "1"

            if for_run is not None:
                cond["run"] = str(for_run)

            return select_param_from_db(
                con=con, task_name=task_name, param_name=param, condition=cond
            )
        except sqlite3.OperationalError as err:
            logger.error(f"Cannot retrieve value {param} due to: {err}")
            return None
