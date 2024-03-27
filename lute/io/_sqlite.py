"""Backend SQLite database utilites.

Functions should be used only by the higher-level database module.
"""

__all__ = ["write_cfg_to_db", "read_latest_db_entry"]
__author__ = "Gabriel Dorlhiac"

import sqlite3
import logging
from typing import List, Dict, Dict, Any, Tuple, Optional

if __debug__:
    logging.basicConfig(level=logging.DEBUG)
else:
    logging.basicConfig(level=logging.INFO)

logger: logging.Logger = logging.getLogger(__name__)


def _does_table_exist(con: sqlite3.Connection, table_name: str) -> bool:
    """Check whether a table exists.

    Args:
        con (sqlite3.Connection): Database connection.

        table_name (str): The table to check for.

    Returns:
        does_exist (bool): Whether the table exists.
    """
    res: sqlite3.Cursor = con.execute(
        f"SELECT name FROM sqlite_master WHERE name='{table_name}'"
    )
    if res.fetchone() is None:
        return False
    else:
        return True


def _get_tables(con: sqlite3.Connection) -> List[str]:
    """Retrieve a list of all tables in a database.

    Args:
        con (sqlite3.Connection): Database connection.

    Returns:
        tables (List[str]): A list of database tables.
    """
    # sql: str = "SELECT name FROM sqlite_schema"
    sql: str = (
        "SELECT name FROM sqlite_schema "
        "WHERE type = 'table' "
        "AND name NOT LIKE 'sqlite_%'"
    )
    with con:
        res: sqlite3.Cursor = con.execute(sql)

    tables: List[str] = [table[0] for table in res.fetchall()]
    return tables


def _get_table_cols(con: sqlite3.Connection, table_name: str) -> Dict[str, str]:
    """Retrieve the columns currently present in a table.

    Args:
        con (sqlite3.Connection): Database connection.

        table_name (str): The table's name.

    Returns:
        cols (Dict[str, str]): A dictionary of column names and types.
    """
    res: sqlite3.Cursor = con.execute(f"PRAGMA table_info({table_name})")
    # Retrieves: list = [col_id, col_name, col_type, -, default_val, -]
    table_info: List[Tuple[int, str, str, int, str, int]] = res.fetchall()

    cols: Dict[str, str] = {col[1]: col[2] for col in table_info}
    return cols


def _get_all_rows_for_table(
    con: sqlite3.Connection, table_name: str
) -> List[Tuple[Any, ...]]:
    """Return all rows for a requested table.

    Args:
        con (sqlite3.Connection): Database connection.

        table_name (str): The table's name.

    Returns:
        rows (List[Tuple[Any, ...]]): ALL rows for a table.
    """
    sql: str = f'SELECT * FROM "{table_name}"'
    with con:
        res: sqlite3.Cursor = con.execute(sql)

    rows: List[Tuple[Any, ...]] = res.fetchall()
    return rows


def _compare_cols(
    cols1: Dict[str, str], cols2: Dict[str, str]
) -> Optional[Dict[str, str]]:
    """Compare whether two sets of columns are identical.

    The comparison is unidirectional - This function tests for columns present
    in `cols2` which are not present in `cols1`, but NOT vice versa. Switch the
    order of the arguments in order to retrieve the other comparison.

    Args:
        cols1 (Dict[str, str]): Dictionary of first set of column names and
            types.

        cols2 (Dict[str, str]): Dictionary of second set of column names and
            types.

    Returns:
        diff (Dict[str, str] | None): Any columns present in `cols2` which
            are not present in `cols1`. If `cols2` has no entries which are
            not present in `cols1`, returns `None`.
    """
    diff: Dict[str, str] = {}

    for col_name in cols2.keys():
        if col_name not in cols1.keys():
            diff[col_name] = cols2[col_name]

    return diff if diff else None


def _make_task_table(
    con: sqlite3.Connection, task_name: str, columns: Dict[str, str]
) -> bool:
    """Create  a sqlite Task table for LUTE's specification.

    Args:
        con (sqlite3.Connection): Database connection.

        task_name (str): The Task's name. This will be provided by the Task.
            In most cases this is the Python class' name.

        columns (Dict[str, str]): A dictionary of columns in the format of
            {COLNAME:TYPE}. These match the parameters of the Task and the Result
            fields of the Task. Additional more general columns are appended
            within this function. Other helper functions can be used for
            generating the dictionary from a TaskParameters object.

    Returns:
        success (bool): Whether the table was created correctly or not.
    """
    # Check existence explicitly because may need to modify...
    if _does_table_exist(con, task_name):
        # Compare current columns vs new columns - the same Task can have
        # different number of parameters -> May need to adjust cols.
        current_cols: Dict[str, str] = _get_table_cols(con, task_name)
        if diff := _compare_cols(current_cols, columns):
            for col in diff.items():
                sql: str = f"ALTER TABLE {task_name} ADD COLUMN {col[0]} {col[1]}"
                with con:
                    con.execute(sql)

    # Table does not yet exist -> Create it
    # Need to escape column names using double quotes since they
    # may contain periods.
    col_str: str = ", ".join(f'"{col}" {columns[col]}' for col in columns)
    db_str: str = (
        f"{task_name}(id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "timestamp DATETIME DEFAULT CURRENT_TIMESTAMP, "
        f"gen_cfg_id INTEGER, exec_cfg_id INTEGER, {col_str}, "
        "valid_flag INTEGER)"
    )
    sql: str = f"CREATE TABLE IF NOT EXISTS {db_str}"
    with con:
        con.execute(sql)
    return _does_table_exist(con, task_name)


def _make_shared_table(
    con: sqlite3.Connection, table_name: str, columns: Dict[str, str]
) -> bool:
    """Create a general configuration table.

    Args:
        con (sqlite3.Connection): Database connection.

        table_name (str): Name of the table to create.

        columns (Dict[str, str]): A dictionary of columns in the format of
            {COLNAME:TYPE}.
    """
    col_str: str = ", ".join(f"{col} {columns[col]}" for col in columns)
    db_str: str = f"{table_name}(id INTEGER PRIMARY KEY AUTOINCREMENT, {col_str})"
    sql: str = f"CREATE TABLE IF NOT EXISTS {db_str}"
    with con:
        con.execute(sql)
    return _does_table_exist(con, table_name)


def _add_task_entry(
    con: sqlite3.Connection,
    task_name: str,
    entry: Dict[str, Any],
) -> None:
    """Add an entry to a task table.

    Args:
        con (sqlite3.Connection): Database connection.

        task_name (str): The Task's name. This will be provided by the Task.
            In most cases this is the Python class' name.

        entry (Dict[str, Any]): A dictionary of entries in the format of
            {COLUMN: ENTRY}. These are assumed to match the columns of the table.
    """
    placeholder_str: str = ", ".join("?" for x in range(len(entry)))
    keys: List[str] = []
    values: List[str] = []
    for key, value in entry.items():
        keys.append(f'"{key}"')
        values.append(value)
    with con:
        ins_str: str = "".join(f':"{x}", ' for x in entry.keys())[:-2]
        res = con.execute(
            f"INSERT INTO {task_name} ({','.join(keys)}) VALUES ({placeholder_str})",
            values,
        )


def _add_row_no_duplicate(
    con: sqlite3.Connection, table_name: str, entry: Dict[str, Any]
) -> int:
    """Add a new row to a table with no duplicates.

    This function will check first to see if the entry exists in the table. If
    there is already a row with the provided information, its ID is returned.
    Otherwise a new row is added and the ID of the newly inserted row is
    returned.

    The tables for general configuration and Executor configuration assume that
    there are no duplicates as information is intended to be shared and linked
    to by multiple Tasks.

    This function ASSUMES the table EXISTS. Perform creation and necessary
    existence checks before using it.

    Args:
        con (sqlite3.Connection): Database connection.

        table_name (str): The table to add a new row with `entry` values to.

        entry (Dict[str, Any]): A dictionary of entries in the format of
            {COLUMN: ENTRY}. These are assumed to match the columns of the table.

    Returns:
        row_id (int): The row id of the newly added entry or the last entry
            which matches the provided values.
    """
    match_strs: List[str] = [f"{key} LIKE '{val}'" for key, val in entry.items()]
    total_match: str = " AND ".join(s for s in match_strs)

    res: sqlite3.Cursor
    with con:
        res = con.execute(f"SELECT id FROM {table_name} WHERE {total_match}")
        if ids := res.fetchall():
            logging.debug(
                f"_{table_name}_table_entry: Rows matching {total_match}: {ids}"
            )
            return ids[-1][0]
        ins_str: str = "".join(f":{x}, " for x in entry.keys())[:-2]
        res = con.execute(
            f"INSERT INTO {table_name} ({','.join(entry.keys())}) VALUES ({ins_str})",
            entry,
        )
        res = con.execute(f"SELECT id FROM {table_name} WHERE {total_match}")
        new_id: int = res.fetchone()[-1]
        logging.debug(
            f"_{table_name}_table_entry: No matching rows - adding new row: {new_id}"
        )
    return new_id


def _select_from_db(
    con: sqlite3.Connection, table_name: str, col_name: str, condition: Dict[str, str]
) -> Optional[Any]:
    sql: str
    if condition:
        param, val = next(iter(condition.items()))
        sql = f"SELECT {col_name} FROM {table_name} WHERE {param} = {val}"
    else:
        sql = f"SELECT {col_name} FROM {table_name}"
    with con:
        res: sqlite3.Cursor = con.execute(sql)
        entries: List[Any] = res.fetchall()
    return entries[-1][0] if entries else None
