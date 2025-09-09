"""SQLite utility functions that are independent of API version."""

from typing import List

__all__: List[str] = []
__author__ = "Gabriel Dorlhiac"

import sqlite3
from typing import Any, Dict, List, Optional, Tuple


class DatabaseError(Exception):
    """General LUTE database error."""

    ...


def does_table_exist(con: sqlite3.Connection, table_name: str) -> bool:
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


def get_tables(con: sqlite3.Connection) -> List[str]:
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


def get_table_cols(con: sqlite3.Connection, table_name: str) -> Dict[str, str]:
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


def get_all_rows_for_table(
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


def compare_cols(
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
