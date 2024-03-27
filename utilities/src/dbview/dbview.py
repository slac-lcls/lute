import os
import sqlite3
import argparse
import importlib.util
from typing import List, Tuple, Any
from types import ModuleType

from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Footer, Header, DataTable, TabbedContent

spec: importlib.machinery.ModuleSpec = importlib.util.spec_from_file_location(
    "_sqlite", f"{os.environ.get('LUTE_BASE', '')}/lute/io/_sqlite.py"
)
_sqlite: ModuleType = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_sqlite)

parser: argparse.ArgumentParser = argparse.ArgumentParser(
    prog="DBView",
    description="LUTE database inspection utility. Read-Only.",
    epilog="Refer to https://github.com/slac-lcls/lute for more information.",
)
parser.add_argument("-p", "--path", type=str, help="Path to SQLite database.")


class DBView(App):
    """DBView - A LUTE database inspector.

    This TUI application facilitates inspection of tables recorded in a LUTE
    database. Databases are opened in read-only mode - no modifications of data
    or records are permitted through this utility.
    """

    BINDINGS = [
        ("d", "toggle_dark", "Toggle dark mode"),
        ("q", "quit", "Quit"),
    ]

    def __init__(self, *args, dbpath: str, **kwargs) -> None:
        """Open a view to a LUTE database.

        Args:
            dbpath (str): Path to a SQLite LUTE database.
        """
        super().__init__(*args, **kwargs)
        self._dbpath: str = dbpath
        self._con: sqlite3.Connection = sqlite3.Connection(self._dbpath)
        self._tables: List[str] = _sqlite._get_tables(self._con)

    def compose(self) -> ComposeResult:
        """Compose our UI."""
        yield Header()
        with TabbedContent(*self._tables):
            for table in self._tables:
                with VerticalScroll(id=f"view_{table}"):
                    yield self.pull_table_data(DataTable(id=f"data_{table}"))
        yield Footer()

    def pull_table_data(self, table: DataTable) -> DataTable:
        """Query database for all rows in a table and add to display."""
        table_name: str = table.id[5:]
        table.add_columns(*_sqlite._get_table_cols(self._con, table_name))
        rows: List[Tuple[Any, ...]] = _sqlite._get_all_rows_for_table(
            self._con, table_name
        )
        for row in rows:
            table.add_row(*row)

        return table

    def action_toggle_dark(self) -> None:
        self.dark = not self.dark


if __name__ == "__main__":
    args: argparse.Namespace = parser.parse_args()
    app: DBView = DBView(dbpath=args.path)
    app.run()
