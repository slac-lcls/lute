"""Classes for describing Task state and results.

Classes:
    TaskResult: Output of a specific analysis task.

    TaskStatus: Enumeration of possible Task statuses (running, pending, failed,
        etc.).

    DescribedAnalysis: Executor's description of a `Task` run (results,
        parameters, env).
"""

from __future__ import annotations

__all__ = ["TaskResult", "TaskStatus", "DescribedAnalysis", "ElogSummaryPlots"]
__author__ = "Gabriel Dorlhiac"

import io
from typing import Any, List, Dict, Optional, Union, TYPE_CHECKING
from dataclasses import dataclass
from enum import Enum

if TYPE_CHECKING:
    from lute.io.models.base import TaskParameters
else:
    from lute.io.parameters import TaskParameters
from lute.io.parameters import RowIds


@dataclass
class TaskParametersDBReference:
    """Contains information about how to reconstruct a TaskParameters object.

    Attributes:
        db_dir (str): The path to the database containing the TaskParameters
            schema definition.

        row_ids (RowIds): The ids of the rows in the various tables required for
            reconstructing the TaskParameters object.
    """

    db_dir: str
    row_ids: RowIds


class TaskStatus(Enum):
    """Possible Task statuses."""

    PENDING = 0
    """
    Task has yet to run. Is Queued, or waiting for prior tasks.
    """
    RUNNING = 1
    """
    Task is in the process of execution.
    """
    COMPLETED = 2
    """
    Task has completed without fatal errors.
    """
    FAILED = 3
    """
    Task encountered a fatal error.
    """
    STOPPED = 4
    """
    Task was, potentially temporarily, stopped/suspended.
    """
    CANCELLED = 5
    """
    Task was cancelled prior to completion or failure.
    """
    TIMEDOUT = 6
    """
    Task did not reach completion due to timeout.
    """


@dataclass
class TaskResult:
    """Class for storing the result of a Task's execution with metadata.

    Attributes:
        task_name (str): Name of the associated task which produced it.

        task_status (TaskStatus): Status of associated task.

        summary (Any): Short (usually text message) summary associated with the result.

        payload (Any): Actual result. May be data in any format.

        impl_schemas (Optional[str]): A string listing `Task` schemas implemented
            by the associated `Task`. Schemas define the category and expected
            output of the `Task`. An individual task may implement/conform to
            multiple schemas. Multiple schemas are separated by ';', e.g.
                * impl_schemas = "schema1;schema2"
    """

    task_name: str
    task_status: TaskStatus
    summary: Any
    payload: Any
    impl_schemas: Optional[str] = None


class BaseSchema(int, Enum):
    NONE = 0
    HDF5 = 1


class VersionSpecifier(int, Enum):
    NONE = 0
    """Version information lacking or not relevant."""
    LUTE_VERSION = 1
    """For a first-party Task, indicates this tracks LUTE's version only."""
    GIT_SHA = 2
    """Specifies version as a specific commit hash."""
    GIT_DIFF = 4
    """Specifies version as a diff. Must be used combined with a GIT_SHA to diff."""


@dataclass
class ElogSummaryPlots:
    """Holds a graphical summary intended for display in the eLog.

    Converts figures to a byte stream of HTML data to be written out, so the
    eLog can properly display them.

    Attributes:
        display_name (str): This represents both a path and how the result will be
            displayed in the eLog. Can include "/" characters. E.g.
            `display_name = "scans/my_motor_scan"` will have plots shown
            on a "my_motor_scan" page, under a "scans" tab. This format mirrors
            how the file is stored on disk as well.

        figures (pn.Tabs, hv.Image, plt.Figure, bytes): The figures to be
            displayed. Except panel/holoviews (bokeh backend) and matplotlib
            plots as well as a raw series of bytes for the HTML file. Figures from
            the plotting libraries will be converted to an HTML byte stream
            automatically.
    """

    display_name: str
    figures: Union[pn.Tabs, hv.Image, plt.Figure, bytes]  # type: ignore # noqa: F821

    def __post_init__(self) -> None:
        self._setup_figures()

    def _setup_figures(self) -> None:
        """Convert figures to an HTML file in a byte stream."""

        if hasattr(self.figures, "save"):
            f: io.BytesIO = io.BytesIO()
            self.figures.save(f)
            f.seek(0)
            self.figures = f.read()


@dataclass
class DescribedAnalysis:
    """Complete analysis description. Held by an Executor."""

    task_result: TaskResult
    task_parameters: Optional[TaskParameters]
    task_env: Dict[str, str]
    executor_name: str
    poll_interval: float
    communicator_desc: List[str]
