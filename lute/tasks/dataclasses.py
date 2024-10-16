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
from typing import Any, List, Dict, Optional, Union
from dataclasses import dataclass
from enum import Enum

from lute.io.models.base import TaskParameters


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

        summary (str): Short message/summary associated with the result.

        payload (Any): Actual result. May be data in any format.

        impl_schemas (Optional[str]): A string listing `Task` schemas implemented
            by the associated `Task`. Schemas define the category and expected
            output of the `Task`. An individual task may implement/conform to
            multiple schemas. Multiple schemas are separated by ';', e.g.
                * impl_schemas = "schema1;schema2"
    """

    task_name: str
    task_status: TaskStatus
    summary: str
    payload: Any
    impl_schemas: Optional[str] = None


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
    figures: Union[pn.Tabs, hv.Image, plt.Figure, bytes]

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
    poll_interval: float
    communicator_desc: List[str]
