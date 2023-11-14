"""Base classes for implementing analysis tasks.

Classes:
    Task: Abstract base class from which all analysis tasks are derived.

    TaskResult: Output of a specific analysis task.

    TaskStatus: Enumeration of possible Task statuses (running, pending, failed,
        etc.).

    BinaryTask: Class to run a third-party executable binary as a `Task`.
"""

__all__ = ["Task", "TaskResult", "TaskStatus", "BinaryTask"]
__author__ = "Gabriel Dorlhiac"

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Dict
from enum import Enum
import os

from ..io.config import TaskParameters
from ..execution.ipc import Message, PipeCommunicator


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
    """

    task_name: str
    task_status: TaskStatus
    summary: str
    payload: Any


class Task(ABC):
    """Abstract base class for analysis tasks.

    Attributes:
        name (str): The name of the Task.
    """

    def __init__(self, *, params: TaskParameters) -> None:
        """Initialize a Task.

        Args:
            params (TaskParameters): Parameters needed to properly configure
                the analysis task. These are NOT related to execution parameters
                (number of cores, etc), except, potentially, in case of binary
                executable sub-classes.
        """
        self.name: str = str(type(self)).split("'")[1].split(".")[-1]
        self._result: TaskResult = TaskResult(
            task_name=self.name,
            task_status=TaskStatus.PENDING,
            summary="PENDING",
            payload="",
        )
        self._task_parameters = params
        self._communicator = PipeCommunicator()

    def run(self) -> None:
        """Calls the analysis routines and any pre/post task functions.

        This method is part of the public API and should not need to be modified
        in any subclasses.
        """
        self._signal_start()
        self._pre_run()
        self._run()
        self._post_run()
        self._signal_result()

    @abstractmethod
    def _run(self) -> None:
        """Actual analysis to run. Overridden by subclasses.

        Separating the calling API from the implementation allows `run` to
        have pre and post task functionality embedded easily into a single
        function call.
        """
        ...

    def _pre_run(self) -> None:
        """Code to run BEFORE the main analysis takes place.

        This function may, or may not, be employed by subclasses.
        """
        ...

    def _post_run(self) -> None:
        """Code to run AFTER the main analysis takes place.

        This function may, or may not, be employed by subclasses.
        """
        ...

    @property
    def result(self) -> TaskResult:
        """TaskResult: Read-only Task Result information."""
        return self._result

    def __call__(self) -> None:
        self.run()

    def _signal_start(self) -> None:
        """Send the signal that the Task will begin shortly."""
        start_msg: Message = Message(
            contents=self._task_parameters, signal="TASK_STARTED"
        )
        self._result.task_status = TaskStatus.RUNNING
        self._communicator.write(start_msg)

    def _signal_result(self) -> None:
        """Send the signal that results are ready along with the results."""
        signal: str = "TASK_RESULT"
        results_msg: Message = Message(contents=self.result, signal=signal)
        self._communicator.write(results_msg)
        time.sleep(0.1)


class BinaryTask(Task):
    """A `Task` interface to analysis with binary executables."""

    def __init__(self, *, params: TaskParameters) -> None:
        """Initialize a Task.

        Args:
            params (TaskParameters): Parameters needed to properly configure
                the analysis task. `Task`s of this type MUST include the name
                of a binary to run and any arguments which should be passed to
                it (as would be done via command line). In addition, a special
                parameter `flag_names` (Dict[str, str]) should be included.
                This is a dictionary of friendly names and their corresponding
                command-line flags. E.g. a binary executable which takes a
                number of cores flag may have a dictionary entrythat looks
                like:
                    * flag_names = { "ncores" : "-n" }
                flag_names must match the corresponding parameter names listed
                in the rest of the TaskParameters object.
        """
        super().__init__(params=params)
        self._cmd = self._task_parameters.executable
        self._args_list: List[str] = []

    def _pre_run(self):
        """Prepare the list of flags and arguments to be executed."""
        super()._pre_run()
        for friendly_name, flag in self._task_parameters.flag_names.items():
            self._args_list.append(flag)
            self._args_list.append(getattr(self._task_parameters, friendly_name))

    def _run(self):
        """Execute the new program by replacing the current process."""
        os.execvp(file=self._cmd, args=self._args_list)
