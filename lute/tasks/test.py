"""Basic test Tasks for testing functionality.

Classes:
    Test(Task): Simplest test Task - runs a 10 iteration loop and returns a
        result.

    TestSocket(Task): Test Task which sends larger data to test socket IPC.

    TestWriteOutput(Task): Test Task which writes an output file.

    TestReadOutput(Task): Test Task which reads in a file. Can be used to test
        database access.
"""

__all__ = ["Test", "TestSocket", "TestWriteOutput", "TestReadOutput"]
__author__ = "Gabriel Dorlhiac"

import time

import numpy as np

from lute.tasks.task import *
from lute.io.models.base import *
from lute.execution.ipc import Message


class Test(Task):
    """Simple test Task to ensure subprocess and pipe-based IPC work."""

    def __init__(self, *, params: TaskParameters) -> None:
        super().__init__(params=params)

    def _run(self) -> None:
        for i in range(10):
            time.sleep(1)
            msg: Message = Message(contents=f"Test message {i}")
            self._report_to_executor(msg)
        if self._task_parameters.throw_error:
            raise RuntimeError("Testing Error!")

    def _post_run(self) -> None:
        self._result.summary = "Test Finished."
        self._result.task_status = TaskStatus.COMPLETED
        time.sleep(0.1)


class TestSocket(Task):
    """Simple test Task to ensure basic IPC over Unix sockets works."""

    def __init__(self, *, params: TaskParameters) -> None:
        super().__init__(params=params)

    def _run(self) -> None:
        for i in range(self._task_parameters.num_arrays):
            msg: Message = Message(contents=f"Sending array {i}")
            self._report_to_executor(msg)
            time.sleep(0.05)
            msg: Message = Message(
                contents=np.random.rand(self._task_parameters.array_size)
            )
            self._report_to_executor(msg)

    def _post_run(self) -> None:
        super()._post_run()
        self._result.summary = f"Sent {self._task_parameters.num_arrays} arrays"
        self._result.payload = np.random.rand(self._task_parameters.array_size)
        self._result.task_status = TaskStatus.COMPLETED


class TestWriteOutput(Task):
    """Simple test Task to write output other Tasks depend on."""

    def __init__(self, *, params: TaskParameters) -> None:
        super().__init__(params=params)

    def _run(self) -> None:
        for i in range(self._task_parameters.num_vals):
            # Doing some calculations...
            time.sleep(0.05)
            if i % 10 == 0:
                msg: Message = Message(contents=f"Processed {i+1} values!")
                self._report_to_executor(msg)

    def _post_run(self) -> None:
        super()._post_run()
        work_dir: str = self._task_parameters.lute_config.work_dir
        out_file: str = f"{work_dir}/{self._task_parameters.outfile_name}"
        array: np.ndarray = np.random.rand(self._task_parameters.num_vals)
        np.savetxt(out_file, array, delimiter=",")
        self._result.summary = "Completed task successfully."
        self._result.payload = out_file
        self._result.task_status = TaskStatus.COMPLETED


class TestReadOutput(Task):
    """Simple test Task to read in output from the test Task above.

    Its pydantic model relies on a database access to retrieve the output file.
    """

    def __init__(self, *, params: TaskParameters) -> None:
        super().__init__(params=params)

    def _run(self) -> None:
        array: np.ndarray = np.loadtxt(self._task_parameters.in_file, delimiter=",")
        self._report_to_executor(msg=Message(contents="Successfully loaded data!"))
        for i in range(5):
            time.sleep(1)

    def _post_run(self) -> None:
        super()._post_run()
        self._result.summary = "Was able to load data."
        self._result.payload = "This Task produces no output."
        self._result.task_status = TaskStatus.COMPLETED
