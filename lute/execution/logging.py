"""Custom loggers for Tasks."""

import logging

from lute.execution.ipc import PipeCommunicator, Message

__all__ = ["PipeCommunicatorHandler", "STD_PYTHON_LOG_FORMAT"]
__author__ = "Gabriel Dorlhiac"

STD_PYTHON_LOG_FORMAT: str = "%(levelname)s:%(name)s:%(message)s"


class PipeCommunicatorHandler(logging.Handler):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._communicator: PipeCommunicator = PipeCommunicator()

    def emit(self, record: logging.LogRecord):
        formatted_message: str = self.format(record)
        msg: Message = Message(contents=formatted_message, signal="TASK_LOG")
        self._communicator.write(msg)
