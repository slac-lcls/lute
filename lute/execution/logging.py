"""Custom loggers for Tasks."""

import logging

from lute.execution.ipc import SocketCommunicator, Message

__all__ = ["get_logger"]
__author__ = "Gabriel Dorlhiac"

STD_PYTHON_LOG_FORMAT: str = "%(levelname)s:%(name)s:%(message)s"
LUTE_TASK_LOG_FORMAT: str = f"TASK_LOG -- %(levelname)s:%(name)s: %(message)s"


class SocketCommunicatorHandler(logging.Handler):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._communicator: SocketCommunicator = SocketCommunicator()
        self._communicator.delayed_setup()

    def emit(self, record: logging.LogRecord):
        formatted_message: str = self.format(record)
        msg: Message = Message(contents=formatted_message, signal="TASK_LOG")
        self._communicator.write(msg)


def get_logger(name: str) -> logging.Logger:
    for other_name, other_logger in logging.root.manager.loggerDict.items():
        if "matplotlib" in other_name and not isinstance(
            other_logger, logging.PlaceHolder
        ):
            other_logger.disabled = True
        elif "numpy" in other_name and not isinstance(
            other_logger, logging.PlaceHolder
        ):
            other_logger.setLevel(logging.CRITICAL)
    logger: logging.Logger = logging.getLogger(name)
    logger.propagate = False
    handler: SocketCommunicatorHandler = SocketCommunicatorHandler()
    formatter: logging.Formatter = logging.Formatter(LUTE_TASK_LOG_FORMAT)
    handler.setFormatter(formatter)
    logger.handlers = []
    logger.addHandler(handler)
    if __debug__:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    return logger
