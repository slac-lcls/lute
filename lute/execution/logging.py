"""Custom loggers for Tasks.

Classes:
    SocketCommunicatorHandler(logging.Handler): Logging handler which passes
        messages via LUTE SocketCommunicator objects.

Functions:
    get_logger(name: str) -> logging.Logger: Grab a standard LUTE logger and
        reduce logging from libraries which output many messages.
"""

import logging

from lute.execution.ipc import SocketCommunicator, Message

__all__ = ["get_logger"]
__author__ = "Gabriel Dorlhiac"

STD_PYTHON_LOG_FORMAT: str = "%(levelname)s:%(name)s:%(message)s"
"""Default Python logging formatter specification."""

LUTE_TASK_LOG_FORMAT: str = f"TASK_LOG -- %(levelname)s:%(name)s: %(message)s"
"""Format specification for the formatter used by the standard LUTE logger."""


class SocketCommunicatorHandler(logging.Handler):
    """Logging handler which passes messages via SocketCommunicator objects."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._communicator: SocketCommunicator = SocketCommunicator()
        self._communicator.delayed_setup()

    def emit(self, record: logging.LogRecord):
        formatted_message: str = self.format(record)
        msg: Message = Message(contents=formatted_message, signal="TASK_LOG")
        self._communicator.write(msg)


def get_logger(name: str) -> logging.Logger:
    """Retrieve a logger with LUTE handler and set log levels of other loggers.

    This function returns a logger with correct log level set by the debug flag.
    In addition, it silences (or reduces) logs from commonly imported libraries
    which produce many messages that make log files hard to parse. This is
    particularly useful when running in debug mode.

    Args:
        name (str): Name of the logger.

    Returns:
        logger (logging.Logger): Custom logger.
    """
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
