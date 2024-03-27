"""Classes and utilities for communication between Executors and subprocesses.

Communicators manage message passing and parsing between subprocesses. They
maintain a limited public interface of "read" and "write" operations. Behind
this interface the methods of communication vary from serialization across
pipes to Unix sockets, etc. All communicators pass a single object called a
"Message" which contains an arbitrary "contents" field as well as an optional
"signal" field.


Classes:

"""

__all__ = [
    "Party",
    "Message",
    "Communicator",
    "PipeCommunicator",
    "LUTE_SIGNALS",
    "SocketCommunicator",
]
__author__ = "Gabriel Dorlhiac"

import logging
import os
import pickle
import select
import socket
import subprocess
import sys
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Set

import _io
from typing_extensions import Self

LUTE_SIGNALS: Set[str] = {
    "NO_PICKLE_MODE",
    "TASK_STARTED",
    "TASK_FAILED",
    "TASK_STOPPED",
    "TASK_DONE",
    "TASK_CANCELLED",
    "TASK_RESULT",
}

if __debug__:
    warnings.simplefilter("default")
    os.environ["PYTHONWARNINGS"] = "default"
    logging.basicConfig(level=logging.DEBUG)
    logging.captureWarnings(True)
else:
    logging.basicConfig(level=logging.INFO)
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"

logger: logging.Logger = logging.getLogger(__name__)


class Party(Enum):
    """Identifier for which party (side/end) is using a communicator.

    For some types of communication streams there may be different interfaces
    depending on which side of the communicator you are on. This enum is used
    by the communicator to determine which interface to use.
    """

    TASK = 0
    """
    The Task (client) side.
    """
    EXECUTOR = 1
    """
    The Executor (server) side.
    """


@dataclass
class Message:
    contents: Optional[Any] = None
    signal: Optional[str] = None


class Communicator(ABC):
    def __init__(self, party: Party = Party.TASK, use_pickle: bool = True) -> None:
        """Abstract Base Class for IPC Communicator objects.

        Args:
            party (Party): Which object (side/process) the Communicator is
                managing IPC for. I.e., is this the "Task" or "Executor" side.
            use_pickle (bool): Whether to serialize data using pickle prior to
                sending it.
        """
        self._party = party
        self._use_pickle = use_pickle
        self.desc = "Communicator abstract base class."

    @abstractmethod
    def read(self, proc: subprocess.Popen) -> Message:
        """Method for reading data through the communication mechanism."""
        ...

    @abstractmethod
    def write(self, msg: Message) -> None:
        """Method for sending data through the communication mechanism."""
        ...

    def __str__(self):
        name: str = str(type(self)).split("'")[1].split(".")[-1]
        return f"{name}: {self.desc}"

    def __repr__(self):
        return self.__str__()

    def __enter__(self) -> Self:
        return self

    def __exit__(self) -> None: ...

    def stage_communicator(self):
        """Alternative method for staging outside of context manager."""
        self.__enter__()

    def clear_communicator(self):
        """Alternative exit method outside of context manager."""
        self.__exit__()


class PipeCommunicator(Communicator):
    """Provides communication through pipes over stderr/stdout.

    The implementation of this communicator has reading and writing ocurring
    on stderr and stdout. In general the `Task` will be writing while the
    `Executor` will be reading. `stderr` is used for sending signals.
    """

    def __init__(self, party: Party = Party.TASK, use_pickle: bool = True) -> None:
        """IPC through pipes.

        Arbitrary objects may be transmitted using pickle to serialize the data.
        If pickle is not used

        Args:
            party (Party): Which object (side/process) the Communicator is
                managing IPC for. I.e., is this the "Task" or "Executor" side.
            use_pickle (bool): Whether to serialize data using Pickle prior to
                sending it. If False, data is assumed to be text whi
        """
        super().__init__(party=party, use_pickle=use_pickle)
        self.desc = "Communicates through stderr and stdout using pickle."

    def read(self, proc: subprocess.Popen) -> Message:
        """Read from stdout and stderr.

        Args:
            proc (subprocess.Popen): The process to read from.

        Returns:
            msg (Message): The message read, containing contents and signal.
        """
        signal: Optional[str]
        contents: Optional[str]
        raw_signal: bytes = proc.stderr.read()
        raw_contents: bytes = proc.stdout.read()
        if raw_signal is not None:
            signal = raw_signal.decode()
        else:
            signal = raw_signal
        if raw_contents:
            if self._use_pickle:
                try:
                    contents = pickle.loads(raw_contents)
                except pickle.UnpicklingError as err:
                    logger.debug("PipeCommunicator (Executor) - Set _use_pickle=False")
                    self._use_pickle = False
                    contents = self._safe_unpickle_decode(raw_contents)
            else:
                try:
                    contents = raw_contents.decode()
                except UnicodeDecodeError as err:
                    logger.debug("PipeCommunicator (Executor) - Set _use_pickle=True")
                    #                    self._use_pickle = True
                    contents = self._safe_unpickle_decode(raw_contents)
        else:
            contents = None

        if signal and signal not in LUTE_SIGNALS:
            # Some tasks write on stderr
            # If the signal channel has "non-signal" info, add it to
            # contents
            if not contents:
                contents = f"({signal})"
            else:
                contents = f"{contents} ({signal})"
            signal = None

        return Message(contents=contents, signal=signal)

    def _safe_unpickle_decode(self, maybe_mixed: bytes) -> Optional[str]:
        """This method is used to unpickle and/or decode a bytes object.

        It attempts to handle cases where contents can be mixed, i.e., part of
        the message must be decoded and the other part unpickled. It handles
        only two-way splits. If there are more complex arrangements such as:
        <pickled>:<unpickled>:<pickled> etc, it will give up.

        The simpler two way splits are unlikely to occur in normal usage. They
        may arise when debugging if, e.g., `print` statements are mixed with the
        usage of the `_report_to_executor` method.

        Note that this method works because ONLY text data is assumed to be
        sent via the pipes. The method needs to be revised to handle non-text
        data if the `Task` is modified to also send that via PipeCommunicator.
        The use of pickle is supported to provide for this option if it is
        necessary. It may be deprecated in the future.

        Be careful when making changes. This method has seemingly redundant
        checks because unpickling will not throw an error if a full object can
        be retrieved. That is, the library will ignore extraneous bytes. This
        method attempts to retrieve that information if the pickled data comes
        first in the stream.

        Args:
            maybe_mixed (bytes): A bytes object which could require unpickling,
                decoding, or both.

        Returns:
            contents (Optional[str]): The unpickled/decoded contents if possible.
                Otherwise, None.
        """
        contents: Optional[str]
        try:
            contents = pickle.loads(maybe_mixed)
            repickled: bytes = pickle.dumps(contents)
            if len(repickled) < len(maybe_mixed):
                # Successful unpickling, but pickle stops even if there are more bytes
                additional_data: str = maybe_mixed[len(repickled) :].decode()
                contents = f"{contents}{additional_data}"
        except pickle.UnpicklingError as err:
            try:
                contents = maybe_mixed.decode()
            except UnicodeDecodeError as err2:
                try:
                    contents = maybe_mixed[: err2.start].decode()
                    contents = f"{contents}{pickle.loads(maybe_mixed[err2.start:])}"
                except Exception as err3:
                    logger.debug(
                        f"PipeCommunicator unable to decode/parse data! {err3}"
                    )
                    contents = None
        except UnicodeDecodeError as err3:
            missing_bytes: int = len(maybe_mixed) - len(repickled)
            logger.debug(
                f"PipeCommunicator has truncated message. Unable to retrieve {missing_bytes} bytes."
            )
        return contents

    def write(self, msg: Message) -> None:
        """Write to stdout and stderr.

         The signal component is sent to `stderr` while the contents of the
         Message are sent to `stdout`.

        Args:
            msg (Message): The Message to send.
        """
        if self._use_pickle:
            signal: bytes
            if msg.signal:
                signal = msg.signal.encode()
            else:
                signal = b""

            contents: bytes = pickle.dumps(msg.contents)

            sys.stderr.buffer.write(signal)
            sys.stdout.buffer.write(contents)

            sys.stderr.buffer.flush()
            sys.stdout.buffer.flush()
        else:
            raw_signal: str
            if msg.signal:
                raw_signal = msg.signal
            else:
                raw_signal = ""

            raw_contents: str
            if isinstance(msg.contents, str):
                raw_contents = msg.contents
            elif msg.contents is None:
                raw_contents = ""
            else:
                raise ValueError(
                    f"Cannot send msg contents of type: {type(msg.contents)} when not using pickle!"
                )
            sys.stderr.write(raw_signal)
            sys.stdout.write(raw_contents)


class SocketCommunicator(Communicator):
    """Provides communication over Unix sockets.

    The path to the Unix socket is defined by the environment variable:
                      `LUTE_SOCKET=/path/to/socket`
    This class assumes proper permissions and that this above environment
    variable has been defined. The `Task` is configured as what would commonly
    be referred to as the `client`, while the `Executor` is configured as the
    server. The Executor continuosly monitors for connections and appends any
    Messages that are received to a queue. Read requests retrieve Messages from
    the queue. Task-side Communicators are fleeting so they open a connection,
    send data, and immediately close and clean up.
    """

    READ_TIMEOUT: float = 0.01
    """
    Maximum time to wait to retrieve data.
    """

    def __init__(self, party: Party = Party.TASK, use_pickle: bool = True) -> None:
        """IPC over a Unix socket.

        Unlike with the PipeCommunicator, pickle is always used to send data
        through the socket.

        Args:
            party (Party): Which object (side/process) the Communicator is
                managing IPC for. I.e., is this the "Task" or "Executor" side.

            use_pickle (bool): Whether to use pickle. Always True currently,
                passing False does not change behaviour.
        """
        super().__init__(party=party, use_pickle=use_pickle)
        self.desc: str = "Communicates through a Unix socket."

        self._data_socket: socket.socket = self._create_socket()
        self._data_socket.setblocking(0)

    def read(self, proc: subprocess.Popen) -> Message:
        """Read data from a socket.

        Socket(s) are continuously monitored, and read from when new data is
        available.

        Args:
            proc (subprocess.Popen): The process to read from. Provided for
                compatibility with other Communicator subtypes. Is ignored.

        Returns:
             msg (Message): The message read, containing contents and signal.
        """
        has_data, _, has_error = select.select(
            [self._data_socket],
            [],
            [self._data_socket],
            SocketCommunicator.READ_TIMEOUT,
        )

        msg: Message
        if has_data:
            connection, _ = has_data[0].accept()
            full_data: bytes = b""
            while True:
                data: bytes = connection.recv(1024)
                if data:
                    full_data += data
                else:
                    break
            msg = pickle.loads(full_data) if full_data else Message()
            connection.close()
        else:
            msg = Message()

        return msg

    def write(self, msg: Message) -> None:
        """Send a single Message.

        The entire Message (signal and contents) is serialized and sent through
        a connection over Unix socket.

        Args:
            msg (Message): The Message to send.
        """
        self._write_socket(msg)

    def _create_socket(self) -> socket.socket:
        """Returns a socket object.

        Returns:
            data_socket (socket.socket): Unix socket object.
        """
        socket_path: str
        try:
            socket_path = os.environ["LUTE_SOCKET"]
        except KeyError as err:
            import uuid
            import tempfile

            # Define a path,up and add to environment
            # Executor-side always created first, Task will use the same one
            socket_path = f"{tempfile.gettempdir()}/lute_{uuid.uuid4().hex}.sock"
            os.environ["LUTE_SOCKET"] = socket_path
            logger.debug(f"SocketCommunicator defines socket_path: {socket_path}")

        data_socket: socket.socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)

        if self._party == Party.EXECUTOR:
            if os.path.exists(socket_path):
                os.unlink(socket_path)
            data_socket.bind(socket_path)
            data_socket.listen(1)
        elif self._party == Party.TASK:
            data_socket.connect(socket_path)

        return data_socket

    def _write_socket(self, msg: Message) -> None:
        """Sends data over a socket from the 'client' (Task) side.

        Communicator objects on the Task-side are fleeting, so a socket is
        opened, data is sent, and then the connection and socket are cleaned up.
        """
        self._data_socket.sendall(pickle.dumps(msg))

        self._clean_up()

    def _clean_up(self) -> None:
        """Clean up connections."""
        # Check the object exists in case the Communicator is cleaned up before
        # opening any connections
        if hasattr(self, "_data_socket"):
            socket_path: str = self._data_socket.getsockname()
            self._data_socket.close()

            if self._party == Party.EXECUTOR:
                os.unlink(socket_path)

    @property
    def socket_path(self) -> str:
        socket_path: str = self._data_socket.getsockname()
        return socket_path

    def __exit__(self):
        self._clean_up()
