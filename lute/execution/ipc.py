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
    "SocketCommunicator",  # Points to one of two classes below. See bottom
]
__author__ = "Gabriel Dorlhiac"

import logging
import os
import pickle
import select
import socket
import subprocess
import sys
import time
import threading
import warnings
import queue
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Set, List, Literal, Union, Tuple

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
                except (pickle.UnpicklingError, ValueError, EOFError) as err:
                    logger.debug("PipeCommunicator (Executor) - Set _use_pickle=False")
                    self._use_pickle = False
                    contents = self._safe_unpickle_decode(raw_contents)
            else:
                try:
                    contents = raw_contents.decode()
                except UnicodeDecodeError as err:
                    logger.debug("PipeCommunicator (Executor) - Set _use_pickle=True")
                    self._use_pickle = True
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
                try:
                    additional_data: str = maybe_mixed[len(repickled) :].decode()
                    contents = f"{contents}{additional_data}"
                except UnicodeDecodeError:
                    # Can't decode the bytes left by pickle, so they are lost
                    missing_bytes: int = len(maybe_mixed) - len(repickled)
                    logger.debug(
                        f"PipeCommunicator has truncated message. Unable to retrieve {missing_bytes} bytes."
                    )
        except (pickle.UnpicklingError, ValueError, EOFError) as err:
            # Pickle may also throw a ValueError, e.g. this bytes: b"Found! \n"
            # Pickle may also throw an EOFError, eg. this bytes: b"F0\n"
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


class RawSocketCommunicator(Communicator):
    """Provides communication over raw Unix or TCP sockets.

    Whether to use TCP or Unix sockets is controlled by the environment:
                           `LUTE_USE_TCP=1`
    If defined, TCP sockets will be used, otherwise Unix sockets will be used.

    Regardless of socket type, the environment variable
                      `LUTE_EXECUTOR_HOST=<hostname>`
    will be defined by the Executor-side Communicator.


    For TCP sockets:
    The Executor-side Communicator should be run first and will bind to all
    interfaces on the port determined by the environment variable:
                            `LUTE_PORT=###`
    If no port is defined, a port scan will be performed and the Executor-side
    Communicator will bind the first one available from a random selection. It
    will then define the environment variable so the Task-side can pick it up.

    For Unix sockets:
    The path to the Unix socket is defined by the environment variable:
                      `LUTE_SOCKET=/path/to/socket`
    This class assumes proper permissions and that this above environment
    variable has been defined. The `Task` is configured as what would commonly
    be referred to as the `client`, while the `Executor` is configured as the
    server.

    If the Task process is run on a different machine than the Executor, the
    Task-side Communicator will open a ssh-tunnel to forward traffic from a local
    Unix socket to the Executor Unix socket. Opening of the tunnel relies on the
    environment variable:
                      `LUTE_EXECUTOR_HOST=<hostname>`
    to determine the Executor's host. This variable should be defined by the
    Executor and passed to the Task process automatically, but it can also be
    defined manually if launching the Task process separately. The Task will use
    the local socket `<LUTE_SOCKET>.task{##}`. Multiple local sockets may be
    created. Currently, it is assumed that the user is identical on both the Task
    machine and Executor machine.
    """

    ACCEPT_TIMEOUT: float = 5
    """
    Maximum time to wait to accept connections. Used by Executor-side.
    """

    def __init__(self, party: Party = Party.TASK, use_pickle: bool = True) -> None:
        """IPC over a TCP or Unix socket.

        Unlike with the PipeCommunicator, pickle is always used to send data
        through the socket.

        Args:
            party (Party): Which object (side/process) the Communicator is
                managing IPC for. I.e., is this the "Task" or "Executor" side.

            use_pickle (bool): Whether to use pickle. Always True currently,
                passing False does not change behaviour.
        """
        super().__init__(party=party, use_pickle=use_pickle)
        self.desc: str = "Communicates through a TCP or Unix socket."

        self._data_socket: socket.socket = self._create_socket()
        # self._data_socket.setblocking(0)

        if self._party == Party.EXECUTOR:
            # Executor created first so we can define the hostname env variable
            os.environ["LUTE_EXECUTOR_HOST"] = socket.gethostname()
            # Setup reader thread
            self._reader_thread: threading.Thread = threading.Thread(
                target=self._read_socket
            )
            self._msg_queue: queue.Queue = queue.Queue()
            self._partial_msg: Optional[bytes] = None
            self._stop_thread: bool = False
            self._data_socket.settimeout(RawSocketCommunicator.ACCEPT_TIMEOUT)
            self._reader_thread.start()
        else:
            # Only used by Party.TASK
            self._use_ssh_tunnel: bool = False
            self._ssh_proc: Optional[subprocess.Popen] = None
            self._local_socket_path: Optional[str] = None

    def read(self, proc: subprocess.Popen) -> Message:
        """Return a message from the queue if available.

        Socket(s) are continuously monitored, and read from when new data is
        available.

        Args:
            proc (subprocess.Popen): The process to read from. Provided for
                compatibility with other Communicator subtypes. Is ignored.

        Returns:
             msg (Message): The message read, containing contents and signal.
        """
        msg: Message
        try:
            msg = self._msg_queue.get(timeout=RawSocketCommunicator.ACCEPT_TIMEOUT)
        except queue.Empty:
            msg = Message()

        return msg

    def _read_socket(self) -> None:
        """Read data from a socket.

        Socket(s) are continuously monitored, and read from when new data is
        available.

        Args:
            proc (subprocess.Popen): The process to read from. Provided for
                compatibility with other Communicator subtypes. Is ignored.

        Returns:
             msg (Message): The message read, containing contents and signal.
        """
        while True:
            if self._stop_thread:
                logger.debug("Stopping socket reader thread.")
                break
            connection: socket.socket
            addr: Union[str, Tuple[str, int]]
            try:
                connection, addr = self._data_socket.accept()
            except socket.timeout:
                continue
            full_data: bytes = b""
            while True:
                data: bytes = connection.recv(8192)
                if data:
                    full_data += data
                else:
                    break
            self._unpack_messages(full_data)
            connection.close()

    def _unpack_messages(self, data: bytes) -> None:
        """Unpacks a byte stream into individual messages.

        Messages are encoded in the following format:
                      MSG,<len(msg)>,<msg>,GSM
        The items between <> represent the expected length of the message and
        the message itself while the bytes versions of the literals "MSG", ","
        and "GSM" are used to delineate and bookend the actual data.

        Partial messages (a series of bytes which cannot be converted to a full
        message) are stored for later. An attempt is made to reconstruct the
        message with the next call to this method.

        Args:
            data (bytes): A raw byte stream containing anywhere from a partial
                message to multiple full messages.
        """
        msg: Message
        working_data: bytes
        if self._partial_msg:
            # Concatenate the previous partial message to the beginning
            working_data = self._partial_msg + data
            self._partial_msg = None
        else:
            working_data = data
        while working_data:
            try:
                # Messages are encoded as MSG,<len>,<msg>,GSM
                end = working_data.find(b",GSM")
                msg_parts: List[bytes] = working_data[:end].split(b",")
                if len(msg_parts) != 3:
                    self._partial_msg = working_data
                    break

                cmd: bytes
                nbytes: bytes
                raw_msg: bytes
                cmd, nbytes, raw_msg = msg_parts
                if len(raw_msg) != int(nbytes):
                    self._partial_msg = working_data
                    break

                msg = pickle.loads(raw_msg)
                self._msg_queue.put(msg)
            except pickle.UnpicklingError:
                self._partial_msg = working_data
                break
            if end < len(working_data):
                # Add 4 since end marks the start of ,GSM sequence
                working_data = working_data[end + 4 :]
            else:
                working_data = b""

    def _read_socket_select(self) -> None:
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
            RawSocketCommunicator.ACCEPT_TIMEOUT,
        )

        msg: Message
        if has_data:
            connection, _ = has_data[0].accept()
            full_data: bytes = b""
            while True:
                data: bytes = connection.recv(8192)
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

    def _find_random_port(
        self, min_port: int = 41923, max_port: int = 64324, max_tries: int = 100
    ) -> Optional[int]:
        """Find a random open port to bind to if using TCP."""
        from random import choices

        sock: socket.socket
        ports: List[int] = choices(range(min_port, max_port), k=max_tries)
        for port in ports:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                sock.bind(("", port))
                sock.close()
                del sock
                return port
            except:
                continue
        return None

    def _init_tcp_socket(self) -> socket.socket:
        """Initialize a TCP socket.

        Executor-side code should always be run first. It checks to see if
        the environment variable
                                `LUTE_PORT=###`
        is defined, if so binds it, otherwise find a free port from a selection
        of random ports. If a port search is performed, the `LUTE_PORT` variable
        will be defined so it can be picked up by the the Task-side Communicator.

        In the event that no port can be bound on the Executor-side, or the port
        and hostname information is unavailable to the Task-side, the program
        will exit.

        Returns:
            data_socket (socket.socket): TCP socket object.
        """
        data_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        port: Optional[Union[str, int]] = os.getenv("LUTE_PORT")
        if self._party == Party.EXECUTOR:
            if port is None:
                # If port is None find one
                # Executor code executes first
                port = self._find_random_port()
                if port is None:
                    # Failed to find a port to bind
                    logger.info(
                        "Executor failed to bind a port. "
                        "Try providing a LUTE_PORT directly! Exiting!"
                    )
                    sys.exit(-1)
                # Provide port env var for Task-side
                os.environ["LUTE_PORT"] = str(port)
            data_socket.bind(("", int(port)))
            data_socket.listen()
        else:
            executor_hostname: Optional[str] = os.getenv("LUTE_EXECUTOR_HOST")
            if executor_hostname is None or port is None:
                logger.info(
                    "Task-side does not have host/port information!"
                    " Check environment variables! Exiting!"
                )
                sys.exit(-1)
            data_socket.connect((executor_hostname, int(port)))
        return data_socket

    def _init_unix_socket(self) -> socket.socket:
        """Returns a Unix socket object.

        Executor-side code should always be run first. It checks to see if
        the environment variable
                                `LUTE_SOCKET=XYZ`
        is defined, if so binds it, otherwise it will create a new path and
        define the environment variable for the Task-side to find.

        On the Task (client-side), this method will also open a SSH tunnel to
        forward a local Unix socket to an Executor Unix socket if the Task and
        Executor processes are on different machines.

        The SSH tunnel is opened with `ssh -L <local>:<remote> sleep 2`.
        This method of communication is slightly slower and incurs additional
        overhead - it should only be used as a backup. If communication across
        multiple hosts is required consider using TCP.  The Task will use
        the local socket `<LUTE_SOCKET>.task{##}`. Multiple local sockets may be
        created. It is assumed that the user is identical on both the
        Task machine and Executor machine.

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
            logger.debug(f"UnixSocketCommunicator defines socket_path: {socket_path}")
        data_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        if self._party == Party.EXECUTOR:
            if os.path.exists(socket_path):
                os.unlink(socket_path)
            data_socket.bind(socket_path)
            data_socket.listen()
        elif self._party == Party.TASK:
            hostname: str = socket.gethostname()
            executor_hostname: Optional[str] = os.getenv("LUTE_EXECUTOR_HOST")
            if executor_hostname is None:
                logger.info("Hostname for Executor process not found! Exiting!")
                self._data_socket.close()
                sys.exit(-1)
            if hostname == executor_hostname:
                data_socket.connect(socket_path)
            else:
                if "uuid" not in locals():
                    import uuid
                self._local_socket_path = f"{socket_path}.task{uuid.uuid4().hex[:4]}"
                self._use_ssh_tunnel = True
                ssh_cmd: List[str] = [
                    "ssh",
                    "-o",
                    "LogLevel=quiet",
                    "-L",
                    f"{self._local_socket_path}:{socket_path}",
                    executor_hostname,
                    "sleep",
                    "2",
                ]
                logger.debug(f"Opening tunnel from {hostname} to {executor_hostname}")
                self._ssh_proc = subprocess.Popen(ssh_cmd)
                time.sleep(0.4)  # Need to wait... -> Use single Task comm at beginning?
                data_socket.connect(self._local_socket_path)
        return data_socket

    def _create_socket(self) -> socket.socket:
        """Create either a Unix or TCP socket.

        If the environment variable:
                              `LUTE_USE_TCP=1`
        is defined, a TCP socket is returned, otherwise a Unix socket.

        Refer to the individual initialization methods for additional environment
        variables controlling the behaviour of these two communication types.

        Returns:
            data_socket (socket.socket): TCP or Unix socket.
        """
        import struct

        use_tcp: Optional[str] = os.getenv("LUTE_USE_TCP")
        sock: socket.socket
        if use_tcp is not None:
            sock = self._init_tcp_socket()
        else:
            sock = self._init_unix_socket()
        sock.setsockopt(
            socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 10000)
        )
        return sock

    def _write_socket(self, msg: Message) -> None:
        """Sends data over a socket from the 'client' (Task) side.

        Messages are encoded in the following format:
                      MSG,<len(msg)>,<msg>,GSM
        The items between <> are replaced with the length of the message and
        the message itself while the bytes versions of the literals "MSG", ","
        and "GSM" are used to delineate and bookend the actual data.

        This structure is used for decoding the message on the other end.
        """
        data: bytes = pickle.dumps(msg)
        cmd: bytes = b"MSG"
        size: bytes = b"%d" % len(data)
        end: bytes = b"GSM"
        sep: bytes = b","
        packed_msg: bytes = cmd + sep + size + sep + data + sep + end
        # self._data_socket.sendall(pickle.dumps(msg))
        self._data_socket.sendall(packed_msg)

    def _clean_up(self) -> None:
        """Clean up connections."""
        # Check the object exists in case the Communicator is cleaned up before
        # opening any connections
        if self._party == Party.EXECUTOR:
            self._stop_thread = True
            self._reader_thread.join()
            logger.debug("Closed reading thread.")

        if hasattr(self, "_data_socket") and not self._data_socket._closed:
            socket_path: str = self.socket_path
            self._data_socket.close()
            if os.getenv("LUTE_USE_TCP"):
                return
            if self._party == Party.EXECUTOR or self._use_ssh_tunnel:
                os.unlink(socket_path)
                if self._ssh_proc is not None:
                    self._ssh_proc.terminate()

    @property
    def socket_path(self) -> str:
        socket_path: str = self._data_socket.getsockname()
        if not socket_path:
            socket_path = os.environ["LUTE_SOCKET"]
            if self._party == Party.TASK and self._use_ssh_tunnel:
                # If _use_ssh_tunnel _local_socket_path is defined.
                return self._local_socket_path
        return socket_path

    @property
    def has_messages(self) -> bool:
        if self._party == Party.TASK:
            # Shouldn't be called on Task-side
            return False

        if self._msg_queue.qsize() > 0:
            return True
        return False

    def __exit__(self):
        self._clean_up()

    def __del__(self):
        self._clean_up()


import zmq


class ZMQCommunicator(Communicator):
    """Provides communication over Unix or TCP sockets with ZMQ.

    Whether to use TCP or Unix sockets is controlled by the environment:
                           `LUTE_USE_TCP=1`
    If defined, TCP sockets will be used, otherwise Unix sockets will be used.

    Regardless of socket type, the environment variable
                      `LUTE_EXECUTOR_HOST=<hostname>`
    will be defined by the Executor-side Communicator.


    For TCP sockets:
    The Executor-side Communicator should be run first and will bind to all
    interfaces on the port determined by the environment variable:
                            `LUTE_PORT=###`
    If no port is defined, ZMQ will bind a random port. The Communicator
    will then define the environment variable so the Task-side can pick it up.

    For Unix sockets:
    The path to the Unix socket is defined by the environment variable:
                      `LUTE_SOCKET=/path/to/socket`
    This class assumes proper permissions and that this above environment
    variable has been defined. The `Task` is configured as what would commonly
    be referred to as the `client`, while the `Executor` is configured as the
    server.

    If the Task process is run on a different machine than the Executor, the
    Task-side Communicator will open a ssh-tunnel to forward traffic from a local
    Unix socket to the Executor Unix socket. Opening of the tunnel relies on the
    environment variable:
                      `LUTE_EXECUTOR_HOST=<hostname>`
    to determine the Executor's host. This variable should be defined by the
    Executor and passed to the Task process automatically, but it can also be
    defined manually if launching the Task process separately. The Task will use
    the local socket `<LUTE_SOCKET>.task{##}`. Multiple local sockets may be
    created. Currently, it is assumed that the user is identical on both the Task
    machine and Executor machine.
    """

    READ_TIMEOUT: float = 0.01
    """
    Maximum time to wait to retrieve data.
    """

    def __init__(self, party: Party = Party.TASK, use_pickle: bool = True) -> None:
        """IPC over a TCP or Unix socket.

        Args:
            party (Party): Which object (side/process) the Communicator is
                managing IPC for. I.e., is this the "Task" or "Executor" side.

            use_pickle (bool): Whether to use pickle. Always True currently,
                passing False does not change behaviour.
        """
        super().__init__(party=party, use_pickle=use_pickle)
        self.desc: str = "Communicates using ZMQ through TCP or Unix sockets."
        if self._party == Party.EXECUTOR:
            # Executor created first so we can define the hostname env variable
            os.environ["LUTE_EXECUTOR_HOST"] = socket.gethostname()

        self._context: zmq.context.Context = zmq.Context()
        self._use_ssh_tunnel: bool = False
        self._ssh_proc: Optional[subprocess.Popen] = None
        self._local_socket_path: Optional[str] = None
        self._data_socket: zmq.sugar.socket.Socket = self._create_socket()

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
        msg: Message = Message()
        try:
            data: bytes = self._data_socket.recv(zmq.NOBLOCK)
            msg = pickle.loads(data)
        except zmq.error.Again:
            pass
        finally:
            return msg

    def write(self, msg: Message) -> None:
        """Send a single Message.

        The entire Message (signal and contents) is serialized and sent through
        a connection over Unix socket.

        Args:
            msg (Message): The Message to send.
        """
        self._write_socket(msg)

    def _init_tcp_socket(self, data_socket: zmq.sugar.socket.Socket) -> None:
        """Initialize a TCP socket.

        Executor-side code should always be run first. It checks to see if
        the environment variable
                                `LUTE_PORT=###`
        is defined, if so binds it, otherwise find a free port from a selection
        of random ports. If a port search is performed, the `LUTE_PORT` variable
        will be defined so it can be picked up by the the Task-side Communicator.

        In the event that no port can be bound on the Executor-side, or the port
        and hostname information is unavailable to the Task-side, the program
        will exit.

        Args:
            data_socket (zmq.socket.Socket): Socket object.
        """
        port: Optional[Union[str, int]] = os.getenv("LUTE_PORT")
        if self._party == Party.EXECUTOR:
            if port is None:
                new_port: int = data_socket.bind_to_random_port("tcp://*")
                if new_port is None:
                    # Failed to find a port to bind
                    logger.info(
                        "Executor failed to bind a port. "
                        "Try providing a LUTE_PORT directly! Exiting!"
                    )
                    sys.exit(-1)
                port = new_port
                os.environ["LUTE_PORT"] = str(port)
            else:
                data_socket.bind(f"tcp://*:{port}")
            logger.debug(f"Executor bound port {port}")
            # data_socket.subscribe(b"") # Can define topics later.
            # data_socket.setsockopt(zmq.LINGER,0)
        else:
            executor_hostname: Optional[str] = os.getenv("LUTE_EXECUTOR_HOST")
            if executor_hostname is None or port is None:
                logger.info(
                    "Task-side does not have host/port information!"
                    " Check environment variables! Exiting!"
                )
                sys.exit(-1)
            data_socket.connect(f"tcp://{executor_hostname}:{port}")

    def _init_unix_socket(self, data_socket: zmq.sugar.socket.Socket) -> None:
        """Initialize a Unix socket object.

        Executor-side code should always be run first. It checks to see if
        the environment variable
                                `LUTE_SOCKET=XYZ`
        is defined, if so binds it, otherwise it will create a new path and
        define the environment variable for the Task-side to find.

        On the Task (client-side), this method will also open a SSH tunnel to
        forward a local Unix socket to an Executor Unix socket if the Task and
        Executor processes are on different machines.

        The SSH tunnel is opened with `ssh -L <local>:<remote> sleep 2`.
        This method of communication is slightly slower and incurs additional
        overhead - it should only be used as a backup. If communication across
        multiple hosts is required consider using TCP.  The Task will use
        the local socket `<LUTE_SOCKET>.task{##}`. Multiple local sockets may be
        created. It is assumed that the user is identical on both the
        Task machine and Executor machine.

        Args:
            data_socket (socket.socket): ZMQ object.
        """
        try:
            socket_path = os.environ["LUTE_SOCKET"]
        except KeyError:
            import uuid
            import tempfile

            # Define a path,up and add to environment
            # Executor-side always created first, Task will use the same one
            socket_path = f"{tempfile.gettempdir()}/lute_{uuid.uuid4().hex}.sock"
            os.environ["LUTE_SOCKET"] = socket_path
            logger.debug(f"ZMQCommunicator defines socket_path: {socket_path}")

        socket_path = f"ipc:/{socket_path}"
        if self._party == Party.EXECUTOR:
            if os.path.exists(socket_path):
                os.unlink(socket_path)
            data_socket.bind(socket_path)
            # data_socket.subscribe(b"")
        elif self._party == Party.TASK:
            hostname: str = socket.gethostname()
            executor_hostname: Optional[str] = os.getenv("LUTE_EXECUTOR_HOST")
            if executor_hostname is None:
                logger.info("Hostname for Executor process not found! Exiting!")
                self._data_socket.close()
                sys.exit(-1)
            if hostname == executor_hostname:
                data_socket.connect(socket_path)
            else:
                if "uuid" not in locals():
                    import uuid
                self._local_socket_path = f"{socket_path}.task{uuid.uuid4().hex[:4]}"
                self._use_ssh_tunnel = True
                ssh_cmd: List[str] = [
                    "ssh",
                    "-o",
                    "LogLevel=quiet",
                    "-L",
                    f"{self._local_socket_path[5:]}:{socket_path[5:]}",  # remove ipc:/
                    executor_hostname,
                    "sleep",
                    "2",
                ]
                logger.debug(f"Opening tunnel from {hostname} to {executor_hostname}")
                self._ssh_proc = subprocess.Popen(ssh_cmd)
                time.sleep(0.4)  # Need to wait... -> Use single Task comm at beginning?
                data_socket.connect(self._local_socket_path)

    def _create_socket(self) -> zmq.sugar.socket.Socket:
        """Create either a Unix or TCP socket.

        If the environment variable:
                              `LUTE_USE_TCP=1`
        is defined, a TCP socket is returned, otherwise a Unix socket.

        Refer to the individual initialization methods for additional environment
        variables controlling the behaviour of these two communication types.

        Returns:
            data_socket (socket.socket): Unix socket object.
        """
        socket_type: Literal[zmq.SUB, zmq.PUB]
        if self._party == Party.EXECUTOR:
            socket_type = zmq.PULL
        else:
            socket_type = zmq.PUSH

        data_socket: zmq.sugar.socket.Socket = self._context.socket(socket_type)
        data_socket.set_hwm(160000)
        # Try TCP first
        use_tcp: Optional[str] = os.getenv("LUTE_USE_TCP")
        if use_tcp is not None:
            self._init_tcp_socket(data_socket)
        else:
            self._init_unix_socket(data_socket)

        return data_socket

    def _write_socket(self, msg: Message) -> None:
        """Sends data over a socket from the 'client' (Task) side."""
        self._data_socket.send(pickle.dumps(msg))

    def _clean_up(self) -> None:
        """Clean up connections."""
        self._data_socket.close()
        self._context.term()

    def __exit__(self):
        self._clean_up()

    def __del__(self):
        self._clean_up()


def SocketCommunicator(*args, **kwargs) -> Communicator:
    """Selector for RawSocketCommunicator or ZMQCommunicator.

    Returns:
        communicator (Communicator): RawSocketCommunicator. ZMQCommunicator
            is currently unused.
    """
    use_zmq: bool = False
    if use_zmq:
        return ZMQCommunicator(*args, **kwargs)
    return RawSocketCommunicator(*args, **kwargs)
