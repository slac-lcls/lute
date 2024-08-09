"""Models for all test Tasks.

Classes:
    TestMultiNodeCommunicationParameters(TaskParameters): Model for test Task
        which verifies that the SocketCommunicator can write back to the
        Executor on a different node.
    TestParameters(TaskParameters): Model for most basic test case. Single
        core first-party Task. Uses only communication via pipes.

    TestBinaryParameters(ThirdPartyParameters): Parameters for a simple multi-
        threaded binary executable.

    TestSocketParameters(TaskParameters): Model for first-party test requiring
        communication via socket.

    TestWriteOutputParameters(TaskParameters): Model for test Task which writes
        an output file. Location of file is recorded in database.

    TestReadOutputParameters(TaskParameters): Model for test Task which locates
        an output file based on an entry in the database, if no path is provided.
"""

__all__ = ["TestMultiNodeCommunicationParameters"]
__author__ = "Gabriel Dorlhiac"

from typing import Dict, Any, Optional, Literal

from pydantic import (
    BaseModel,
    Field,
    validator,
)

from .base import TaskParameters, ThirdPartyParameters
from ..db import read_latest_db_entry


class TestMultiNodeCommunicationParameters(TaskParameters):
    """Parameters for the test Task `TestMultiNodeCommunication`.

    Test verifies communication across multiple machines.
    """

    send_obj: Literal["plot", "array"] = Field(
        "array", description="Object to send to Executor. `plot` or `array`"
    )
    arr_size: Optional[int] = Field(
        None, description="Size of array to send back to Executor."
    )
