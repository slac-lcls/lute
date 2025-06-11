"""Models for MPI test Tasks.

Classes:
    TestMultiNodeCommunicationParameters(TaskParameters): Model for test Task
        which verifies that the SocketCommunicator can write back to the
        Executor on a different node.
"""

__all__ = ["TestMultiNodeCommunicationParameters"]
__author__ = "Gabriel Dorlhiac"

from typing import Optional, Literal

from lute.io.models.base import TaskParameters, Field


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
