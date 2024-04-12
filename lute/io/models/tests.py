"""Models for all test Tasks.

Classes:
    TestParameters(TaskParameters): Model for most basic test case. Single
        core first-party Task. Uses only communication via pipes.

    TestBinaryParameters(BaseBinaryParameters): Parameters for a simple multi-
        threaded binary executable.

    TestSocketParameters(TaskParameters): Model for first-party test requiring
        communication via socket.

    TestWriteOutputParameters(TaskParameters): Model for test Task which writes
        an output file. Location of file is recorded in database.

    TestReadOutputParameters(TaskParameters): Model for test Task which locates
        an output file based on an entry in the database, if no path is provided.
"""

__all__ = [
    "TestParameters",
    "TestBinaryParameters",
    "TestBinaryErrParameters",
    "TestSocketParameters",
    "TestWriteOutputParameters",
    "TestReadOutputParameters",
]
__author__ = "Gabriel Dorlhiac"

from typing import Dict, Any

from pydantic import (
    BaseModel,
    Field,
    validator,
)

from .base import TaskParameters, BaseBinaryParameters
from ..db import read_latest_db_entry


class TestParameters(TaskParameters):
    """Parameters for the test Task `Test`."""

    float_var: float = 0.01
    str_var: str = "test"

    class CompoundVar(BaseModel):
        int_var: int = 1
        dict_var: Dict[str, str] = {"a": "b"}

    compound_var: CompoundVar
    throw_error: bool = False


class TestBinaryParameters(BaseBinaryParameters):
    executable: str = "/sdf/home/d/dorlhiac/test_tasks/test_threads"
    p_arg1: int = 1


class TestBinaryErrParameters(BaseBinaryParameters):
    """Same as TestBinary, but exits with non-zero code."""

    executable: str = "/sdf/home/d/dorlhiac/test_tasks/test_threads_err"
    p_arg1: int = 1


class TestSocketParameters(TaskParameters):
    array_size: int = 10000
    num_arrays: int = 10


class TestWriteOutputParameters(TaskParameters):
    outfile_name: str = Field(
        "test_output.txt", description="Outfile name without full path."
    )
    num_vals: int = Field(100, description='Number of values to "process"')


class TestReadOutputParameters(TaskParameters):
    in_file: str = Field("", description="File to read in. (Full path)")

    @validator("in_file", always=True)
    def validate_in_file(cls, in_file: str, values: Dict[str, Any]) -> str:
        if in_file == "":
            filename: str = read_latest_db_entry(
                f"{values['lute_config'].work_dir}", "TestWriteOutput", "outfile_name"
            )
            in_file: str = f"{values['lute_config'].work_dir}/{filename}"
        return in_file
