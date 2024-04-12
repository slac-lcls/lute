"""
Classes for indexing tasks in SFX.

Classes:
    ConcatenateStreamFIles: task that merges multiple stream files into a single file.
"""

__all__ = ["ConcatenateStreamFiles"]
__author__ = "Valerio Mariani"

import shutil
import sys
from pathlib import Path
from typing import BinaryIO, List

import numpy
from mpi4py import MPI

from lute.execution.ipc import Message
from lute.io.models.base import *
from lute.tasks.task import *


class ConcatenateStreamFiles(Task):
    """
    Task that merges stream files located within a directory tree.
    """

    def __init__(self, *, params: TaskParameters) -> None:
        super().__init__(params=params)

    def _run(self) -> None:

        stream_file_path: Path = Path(self._task_parameters.in_file)
        stream_file_list: List[Path] = list(
            stream_file_path.rglob(f"{self._task_parameters.tag}_*.stream")
        )

        processed_file_list = [str(stream_file) for stream_file in stream_file_list]

        msg: Message = Message(
            contents=f"Merging following stream files: {processed_file_list} into "
            f"{self._task_parameters.out_file}",
        )
        self._report_to_executor(msg)

        wfd: BinaryIO
        with open(self._task_parameters.out_file, "wb") as wfd:
            infile: Path
            for infile in stream_file_list:
                fd: BinaryIO
                with open(infile, "rb") as fd:
                    shutil.copyfileobj(fd, wfd)
