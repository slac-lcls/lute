"""
Class involved in structure solution for SFX.

Classes:
    EditSHELXDInstructions: Modifies output from SHELXC, i.e. instructions for SHELXD.
"""

__all__ = ["EditSHELXDInstructions"]
__author__ = "Gabriel Dorlhiac"

import shutil
import sys
import logging
from pathlib import Path
from typing import BinaryIO, List

import numpy
from mpi4py import MPI

from lute.execution.ipc import Message
from lute.io.models.base import *
from lute.tasks.task import *

if __debug__:
    logging.basicConfig(level=logging.DEBUG)
    logging.captureWarnings(True)
else:
    logging.basicConfig(level=logging.INFO)

logger: logging.Logger = logging.getLogger(__name__)


class EditSHELXDInstructions(Task):
    """
    Task that merges stream files located within a directory tree.
    """

    def __init__(self, *, params: TaskParameters) -> None:
        super().__init__(params=params)

    def _run(self) -> None:
        with open(self._task_parameters.in_file) as f:
            lines_in: List[str] = f.readlines()
        print(lines_in, flush=True)

        lines_out: List[str] = []
        write_shel: bool = False
        write_mind: bool = False
        write_esel: bool = False
        write_test: bool = False
        for line in lines_in:
            new_line: str
            if "SHEL" in line:
                new_line = f"SHEL {self._task_parameters.SHEL[0]} {self._task_parameters.SHEL[1]}\n"
                write_shel = True
                lines_out.append(new_line)
            elif "MIND" in line:
                new_line = f"MIND {self._task_parameters.MIND[0]} {self._task_parameters.MIND[1]}\n"
                write_mind = True
                lines_out.append(new_line)
            elif "ESEL" in line:
                new_line = f"ESEL {self._task_parameters.ESEL}\n"
                write_esel = True
                lines_out.append(new_line)
            elif "TEST" in line:
                new_line = f"TEST {self._task_parameters.TEST[0]} {self._task_parameters.TEST[1]}\n"
                write_test = True
                lines_out.append(new_line)
            elif "END" in line:
                if not write_shel:
                    new_line = f"SHEL {self._task_parameters.SHEL[0]} {self._task_parameters.SHEL[1]}\n"
                    lines_out.append(new_line)
                if not write_mind:
                    new_line = f"MIND {self._task_parameters.MIND[0]} {self._task_parameters.MIND[1]}\n"
                    lines_out.append(new_line)
                if not write_esel:
                    new_line = f"ESEL {self._task_parameters.ESEL}\n"
                    lines_out.append(new_line)
                if not write_test:
                    new_line = f"TEST {self._task_parameters.TEST[0]} {self._task_parameters.TEST[1]}\n"
                    lines_out.append(new_line)
                lines_out.append("END\n")
            else:
                lines_out.append(line)
        with open(self._task_parameters.out_file, "w") as f:
            for line in lines_out:
                f.write(line)
