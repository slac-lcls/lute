"""
Classes related to converting xtc1 files to xtc2.

Classes:
    Xtc: class for working with xtc files, mainly to convert xtc1 to xtc2.
         Based on Mona's converter from https://github.com/monarin/xtc1to2
"""

import logging
import subprocess
import time
from lute.execution.logging import get_logger
from lute.io.models.xtc import ConvertXtc1to2Parameters
from lute.tasks.task import Task
from typing import List, cast

logger: logging.Logger = get_logger(__name__)

class ConvertXtc1to2(Task):
    """
    A task that launches two zmq subprocesses in two different conda environments
    (old psana and new psana) to convert (old) Xtc1 files to (new) Xtc2 files.
    """

    def __init__(self, *, params: ConvertXtc1to2Parameters) -> None:
        super().__init__(params=params)

    def _run(self) -> None:
        self._task_parameters = cast(
                ConvertXtc1to2Parameters, self._task_parameters
                )

        logger.debug("Starting [XTC1 Sender] in psana 1")

        par: ConvertXtc1to2Parameters = self._task_parameters

        zmq_process1_cmd: str = f'''
            source /sdf/group/lcls/ds/ana/sw/conda1/manage/bin/psconda.sh && \
            python3 lute/tasks/util/xtc_push.py'''
            # -e {par.exp} -r {par.run} -s {par.reshape} \
            #-m {par.mode} -d {par.detector} -g {par.geometry} -f {par.eventfile}'''

        result_p1: subprocess.Popen = self._start_zmq_proc(
                zmq_process1_cmd, "[XTC1 Sender]"
                )

        time.sleep(1)

        logger.debug("Starting [XTC2 Writer] in psana 2")

        zmq_process2_cmd: str = f'''
            source /sdf/group/lcls/ds/ana/sw/conda2/manage/bin/psconda.sh && \
            export PYTHONPATH=/sdf/home/k/kmecseki/munka/lcls2/psana:$PYTHONPATH && \
            python3 lute/tasks/util/xtc_pull.py -n {par.node_id}'''

        result_p2: subprocess.Popen = self._start_zmq_proc(
                zmq_process2_cmd, "[XTC2 Writer]"
                )

        out_p1, err_p1 = result_p1.communicate()
        out_p2, err_p2 = result_p2.communicate()

        logger.debug(f"[XTC1 Sender] Output::\n{out_p1}")
        logger.error(f"[XTC1 Sender] Error:\n{err_p1}")
        logger.debug(f"[XTC2 Writer] Output:\n{out_p2}")
        logger.error(f"[XTC2 Writer] Error::\n{err_p2}")

    def _start_zmq_proc(self, cmd: List[str], name: str) -> subprocess.Popen:
        """Helper function to source the correct conda env and spawn a subprocess."""
        process: subprocess.Popen = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=True,
                text=True
        )
        logger.debug(f"{name} started")

        return process
