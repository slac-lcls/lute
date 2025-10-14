"""
Task for converting xtc1 files to xtc2 format using zmq-based communication
between psana1 and psana2 environments.

Classes:
    - ConvertXtc1to2(Task): Convert xtc1 files to xtc2 format.

Based on Mona's converter:
    https://github.com/monarin/xtc1to2
"""

import json
import logging
import os
import subprocess
import time
from typing import Union, cast

from lute.execution.logging import get_logger
from lute.io.models.xtc import ConvertXtc1to2Parameters
from lute.tasks.task import Task

logger: logging.Logger = get_logger(__name__)


class ConvertXtc1to2(Task):
    """
    A task that launches two zmq subprocesses in two different conda environments
    (old psana and new psana) to convert (old) xtc1 files to (new) xtc2 files.

    This involves:
        - Process 1: Opening the xtc1 data using psana1 environment.
        - Process 2: Receives and writes xtc2 data using psana2 environment.
        - ZeroMQ used for communication between the processes.

    Args:
        params (ConvertXtc1to2Parameters): Configuration for the conversion task.
    """

    def __init__(self, *, params: ConvertXtc1to2Parameters) -> None:
        super().__init__(params=params)

    def _run(self) -> None:
        self._task_parameters = cast(ConvertXtc1to2Parameters, self._task_parameters)
        par: ConvertXtc1to2Parameters = self._task_parameters
        exp: str = par.lute_config.experiment
        run: Union[int, str] = par.lute_config.run
        logger.debug("Starting [XTC1 Sender] in psana 1")
        json_access_pattern: str = json.dumps(par.xtc1_access_pattern)
        py_path: str = f"{os.getenv('LUTE_PATH')}:{os.getenv('PYTHONPATH')}"
        zmq_process1_cmd: str = (
            f"source /sdf/group/lcls/ds/ana/sw/conda1/manage/bin/psconda.sh && "
            f"export PYTHONPATH={py_path} && "
            f"python3 lute/tasks/util/xtc_push.py -a '{json_access_pattern}' -e {exp} "
            f"-r {par.lute_config.run} -m {par.mode} -d {par.detector} "
            f"-g {par.geometry} -l {par.resolution} "
        )
        if par.eventfile != "":
            zmq_process1_cmd += f"-f {par.eventfile} "

        if par.verify == "True":
            zmq_process1_cmd += f"-v 1 -t {par.testfile}"
        else:
            zmq_process1_cmd += "-v 0"
        result_p1: subprocess.Popen = self._start_zmq_proc(
            zmq_process1_cmd, "[XTC1 Sender]"
        )

        time.sleep(1)

        logger.debug("Starting [XTC2 Writer] in psana 2")
        zmq_process2_cmd: str = (
            f"source /sdf/group/lcls/ds/ana/sw/conda2/manage/bin/psconda.sh && "
            f"export PYTHONPATH=/sdf/home/k/kmecseki/munka/lcls2/psana:$PYTHONPATH && "
            f"python3 lute/tasks/util/xtc_pull.py -d {par.detector} -e {exp} "
            f"-f {par.output_file} -l {par.resolution} -n {par.node_id} -r {run} "
        )
        if par.verify == "True":
            zmq_process2_cmd += "-v 1"
        else:
            zmq_process2_cmd += "-v 0"
        result_p2: subprocess.Popen = self._start_zmq_proc(
            zmq_process2_cmd, "[XTC2 Writer]"
        )

        out_p1, err_p1 = result_p1.communicate()
        out_p2, err_p2 = result_p2.communicate()

        logger.debug(f"[XTC1 Sender] Output::\n{out_p1}")
        logger.error(f"[XTC1 Sender] Error:\n{err_p1}")
        logger.debug(f"[XTC2 Writer] Output:\n{out_p2}")
        logger.error(f"[XTC2 Writer] Error::\n{err_p2}")

    def _start_zmq_proc(self, cmd: str, name: str) -> subprocess.Popen:
        """Helper function to source the correct conda env and spawn a subprocess."""

        process: subprocess.Popen = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, text=True
        )
        logger.debug(f"{name} started")
        logger.debug(f"Command: {cmd}")

        return process
