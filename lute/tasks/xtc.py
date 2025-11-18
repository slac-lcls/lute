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
from typing import Any, Dict, List, Union

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
        self._task_parameters: ConvertXtc1to2Parameters
        super().__init__(params=params)

    def _run(self) -> None:
        par: ConvertXtc1to2Parameters = self._task_parameters
        exp: str = par.lute_config.experiment
        run: Union[int, str] = par.lute_config.run
        logger.debug("Starting [XTC1 Sender] in psana 1")
        data_spec: Dict[str, Any] = {}
        detnames: List[str] = []
        for detname, specs in par.xtc1_access_pattern.items():
            det_specs: List[Any] = []
            for spec in specs:
                det_specs.append(spec.dict())
            data_spec[detname] = det_specs
            detnames.append(detname)

        detname_csv: str = ",".join(detnames)
        json_access_pattern: str = json.dumps(data_spec)
        lute_location: str = os.getenv("LUTE_PATH", "")
        assert lute_location
        zmq_process1_cmd: str = (
            f"source /sdf/group/lcls/ds/ana/sw/conda1/manage/bin/psconda.sh && "
            f"python3 {lute_location}lib/python3.9/site-packages/lute/tasks/util/xtc_push.py "
            f"-a '{json_access_pattern}' -e {exp} "
            f"-r {par.lute_config.run} -m {par.mode} "
        )
        if par.eventfile != "":
            zmq_process1_cmd += f"-f {par.eventfile} "
        elif par.nevents is not None:
            zmq_process1_cmd += f"-n {par.nevents}"

        result_p1: subprocess.Popen = self._start_zmq_proc(
            zmq_process1_cmd, "[XTC1 Sender]"
        )

        time.sleep(1)

        logger.debug("Starting [XTC2 Writer] in psana 2")
        zmq_process2_cmd: str = (
            f"source /sdf/group/lcls/ds/ana/sw/conda2/manage/bin/psconda.sh && "
            f"python3 {lute_location}lib/python3.9/site-packages/lute/tasks/util/xtc_pull.py "
            f"-d {detname_csv} -e {exp} "
            f"-f {par.output_file} -n {par.node_id} -r {run} "
        )
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
