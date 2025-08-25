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
from typing import List

logger: logging.Logger = get_logger(__name__)

class Xtc1to2(Task):
    """
    A task that launches two zmq subprocesses in two different conda environments
    (old psana and new psana) to convert (old) Xtc1 files to (new) Xtc2 files.
    """

    def __init__(self) -> None:
        pass

    def _run(self) -> None:
        logger.debug("Opening Xtc1 in psana 1 env, starting Zmq process 1\n")
        
        zmq_process1_cmd: List[str] = [
            'bash',
            '-c',
            '''source /sdf/group/lcls/ds/ana/sw/conda1/manage/bin/psconda.sh && \
            python lute/tasks/util/xtc_push.py'''
            ]
        
        result_p1: subprocess.Popen = _start_zmq_proc(zmq_process1_cmd)
        
        logger.debug("Zmq process 1 started\n")
            
        # Making sure Zmq sender is properly running
        time.sleep(1)

        logger.debug("Opening Xtc2 in psana 2 env, starting Zmq process 1\n")
        
        zmq_process2_cmd: List[str] = [
            'bash',
            '-c',
            '''source /sdf/group/lcls/ds/ana/sw/conda2/manage/bin/psconda.sh && \
            python lute/tasks/util/xtc_pull.py'''
            ]
        
        result_p2: subprocess.Popen = _start_zmq_proc(zmq_process2_cmd)
        
        logger.debug("Zmq process 2 started\n")

        out_p1, err_p1 = result_p1.communicate()
        out_p2, err_p2 = result_p2.communicate()

        logger.debug(f"Xtc converter Zmq push output:\n{out_p1}")
        logger.error(f"Xtc converter Zmq push error:\n{err_p1}")
        logger.debug(f"Xtc converter Zmq pull output:\n{out_p2}")
        logger.error(f"Xtc converter Zmq pull error:\n{err_p2}")

    def _start_zmq_proc(self, cmd: List[str]) -> subprocess.Popen:
        """Helper function to source the correct conda env and spawn a subprocess."""
        process: subprocess.Popen = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
                text=True,
                shell=True
        )

        if process.stdout is not None:
            os.set_blocking(process.stdout.fileno(), False)
        
        if process.stderr is not None:
            os.set_blocking(process.stderr.fileno(), False)

        return process

