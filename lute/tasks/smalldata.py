"""Tasks for working with SmallData HDF5 files.

Classes defined in this module provide an interface to extracting data from
SmallData files and analyzing it.

Classes:
    AnalyzeSmallDataXSS(Task): Analyze scattering data for a single detector in
        a SmallData file.

    AnalyzeSmallDataXAS(Task): Analyze absorption data for a single detector in
        a SmallData file.
"""

__all__ = ["AnalyzeSmallDataXSS", "AnalyzeSmallDataXAS"]
__author__ = "Gabriel Dorlhiac"

import sys
import logging
from typing import List, Optional, Dict, Tuple, Union

import h5py
import holoviews as hv
import numpy as np
import panel as pn
import matplotlib.pyplot as plt
from mpi4py import MPI
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

from lute.execution.ipc import Message
from lute.io.models.base import TaskParameters
from lute.tasks.task import *
from lute.tasks.dataclasses import ElogSummaryPlots
from lute.tasks.math import gaussian, sigma_to_fwhm
from lute.tasks._smalldata import AnalyzeSmallData


logger: logging.Logger = logging.getLogger(__name__)

if __debug__:
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.INFO)


class AnalyzeSmallDataXSS(AnalyzeSmallData):
    """Task to analyze XSS profiles stored in a SmallData HDF5 file."""

    def __init__(self, *, params: TaskParameters, use_mpi: bool = True) -> None:
        super().__init__(params=params, use_mpi=use_mpi)

    def _pre_run(self) -> None:
        # Currently scattering data is extracted as standard since its used
        # for all analysis types (XSS, XAS, XES,...)
        self._extract_standard_data()

    def _run(self) -> None:
        diff: np.ndarray[np.float64]
        bins: np.ndarray[np.float64]
        laser_on: np.ndarray[np.float64]
        bins, diff, laser_on = self._calc_scan_binned_difference_xss()

        def sum_diff(
            diff0: np.ndarray[np.float64], diff1: np.ndarray[np.float64]
        ) -> np.ndarray[np.float64]:
            return diff0 + diff1

        def laser_on_mean(
            laser_on0: np.ndarray[np.float64], laser_on1: np.ndarray[np.float64]
        ) -> np.ndarray[np.float64]:
            return laser_on0.sum(axis=0) + laser_on1.sum(axis=0)

        if self._mpi_size > 1:
            diff = self._mpi_comm.reduce(diff, op=sum_diff)
            laser_on = self._mpi_comm.reduce(laser_on, op=laser_on_mean)
        else:
            laser_on = np.nansum(laser_on, axis=0)

        if self._mpi_rank == 0:
            diff /= self._mpi_size
            laser_on /= self._total_num_events
            name: str = self._scan_var_name if self._scan_var_name else "By_Event"
            plots: pn.Tabs = self.plot_all_xss(laser_on, bins, diff, name)
            plot_display_name: str
            run: int
            try:
                run = int(self._task_parameters.lute_config.run)
            except ValueError:
                run = 0
            exp_run: str = f"{run:04d}_{name}_XSS"
            if "lens" in name:
                plot_display_name = f"lens_scans/{exp_run}"
            else:
                plot_display_name = f"time_scans/{exp_run}"

            self._result.payload = ElogSummaryPlots(plot_display_name, plots)


class AnalyzeSmallDataXAS(AnalyzeSmallData):
    """Task to analyze XAS data stored in a SmallData HDF5 file."""

    def __init__(self, *, params: TaskParameters, use_mpi: bool = True) -> None:
        super().__init__(params=params, use_mpi=use_mpi)

    def _pre_run(self) -> None:
        # Currently scattering data is extracted as standard since its used
        # for all analysis types (XSS, XAS, XES,...)
        self._extract_standard_data()
        self._extract_xas(self._task_parameters.xas_detname)

    def _run(self) -> None:
        # XAS returns two sets of binned data
        # Bins raw TR-XAS first, then bins by scan
        diff: np.ndarray[np.float64]
        ccm_bins: np.ndarray[np.float64]
        laser_on: np.ndarray[np.float64]
        laser_off: np.ndarray[np.float64]
        ccm_bins, diff, laser_on, laser_off = self._calc_binned_difference_xas()

        def sum_diff(
            diff0: np.ndarray[np.float64], diff1: np.ndarray[np.float64]
        ) -> np.ndarray[np.float64]:
            return diff0 + diff1

        if self._mpi_size > 1:
            diff = self._mpi_comm.reduce(diff, op=MPI.SUM)
            laser_on = self._mpi_comm.reduce(laser_on, op=MPI.SUM)
            laser_off = self._mpi_comm.reduce(laser_off, op=MPI.SUM)

        all_plots: List[ElogSummaryPlots] = []
        run: int
        try:
            run = int(self._task_parameters.lute_config.run)
        except ValueError:
            run = 0
        plot_display_name: str
        exp_run: str
        if self._mpi_rank == 0:
            diff /= self._mpi_size
            laser_on /= self._mpi_size
            laser_off /= self._mpi_size
            plots: pn.Tabs = self.plot_all_xas(laser_on, laser_off, ccm_bins, diff)
            exp_run = f"{run:04d}_XAS"
            plot_display_name = f"XAS_{exp_run}"
            all_plots.append(ElogSummaryPlots(plot_display_name, plots))

        scan_bins: np.ndarray[np.float64]
        scan_bins, diff, laser_on, laser_off = self._calc_scan_binned_difference_xas()
        if self._mpi_size > 1:
            diff = self._mpi_comm.reduce(diff, op=MPI.SUM)
            laser_on = self._mpi_comm.reduce(laser_on, op=MPI.SUM)
            laser_off = self._mpi_comm.reduce(laser_off, op=MPI.SUM)

        if self._mpi_rank == 0:
            plots: pn.Tabs = self.plot_xas_scan_hv(laser_on, laser_off, scan_bins, diff)
            name: str = self._scan_var_name if self._scan_var_name else "By_Event"
            exp_run = f"{run:04d}_{name}_XAS"
            if "lens" in name:
                plot_display_name = f"lens_scans/{exp_run}"
            elif "lxe_opa" in name:
                plot_display_name = f"power_scans/{exp_run}"
            else:
                plot_display_name = f"time_scans/{exp_run}"

            all_plots.append(ElogSummaryPlots(plot_display_name, plots))
        self._result.payload = all_plots

    def _post_run(self) -> None: ...
