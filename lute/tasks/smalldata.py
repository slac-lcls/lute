"""Tasks for working with SmallData HDF5 files.

Classes defined in this module provide an interface to extracting data from
SmallData files and analyzing it.

Classes:
    AnalyzeSmallDataXSS(Task): Analyze scattering data for a single detector in
        a SmallData file.

    AnalyzeSmallDataXAS(Task): Analyze absorption data for a single detector in
        a SmallData file.

    AnalyzeSmallDataXES(Task): Analyze emission data for a single detector in
        a SmallData file.
"""

__all__ = ["AnalyzeSmallDataXSS", "AnalyzeSmallDataXAS", "AnalyzeSmallDataXES"]
__author__ = "Gabriel Dorlhiac"

from typing import List, Optional, cast

import numpy as np
import numpy.typing as npt
import panel as pn
from mpi4py import MPI

from lute.io.models.smd import (
    AnalyzeSmallDataXASParameters,
    AnalyzeSmallDataXESParameters,
    AnalyzeSmallDataXSSParameters,
)
from lute.tasks.dataclasses import ElogSummaryPlots
from lute.tasks._smalldata import AnalyzeSmallData


def sum_diff(
    diff0: npt.NDArray[np.float64], diff1: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    return diff0 + diff1


def laser_on_mean(
    laser_on0: npt.NDArray[np.float64], laser_on1: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    return laser_on0.sum(axis=0) + laser_on1.sum(axis=0)


def laser_off_mean(
    laser_off0: npt.NDArray[np.float64], laser_off1: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    return laser_off0.sum(axis=0) + laser_off1.sum(axis=0)


def sum_tjump(
    tjumps0: npt.NDArray[np.float64], tjumps1: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    return tjumps0 + tjumps1


class AnalyzeSmallDataXSS(AnalyzeSmallData):
    """Task to analyze XSS profiles stored in a SmallData HDF5 file."""

    def __init__(
        self,
        *,
        params: AnalyzeSmallDataXSSParameters,
        use_mpi: bool = True,
        row_ids=None,
    ) -> None:
        super().__init__(params=params, use_mpi=use_mpi, row_ids=row_ids)
        self._task_parameters = cast(
            AnalyzeSmallDataXSSParameters, self._task_parameters
        )

    def _pre_run(self) -> None:
        # Currently scattering data is extracted as standard since its used
        # for all analysis types (XSS, XAS, XES,...)
        self._extract_standard_data()
        self._extract_xss()

    def _run(self) -> None:
        diff: Optional[npt.NDArray[np.float64]]
        bins: npt.NDArray[np.float64]
        laser_on: Optional[npt.NDArray[np.float64]]
        laser_off: Optional[npt.NDArray[np.float64]]
        tjumps: Optional[npt.NDArray[np.float64]]
        processed_tjumps: Optional[npt.NDArray[np.float64]]
        bins, diff, laser_on, laser_off = self._calc_scan_binned_difference_xss()
        tjumps, processed_tjumps = self._calc_tjump_xss()

        if self._mpi_size > 1:
            diff = self._mpi_comm.reduce(diff, op=sum_diff)
            laser_on = self._mpi_comm.reduce(laser_on, op=laser_on_mean)
            laser_off = self._mpi_comm.reduce(laser_off, op=laser_off_mean)
            tjumps = self._mpi_comm.reduce(tjumps, op=sum_tjump)
            processed_tjumps = self._mpi_comm.reduce(processed_tjumps, op=sum_tjump)
        if (
            self._mpi_rank == 0
            and diff is not None
            and laser_on is not None
            and laser_off is not None
            and tjumps is not None
            and processed_tjumps is not None
        ):
            diff /= self._mpi_size
            laser_on /= self._total_num_events
            laser_off /= self._total_num_events
            tjumps /= self._mpi_size
            processed_tjumps /= self._mpi_size
            name: str = self._scan_var_name if self._scan_var_name else "by_event"
            plots = self.plot_all_xss(
                laser_on, bins, diff, name, tjumps, processed_tjumps
            )
            plot_display_name: str
            run: int
            try:
                run = int(self._task_parameters.lute_config.run)
            except ValueError:
                run = 0
            if name == "by_event":
                plot_display_name = f"XSS/{run:04d}_by_event"
            else:
                plot_display_name = f"XSS/{run:04d}_{name}"

            self._result.payload = ElogSummaryPlots(plot_display_name, plots)


class AnalyzeSmallDataXAS(AnalyzeSmallData):
    """Task to analyze XAS data stored in a SmallData HDF5 file."""

    def __init__(
        self,
        *,
        params: AnalyzeSmallDataXASParameters,
        use_mpi: bool = True,
        row_ids=None,
    ) -> None:
        super().__init__(params=params, use_mpi=use_mpi, row_ids=row_ids)

    def _pre_run(self) -> None:
        # Currently scattering data is extracted as standard since its used
        # for all analysis types (XSS, XAS, XES,...)
        self._task_parameters = cast(
            AnalyzeSmallDataXASParameters, self._task_parameters
        )
        self._extract_standard_data()
        self._extract_xas(self._task_parameters.xas_detname)

    def _run(self) -> None:
        # XAS returns two sets of binned data
        # Bins raw TR-XAS first, then bins by scan
        diff: Optional[npt.NDArray[np.float64]]
        ccm_bins: Optional[npt.NDArray[np.float64]]
        laser_on: Optional[npt.NDArray[np.float64]]
        laser_off: Optional[npt.NDArray[np.float64]]
        ccm_bins, diff, laser_on, laser_off = self._calc_binned_difference_xas()

        # We check None on ccm_bins because the monochromator is not always
        # scanned -> We return None if there aren't enough bins to be worth
        # plotting
        if self._mpi_size > 1 and ccm_bins is not None:
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
        plots: Optional[pn.Tabs]
        if (
            self._mpi_rank == 0
            and ccm_bins is not None
            and diff is not None
            and laser_on is not None
            and laser_off is not None
        ):
            # Check None again
            diff /= self._mpi_size
            laser_on /= self._mpi_size
            laser_off /= self._mpi_size
            plots = self.plot_all_xas(laser_on, laser_off, ccm_bins, diff)
            plot_display_name = f"XAS/{run:04d}"
            all_plots.append(ElogSummaryPlots(plot_display_name, plots))

        scan_bins: Optional[npt.NDArray[np.float64]]
        scan_bins, diff, laser_on, laser_off = self._calc_scan_binned_difference_xas()
        if self._mpi_size > 1 and scan_bins is not None:
            diff = self._mpi_comm.reduce(diff, op=MPI.SUM)
            laser_on = self._mpi_comm.reduce(laser_on, op=MPI.SUM)
            laser_off = self._mpi_comm.reduce(laser_off, op=MPI.SUM)

        if (
            self._mpi_rank == 0
            and scan_bins is not None
            and diff is not None
            and laser_on is not None
            and laser_off is not None
        ):
            plots = self.plot_xas_scan_hv(laser_on, laser_off, scan_bins, diff)
            if plots is not None:
                name: str = self._scan_var_name if self._scan_var_name else "by_event"
                if name == "by_event":
                    plot_display_name = f"XAS/{run:04d}_by_event"
                else:
                    plot_display_name = f"XAS/{run:04d}_{name}"

                all_plots.append(ElogSummaryPlots(plot_display_name, plots))
        self._result.payload = all_plots

    def _post_run(self) -> None: ...


class AnalyzeSmallDataXES(AnalyzeSmallData):
    """Task to analyze XES data stored in a SmallData HDF5 file."""

    def __init__(
        self,
        *,
        params: AnalyzeSmallDataXESParameters,
        use_mpi: bool = True,
        row_ids=None,
    ) -> None:
        super().__init__(params=params, use_mpi=use_mpi, row_ids=row_ids)

    def _pre_run(self) -> None:
        # Currently scattering data is extracted as standard since its used
        # for all analysis types (XSS, XAS, XES,...)
        self._task_parameters = cast(
            AnalyzeSmallDataXESParameters, self._task_parameters
        )
        self._extract_standard_data()
        self._extract_xes(self._task_parameters.xes_detname)

    def _run(self) -> None:
        # XES returns two sets of data
        # Average TR-XES first, then bins by scan variable
        diff: Optional[npt.NDArray[np.float64]]
        laser_on: Optional[npt.NDArray[np.float64]]
        laser_off: Optional[npt.NDArray[np.float64]]
        diff, laser_on, laser_off = self._calc_avg_difference_xes()

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
        plots: Optional[pn.Tabs]
        if (
            self._mpi_rank == 0
            and diff is not None
            and laser_on is not None
            and laser_off is not None
        ):
            diff /= self._mpi_size
            laser_on /= self._mpi_size
            laser_off /= self._mpi_size
            energy_bins: Optional[npt.NDArray[np.float64]] = None
            plots = self.plot_xes_hv(laser_on, laser_off, energy_bins, diff)
            plot_display_name = f"XES/{run:04d}"
            all_plots.append(ElogSummaryPlots(plot_display_name, plots))

        scan_bins: npt.NDArray[np.float64]
        scan_bins, diff, laser_on, laser_off = self._calc_scan_binned_difference_xes()
        if self._mpi_size > 1:
            diff = self._mpi_comm.reduce(diff, op=MPI.SUM)
            laser_on = self._mpi_comm.reduce(laser_on, op=MPI.SUM)
            laser_off = self._mpi_comm.reduce(laser_off, op=MPI.SUM)

        if (
            self._mpi_rank == 0
            and diff is not None
            and laser_on is not None
            and laser_off is not None
        ):
            plots = self.plot_xes_scan_hv(laser_on, laser_off, scan_bins, diff)
            if plots is not None:
                name: str = self._scan_var_name if self._scan_var_name else "by_event"
                if name == "by_event":
                    plot_display_name = f"XES/{run:04d}_by_event"
                else:
                    plot_display_name = f"XES/{run:04d}_{name}"

                all_plots.append(ElogSummaryPlots(plot_display_name, plots))
        self._result.payload = all_plots

    def _post_run(self) -> None: ...
