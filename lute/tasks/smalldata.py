"""Tasks for working with SmallData HDF5 files.

Classes defined in this module provide an interface to extracting data from
SmallData files and analyzing it.

Classes:
    AnalyzeSmallDataXSS(Task): Analyze scattering data for a single detector in
        a SmallData file.
"""

__all__ = ["AnalyzeSmallDataXSS"]
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
from lute.io.models.base import *
from lute.tasks.task import *
from lute.tasks.dataclasses import ElogSummaryPlots
from lute.tasks.math import gaussian, sigma_to_fwhm


logger: logging.Logger = logging.getLogger(__name__)

if __debug__:
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.INFO)


class AnalyzeSmallDataXSS(Task):
    """Task to analyze XSS profiles stored in a SmallData HDF5 file."""

    _LARGE_DETNAMES: List[str] = ["epix10k2M", "Rayonix", "Jungfrau4M"]
    _SMALL_DETNAMES: List[str] = ["epix_1", "epix_2"]

    def __init__(self, *, params: TaskParameters, use_mpi: bool = True) -> None:
        super().__init__(params=params, use_mpi=use_mpi)
        hv.extension("bokeh")
        pn.extension()
        self._mpi_comm: MPI.Intracomm = MPI.COMM_WORLD
        self._mpi_rank: int = self._mpi_comm.Get_rank()
        self._mpi_size: int = self._mpi_comm.Get_size()
        try:
            self._smd_h5: h5py.File = h5py.File(self._task_parameters.smd_path, "r")
        except Exception:
            if self._mpi_rank == 0:
                logger.debug(f"Failed to open file: {self._task_parameters.smd_path}!")
            self._mpi_comm.Barrier()
            sys.exit(-1)

        self._events_per_rank: np.ndarray[np.int]
        self._start_indices_per_rank: np.ndarray[np.int]
        if self._mpi_rank == 0:
            self._total_num_events: int = len(self._smd_h5["event_time"][()])
            quotient: int
            remainder: int
            quotient, remainder = divmod(self._total_num_events, self._mpi_size)
            self._events_per_rank = np.array(
                [
                    quotient + 1 if rank < remainder else quotient
                    for rank in range(self._mpi_size)
                ]
            )
            self._start_indices_per_rank = np.zeros(self._mpi_size, dtype=np.int)
            self._start_indices_per_rank[1:] = np.cumsum(self._events_per_rank[:-1])
        else:
            self._events_per_rank = np.zeros(self._mpi_size, dtype=np.int)
            self._start_indices_per_rank = np.zeros(self._mpi_size, dtype=np.int)
        self._mpi_comm.Bcast(self._events_per_rank, root=0)
        self._mpi_comm.Bcast(self._start_indices_per_rank, root=0)
        self._xss_detname: str
        if self._task_parameters.xss_detname is None:
            for key in self._smd_h5.keys():
                if key in AnalyzeSmallData._LARGE_DETNAMES:
                    self._xss_detname = key
                    logger.debug(f"Assuming XSS detector is: {key}.")
                    break
            else:
                logger.error("No XSS detname provided and could not determine detname!")
                self._mpi_comm.Barrier()
                sys.exit(-1)
        else:
            self._xss_detname = self._task_parameters.xss_detname

        # Basic RANK-LOCAL variables - Each rank holds a subset of data
        ################################################################
        self._num_events: int = self._events_per_rank[self._mpi_rank]
        self._start_idx: int = self._start_indices_per_rank[self._mpi_rank]
        self._stop_idx: int = self._start_idx + self._num_events
        logger.debug(
            f"Rank {self._mpi_rank}: Processing {self._num_events} events from "
            f"event {self._start_idx}."
        )

        # Generic filtering variables
        self._filter_dict: Dict[str, np.ndarray[np.float64]] = {}

        # Azimuthal integration variables
        self._az_int: np.ndarray[np.float64, np.float64, np.float64]
        self._q_vals: np.ndarray[np.float64]
        self._phi_vals: np.ndarray[np.float64]
        self._integrated_intensity: np.ndarray[np.float64]
        self._xray_intensity: np.ndarray[np.float64]

        # Scan vars
        self._scan_var_name: Optional[str] = None
        self._scan_values: np.ndarray[np.float64]

    def _pre_run(self) -> None:
        self._extract_data()

    def _run(self) -> None:
        diff: np.ndarray[np.float64]
        bins: np.ndarray[np.float64]
        bins, diff, laser_on = self._calc_binned_difference_xss()

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
            plots: pn.Tabs = self.plot_all(laser_on, bins, diff, name)
            plot_display_name: str
            run: int = int(self._task_parameters.lute_config.run)
            exp_run: str = f"{run:04d}_{name}_XSS"
            if "lens" in name:
                plot_display_name = f"lens_scans/{exp_run}"
            else:
                plot_display_name = f"time_scans/{exp_run}"

            self._result.payload = ElogSummaryPlots(plot_display_name, plots)

    def _post_run(self) -> None: ...

    # Azimuthal integration data
    # Extracts: azav (2D), q_values (1D), phi values (1D)
    ############################################################################
    def _extract_data(self) -> None:
        """Setup up stored attributes by taking data from the smalldata hdf5 file."""

        logger.debug("XSS Analysis: extract_data")

        self._extract_az_int()
        try:
            self._xray_intensity = self._smd_h5[self._task_parameters.ipm_var][
                self._start_idx : self._stop_idx
            ]
        except KeyError as e:
            logger.error("ipm data not found! Check config!")
        self._integrated_intensity = np.nansum(self._az_int, axis=(1, 2))
        self._setup_std_filters()

        if isinstance(self._task_parameters.scan_var, str):
            scan_var: str = self._task_parameters.scan_var
            try:
                self._scan_values = self._smd_h5[f"scan/{scan_var}"][
                    self._start_idx : self._stop_idx
                ]
                self._scan_var_name = scan_var
            except KeyError as e:
                logger.debug(f"Scan variable {scan_var} not found!")
        elif isinstance(self._task_parameters.scan_var, list):
            for scan_var in self._task_parameters.scan_var:
                try:
                    self._scan_values = self._smd_h5[f"scan/{scan_var}"][
                        self._start_idx : self._stop_idx
                    ]
                    self._scan_var_name = scan_var
                    break
                except KeyError as e:
                    logger.debug(f"Scan variable {scan_var} not found!")
                    continue
        if not hasattr(self, "_scan_values"):
            # Create a set of scan values if none of the above scan
            # variables are found, either lxt_fast or linear set.
            try:
                self._scan_values = self._smd_h5["enc/lasDelay"][
                    self._start_idx : self._stop_idx
                ]
                self._scan_var_name = "lxt_fast"
                logger.debug("Using scan variable lxt_fast")
            except KeyError as e:
                logger.debug("Scan variable lxt_fast not found!")
                self._scan_values = np.linspace(
                    self._start_idx, self._stop_idx, self._num_events
                )
                logger.debug(
                    "No scans found, using linearly spaced data for _scan_values."
                )

    def _extract_az_int(self) -> None:
        """Try to extract the azimuthal integration data from HDF5 file.

        This internal method will search first to see if a detector has been
        provided as the scattering detector. If not, it will attempt to guess
        which detector to use. It will attempt to extract both the internal
        integration data and PyFAI integrated data (which have different
        interfaces). If both are present, it will only extract PyFAI data.
        """
        detname: str = self._xss_detname
        try:
            self._extract_az_int_smd_internal(detname)
            logger.debug(
                f"Extracted internal azimuthal integration data for {detname}."
            )
        except Exception:
            try:
                self._extract_az_int_pyfai(detname)
                logger.debug(
                    f"Extracted PyFAI azimuthal integration data for {detname}."
                )
            except Exception:
                logger.error("No azimuthal integration data found.")
                self._mpi_comm.Barrier()
                sys.exit(-1)

    def _extract_az_int_smd_internal(self, detname: str) -> None:
        """Extract stored data from SmallData's azimuthal integration algorithm.

        Args:
            detname (str): The detector name to extract data for.
        """

        self._az_int = self._smd_h5[f"{detname}/azav_azav"][
            self._start_idx : self._stop_idx
        ]  # 3D
        self._q_vals = self._smd_h5[f"UserDataCfg/{detname}/azav__azav_q"][()]
        self._phi_vals = self._smd_h5[f"UserDataCfg/{detname}/azav__azav_phiVec"][()]

    def _extract_az_int_pyfai(self, detname: str) -> None:
        """Extract stored data from PyFAI's azimuthal integration algorithm.

        Args:
            detname (str): The detector name to extract data for.
        """

        self._az_int = self._smd_h5[f"{detname}/pyfai_azav"][
            self._start_idx : self._stop_idx
        ]  # 3D
        self._q_vals = self._smd_h5[f"{detname}/pyfai_q"][()][0]
        self._phi_vals = self._smd_h5[f"{detname}/pyfai_az"][()][0]

    def _setup_std_filters(self) -> None:
        """Setup the individual standard event filters.

        Sets up light status (X-ray and laser) filters, and adjustable filters
        based on, e.g., thresholds.
        """
        # For filtering based on event type
        self._filter_dict["xray on"] = (
            self._smd_h5["lightStatus/xray"][self._start_idx : self._stop_idx] == 1
        )
        self._filter_dict["laser on"] = (
            self._smd_h5["lightStatus/laser"][self._start_idx : self._stop_idx] == 1
        )

        self._filter_dict["xray off"] = (
            self._smd_h5["lightStatus/xray"][self._start_idx : self._stop_idx] == 0
        )
        self._filter_dict["laser off"] = (
            self._smd_h5["lightStatus/laser"][self._start_idx : self._stop_idx] == 0
        )

        # Create scattering and ipm thresholds for first time
        self._update_filters()

    def _update_filters(self) -> None:
        """Update stored event filters that are based on adjustable parameters.

        E.g. update the minimum scattering intensity to use in analyses.
        """
        scattering_thresh: float = self._task_parameters.thresholds.min_Iscat
        ipm_thresh: float = self._task_parameters.thresholds.min_ipm
        self._filter_dict["total scattering"] = (
            self._integrated_intensity > scattering_thresh
        )
        if hasattr(self, "_xray_intensity"):
            self._filter_dict["ipm"] = self._xray_intensity > ipm_thresh
        else:
            self._filter_dict["ipm"] = np.ones([self._num_events], dtype=bool)

    def _calc_norm_by_qrange(
        self, q_limits: Tuple[float, float] = (0.9, 3.5)
    ) -> np.ndarray[np.float64]:
        """Calculate a normalization factor by averaging over a range of Q values.

        Args:
            q_limits (Tuple[float, float]): The lower and upper limit of the
                Q-range to average.

        Returns:
            norm (np.ndarray[np.float64]): The 2D norm with shape
                (num_events, num_phi)
        """
        norm_range: np.array = self._q_vals > q_limits[0]
        norm_range &= self._q_vals < q_limits[1]
        return np.nanmean(self._az_int[..., norm_range], axis=-1)

    def _calc_norm_by_max(self) -> np.ndarray[np.float64]:
        """Calculate a normalization factor by taking the maximum of each profile.

        Returns:
        norm (np.ndarray[np.float64]): The 1D norm with shape (num_events)
        """
        return np.nanmax(self._az_int, axis=(1, 2))

    def _aggregate_filters(
        self, filter_vars: str = ("xray on, laser on, total scattering, ipm")
    ) -> np.ndarray[np.bool_]:
        """Combine a number of possible data filters.

        Takes a string of filters to be applied in the order specified. The
        order matters for some filters, e.g. if "percentage" is selected it
        should be applied after total scattering. See below for a list of
        filters.

        Possible filters:
            "xray on": Take shots where X-rays are on,
            "xray off": Take shots where X-rays are off,
            "laser on": Take shots where laser is on,
            "laser off": Take shots where laser is off,
            "ipm": Take shots with X-ray intensity above a value (ipm value),
            "total scattering": Take shots with minimum integrated scattering,
            "percentage": Extract this percentage of data centered on the max
                scattering intensity. Should be applied after total scatteirng.
                E.g. Include 50% of shots centered around the Q-value with
                maximum scattering.

        Args:
            filter_vars (str): A string of comma-seperated filters to apply,
                taken from the list above. E.g. "xray on, laser on".

        Returns:
            total_filter (np.ndarray[np.bool]): A 1D boolean array with a shape
                of (num_events) which can be used to index and filter a data array.
        """
        logger.debug("XSS Analysis: aggregate_filters")
        filters: List[str] = [f.strip() for f in filter_vars.split(",")]
        total_filter: np.ndarray = np.ones([self._num_events], dtype=bool)

        for data_filter in filters:
            logger.debug(f"Filter: {data_filter}")
            try:
                total_filter &= self._filter_dict[data_filter]
            except KeyError as e:
                logger.debug(
                    f"Unrecognized filter requested: {data_filter}. "
                    "Ignoring and continuing processing."
                )
                continue

        return total_filter

    def _calc_dark_mean(
        self, profiles: np.ndarray[np.float64]
    ) -> np.ndarray[np.float64]:
        """Calculate the dark (X-ray off) mean of a set of XSS profiles.

        Args:
            profiles (np.ndarray[np.float64]): Set of profiles to calculate the
                dark mean from. Shape: (n_events, q_bins)
        Returns:
            dark_mean (np.ndarray[np.float64]): The calculated dark mean.
                Shape: (q_bins)
        """
        dark_mean: np.ndarray[np.float64]
        try:
            dark_mean = np.nanmean(profiles[self._filter_dict["xray off"]], axis=0)
            dark_mean = self._mpi_comm.allreduce(dark_mean, op=MPI.SUM)
            dark_mean /= self._mpi_size
        except KeyError:
            logger.debug(
                "No X-ray off event information for filtering. "
                "Dark mean will be all zeros."
            )
            dark_mean = np.zeros([self._num_events])
        return dark_mean

    def _calc_scan_bins(self) -> np.ndarray[np.float64]:
        """Calculate a set of scan bins.

        Returns:
            scan_bins (np.ndarray[np.float64]): 1D set of scan bins.
        """
        # Create a full set of scan values
        self._all_scan_values: np.ndarray[np.float64] = np.zeros(
            np.sum(self._events_per_rank), dtype=np.float64
        )
        self._mpi_comm.Gatherv(
            self._scan_values,
            [
                self._all_scan_values,
                self._events_per_rank,
                self._start_indices_per_rank,
                MPI.DOUBLE,
            ],
            root=0,
        )
        logger.debug("XSS Analysis: time_bins")
        scan_bins: np.ndarray[np.float64]
        if self._mpi_rank == 0:
            if self._scan_var_name is not None and "lxt_fast" in self._scan_var_name:
                nbins: int = 70  # 200
                _, edges = np.histogram(np.unique(self._all_scan_values), bins=nbins)
                scan_bins = edges
            else:
                scan_bins = np.unique(self._all_scan_values)
        else:
            scan_bins = None
        scan_bins = self._mpi_comm.bcast(scan_bins, root=0)
        return scan_bins

    def _calc_binned_difference_xss(
        self,
    ) -> Tuple[np.ndarray[np.float], np.ndarray[np.float64], np.ndarray[np.float64]]:
        """Calculate the binned difference.

        Calculates the 1D difference scattering for each bin of a scan variable.
        Final difference shape is 2D: (q_bins, scan_bins). Also returns bins and
        the laser on profiles.

        Returns:
            bins (np.ndarray[np.float64]): 1D array of scan bins used.

            diff (np.ndarray[np.float64]): 2D binned difference scattering of shape
                (q_bins, scan_bins)

            laser_on (np.ndarray[np.float64]): 2D laser on scattering profiles
                of shape (n_events_las_on, q_bins)
        """
        logger.debug("XSS Analysis: profiles_1d")
        profiles: np.ndarray[np.float64, np.float64] = np.nansum(self._az_int, axis=1)
        dark_mean: np.ndarray[np.float64] = self._calc_dark_mean(profiles)
        if len(np.unique(dark_mean)) > 1:
            # Can be len == 1 if all nan
            profiles -= dark_mean
        norm: np.ndarray[np.float64] = self._calc_norm_by_qrange()
        profiles = (profiles.T / np.nanmean(norm, axis=-1).T).T
        del dark_mean
        del norm

        filter_las_on: np.ndarray[np.float64] = self._aggregate_filters()
        filter_las_off: np.ndarray[np.float64] = self._aggregate_filters(
            filter_vars="xray on, laser off, ipm, total scattering"
        )

        bins: np.ndarray[np.float64] = self._calc_scan_bins()
        binned_on: np.ndarray[np.float64] = np.zeros((len(self._q_vals), len(bins)))
        binned_off: np.ndarray[np.float64] = np.zeros((len(self._q_vals), len(bins)))
        scanvals_las_on: np.ndarray[np.float64] = self._scan_values[filter_las_on]
        scanvals_las_off: np.ndarray[np.float64] = self._scan_values[filter_las_off]

        idx: int
        scan_bin: float
        for idx, scan_bin in enumerate(bins):
            if self._scan_var_name is not None and "lxt_fast" in self._scan_var_name:
                if idx == len(bins) - 1:
                    continue
                mask_on: np.ndarray[np.bool_] = (scanvals_las_on >= scan_bin) * (
                    scanvals_las_on < bins[idx + 1]
                )
                mask_off: np.ndarray[np.bool_] = (scanvals_las_off >= scan_bin) * (
                    scanvals_las_off < bins[idx + 1]
                )
                binned_on[:, idx] = np.nanmean(profiles[filter_las_on][mask_on], axis=0)
                binned_off[:, idx] = np.nanmean(
                    profiles[filter_las_off][mask_off], axis=0
                )
            else:
                binned_on[:, idx] = np.nanmean(
                    profiles[filter_las_on][(scanvals_las_on == scan_bin)], axis=0
                )
                binned_off[:, idx] = np.nanmean(
                    profiles[filter_las_off][(scanvals_las_off == scan_bin)], axis=0
                )
        diff: np.ndarray[np.float64] = np.nan_to_num(binned_on) - np.nan_to_num(
            binned_off
        )
        return bins, diff, profiles[filter_las_on]

    def _find_solvent_argmax(self, corrected_profile: np.ndarray) -> int:
        """Find the index of the solvent ring maximum.

        Currently just a hack around SciPy find_peaks.

        Args:
            corrected_profile (np.ndarray[np.float64]): 1D normalized and dark
                mean corrected, laser on, scattering profile.

        Returns:
            peak_idx (int): The index where the solvent maximum is located.
        """
        res: Tuple[np.ndarray, Dict[str, np.ndarray]] = find_peaks(corrected_profile, 1)
        try:
            peak_indices: np.ndarray = res[0]
            peak_heights: np.ndarray = res[1]["peak_heights"]
        except KeyError as e:
            logger.debug("No peaks found")
            return np.argmax(corrected_profile)

        peak_idx: int = peak_indices[
            np.argmax(peak_heights)
        ]  # peak_indices[0]  # peak_indices[peak_select - 1]
        self._solvent_peak_idx = peak_idx
        return peak_idx

    def _fit_overlap(
        self,
        laser_on: np.ndarray[np.float64],
        bins: np.ndarray[np.float64],
        diff: np.ndarray[np.float64],
        guess: Optional[List[float]] = None,
    ) -> Tuple[np.ndarray[np.float64], np.ndarray[np.float64], np.ndarray[np.float64]]:
        """Fit overlap based on scattering difference signal to a Gaussian.

        Args:
            laser_on (np.ndarray[np.float64]): 1D corrected average laser on
                scattering profile.

            bins (np.ndarray[np.float64]): 1D set of bins used for difference
                signal.

            diff (np.ndarray[np.float64]): 2D difference signal of shape
                (q_bins, scan_bins).

            guess (Optional[List[float]]): A list of initial parameter guesses
                for Gaussian fit in the order (amplitude, x0, sigma, background
                offset)

        Returns:
            raw_curve (np.ndarray[np.float64]): 1D difference slice at a specific
                Q value used for calculating the overlap.

            opt (np.ndarray[np.float64]): Optimized parameters.

            res (np.ndarray[np.float64]): Covariances.
        """
        logger.debug("XSS Analysis: fit_overlap")
        Tuple[np.ndarray[np.float64], np.ndarray[np.float64], np.ndarray[np.float64]]
        max_pt: int = self._find_solvent_argmax(laser_on)
        try:
            idx_peak_q: int = max_pt + 10
        except IndexError as e:
            logger.debug(f"{e}: No non-nan values found.")
            idx_peak_q: int = 15

        raw_curve: np.ndarray = diff[idx_peak_q]
        if not guess:
            max_val: int = raw_curve.max()
            min_val: int = raw_curve.min()
            # guess = [amplitude, center, stddev, bkgnd]
            guess = [0, 0, 0, 0]
            x0: float
            if max_val > np.abs(min_val):
                x0 = bins[raw_curve.argmax()]
                guess[0] = max_val
            else:
                x0 = bins[raw_curve.argmin()]
                guess[0] = min_val
            guess[1] = x0
            guess[2] = np.abs(bins[-1] - bins[0]) / 20
            guess[3] = np.min(raw_curve)
        try:
            opt, res = curve_fit(gaussian, bins, raw_curve, p0=guess, maxfev=999999)
        except ValueError as e:
            logger.debug(f"{e}:\n No non-nan values...")
            opt = guess
            res = None

        return raw_curve, opt, res

    def _fit_convolution_fwhm(
        self, trace: np.ndarray, bins: np.ndarray[np.float_]
    ) -> float:
        """Calculate the FWHM of a convolution signal.

        Args:
            trace (np.ndarray[np.float64]): 1D convolution trace.

            bins (np.ndarray[np.float64]): 1D set of bins used.

        Returns:
            fwhm (float): Calculated FWHM.
        """
        from scipy.interpolate import UnivariateSpline

        x = np.linspace(0, len(trace) - 1, len(trace))
        spline = UnivariateSpline(x, (trace - np.max(trace) / 2), s=0)
        roots = spline.roots()
        if len(roots) == 2:
            lb, rb = roots
            fwhm = rb - lb
        else:
            # bins: np.ndarray = self.time_bins[0]
            max_val: int = trace.max()
            min_val: int = trace.min()
            # guess = [amplitude, center, stddev, bkgnd]
            guess: List[Union[float, int]] = [0, 0, 0, 0]
            x0: Union[float, int]
            if max_val > np.abs(min_val):
                x0 = bins[trace.argmax()]
                guess[0] = max_val
            else:
                x0 = bins[trace.argmin()]
                guess[0] = min_val
            guess[1] = x0
            guess[2] = np.abs(bins[-1] - bins[0]) / 20
            guess[3] = np.min(trace)
            opt, res = curve_fit(
                gaussian, bins, np.nan_to_num(trace), p0=guess, maxfev=999999
            )
            fwhm = sigma_to_fwhm(opt[2])
        return fwhm

    def _convolution_fit(
        self,
        laser_on: np.ndarray[np.float64],
        bins: np.ndarray[np.float64],
        diff: np.ndarray[np.float64],
    ) -> Tuple[np.ndarray, np.ndarray, int, float]:
        """Fits a time scan through convolution with Heaviside kernel.

        Args:
            laser_on (np.ndarray[np.float64]): 1D corrected average laser on
                scattering profile.

            bins (np.ndarray[np.float64]): 1D set of bins used for difference
                signal.

            diff (np.ndarray[np.float64]): 2D difference signal of shape
                (q_bins, scan_bins).

        Returns:
            raw_curve (np.ndarray[np.float64]): 1D difference slice at a specific
                Q value used for calculating the overlap.

            trace (np.ndarray[np.float64]): 1D convolution trace.

            center (int): The index of the center from the convolution.

            fwhm (float): Width of the convlution signal (fwhm).
        """
        from scipy.signal import fftconvolve

        logger.debug("XSS Analysis: convolution_fit")
        results: List[Tuple] = []
        max_pt: int = self._find_solvent_argmax(laser_on)
        raw_curve: np.ndarray = np.nan_to_num(diff[max_pt + 10])
        npts: int = len(raw_curve)
        kernel: np.ndarray = np.zeros([npts])
        kernel[: npts // 2] = 1
        trace: np.ndarray = fftconvolve(raw_curve, kernel, mode="same")
        center: int = trace.argmax()
        fwhm: float = self._fit_convolution_fwhm(trace, bins)
        return raw_curve, trace, center, fwhm

    def plot_avg_xss(
        self,
        laser_on: np.ndarray[np.float64],
        q_vals: np.ndarray[np.float64],
        phis: np.ndarray[np.float64],
    ) -> plt.Figure: ...

    def plot_difference_hv(
        self,
        bins: np.ndarray[np.float64],
        q_vals: np.ndarray[np.float64],
        diff: np.ndarray[np.float64],
        scan_var_name: str,
    ) -> pn.GridSpec:
        """Plot the binned difference scattering.

        Args:
            bins (np.ndarray[np.float64]): 1D set of bins used for difference
                signal.

            q_vals (np.ndarray[np.float64]): 1D set of Q bins.

            diff (np.ndarray[np.float64]): 2D difference signal of shape
                (q_bins, scan_bins).

            scan_var_name (str): Name of the scan variable (for titles, etc.).

        Returns:
            plot (pn.GridSpec): Plotted binned difference.
        """
        grid: pn.GridSpec = pn.GridSpec(name="Difference Profiles")
        xdim: hv.core.dimension.Dimension = hv.Dimension(
            ("scan_var", f"Scan Var {scan_var_name}")
        )
        ydim: hv.core.dimension.Dimension = hv.Dimension(("Q", "Q"))
        diff_img: hv.Image = hv.Image(
            (bins, q_vals, diff.T),
            kdims=[xdim, ydim],
        ).opts(shared_axes=False)
        contours = diff_img + hv.operation.contours(diff_img, levels=5)
        contours.opts(
            hv.opts.Contours(cmap="fire", colorbar=True, tools=["hover"], width=325)
        ).opts(shared_axes=False)
        grid[:2, :2] = contours
        xdim: hv.core.dimension.Dimension = hv.Dimension(("Q", f"Q"))
        ydim: hv.core.dimension.Dimension = hv.Dimension(("dS", "dS"))
        diff_curves: hv.Overlay
        # diff_curves = hv.Curve((q_vals, diff[:, 0]))
        # for i, _ in enumerate(bins):
        #    if i == 0:
        #        continue
        #    else:
        #        diff_curves *= hv.Curve(
        #            (
        #                q_vals,
        #                diff[:, i],
        #            )
        #        ).opts(xlabel=xdim.label, ylabel=ydim.label)

        # grid[2:4, :] = diff_curves.opts(shared_axes=False)
        return grid

    def plot_overlap_fit(
        self,
        laser_on: np.ndarray[np.float64],
        bins: np.ndarray[np.float64],
        diff: np.ndarray[np.float64],
    ) -> plt.Figure:
        """Plot the overlap fit to a slice of the binned difference.

        Args:
            laser_on (np.ndarray[np.float64]): 1D corrected average laser on
                scattering profile.

            bins (np.ndarray[np.float64]): 1D set of bins used for difference
                signal.

            diff (np.ndarray[np.float64]): 2D difference signal of shape
                (q_bins, scan_bins).

        Returns:
            plot (plt.Figure): Plotted overlap fit.
        """
        logger.debug("XSS Analysis: plot_overlap_fit")
        fig, ax = plt.subplots(1, 1, figsize=(6, 3), dpi=200)
        if self._scan_var_name is not None and "lxt" in self._scan_var_name:
            raw_curve: np.ndarray[np.float64]
            trace: np.ndarray[np.float64]
            center: int
            fwhm: float
            raw_curve, trace, center, fwhm = self._convolution_fit(laser_on, bins, diff)
            msg: str = (
                f"Scan Var: {self._scan_var_name}\n"
                f"Overlap Position: {bins[center]}\n"
                f"FWHM: {fwhm}"
            )
            logger.info(msg)
            ax.plot(bins, raw_curve, label="Raw")
            ax2 = ax.twinx()
            ax2.plot(bins, trace, color="orange", label="Convolution")
            ax.set_title(msg)
            ax.set_xlabel(f"Scan Variable ({self._scan_var_name})")
            ax.set_ylabel(r"$\Delta$S")
            plt.legend(loc=0, frameon=False)
        else:
            raw_curve: np.ndarray[np.float64]
            opt: np.ndarray[np.float64]
            raw_curve, opt, _ = self._fit_overlap(laser_on, bins, diff)
            msg: str = (
                f"Scan Var: {self._scan_var_name}\n"
                f"Overlap Position: {opt[1]}\n"
                f"FWHM: {sigma_to_fwhm(opt[2])}"
            )
            logger.info(msg)
            ax.plot(bins, raw_curve)
            ax.plot(bins, gaussian(bins, *opt))
            ax.set_title(msg)
            ax.set_xlabel(f"Scan Variable ({self._scan_var_name})")
            ax.set_ylabel(r"$\Delta$S")
        fig.tight_layout()
        return fig

    def plot_all(
        self,
        laser_on: np.ndarray[np.float64],
        bins: np.ndarray[np.float64],
        diff: np.ndarray[np.float64],
        scan_var_name: str,
    ) -> pn.Tabs:
        """Plot all relevant plots.

        Args:
            laser_on (np.ndarray[np.float64]): 1D corrected average laser on
                scattering profile.

            bins (np.ndarray[np.float64]): 1D set of bins used for difference
                signal.

            diff (np.ndarray[np.float64]): 2D difference signal of shape
                (q_bins, scan_bins).

            scan_var_name (str): Name of the scan variable (for titles, etc.).

        Returns:
            plot (pn.Tabs): All plots in separated tabs in a pn.Tabs object.
        """
        logger.debug("XSS Analysis: plot_all")
        # avg_fig = self.plot_avg_xss()
        overlap_fig = self.plot_overlap_fit(laser_on, bins, diff)

        # avg_grid = pn.GridSpec(name="TR X-ray Scattering")
        # avg_grid[:2, :2] = avg_fig

        diff_grid = self.plot_difference_hv(laser_on, bins, diff, scan_var_name)
        overlap_grid = pn.GridSpec(name="Overlap Fit")
        overlap_grid[:2, :2] = overlap_fig

        tabbed_display = pn.Tabs(diff_grid)
        tabbed_display.append(overlap_grid)
        # tabbed_display.append(avg_grid)

        return tabbed_display
