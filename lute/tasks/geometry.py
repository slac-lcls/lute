"""Tasks for working with and optimizing geometry.

Classes:
"""

__all__ = ["OptimizeAgBhGeometryExhaustive"]
__author__ = "Gabriel Dorlhiac"

import os
import logging
import itertools
from typing import List, Optional, Tuple, Union, Any

import h5py
import holoviews as hv
import lmfit
import numpy as np
import panel as pn
import psana
from mpi4py import MPI
from scipy import sparse
from scipy.signal import find_peaks

from lute.io.models.base import TaskParameters
from lute.execution.logging import get_logger
from lute.tasks._geometry import geometry_optimize_residual
from lute.tasks.task import Task
from lute.tasks.dataclasses import TaskStatus

logger: logging.Logger = get_logger(__name__)


class OptimizeAgBhGeometryExhaustive(Task):
    """Task to perform geometry optimization."""

    def __init__(self, *, params: TaskParameters, use_mpi: bool = True) -> None:
        super().__init__(params=params, use_mpi=use_mpi)
        hv.extension("bokeh")
        pn.extension()
        self._mpi_comm: MPI.Intracomm = MPI.COMM_WORLD
        self._mpi_rank: int = self._mpi_comm.Get_rank()
        self._mpi_size: int = self._mpi_comm.Get_size()

        self._mask: np.ndarray[bool]
        if isinstance(self._task_parameters.mask, str):
            is_valid: bool
            dtype: Optional[str]
            is_valid, dtype = self._check_if_path_and_type(self._task_parameters.mask)
            if is_valid and dtype == "numpy":
                self._mask = np.load(self._task_parameters.mask)

    def _check_if_path_and_type(self, string: str) -> Tuple[bool, Optional[str]]:
        """Check if a string is a valid path and determine the filetype.

        Args:
            string (str): The string that may be a file path.

        Returns:
            is_valid_path (bool): If it is a valid path.

            powder_type (Optional[str]): If is_valid_path, the file type.
        """
        is_valid_path: bool = False
        powder_type: Optional[str] = None
        if os.path.exists(string):
            is_valid_path = True
        else:
            return is_valid_path, powder_type
        try:
            with h5py.File(string) as h5:
                powder_type = "smd"
                is_valid_path = True

            return is_valid_path, powder_type
        except:
            ...

        try:
            np.load(string)
            powder_type = "numpy"
            is_valid_path = True
            return is_valid_path, powder_type
        except ValueError:
            ...

        return is_valid_path, powder_type

    def _extract_powder(self, powder_path: str) -> Optional[np.ndarray[np.float64]]:
        powder: Optional[np.ndarray[np.float64]] = None
        if isinstance(powder_path, str):
            is_valid: bool
            dtype: Optional[str]
            is_valid, dtype = self._check_if_path_and_type(powder_path)
            if is_valid and dtype == "numpy":
                powder = np.load(powder_path)
            elif is_valid and dtype == "smd":
                h5: h5py.File
                with h5py.File(powder_path) as h5:
                    unassembled: np.ndarray[np.float64] = h5[
                        f"Sums/{self._task_parameters.detname}_calib"
                    ][()]
                    if unassembled.shape == 2:
                        # E.g. Rayonix
                        powder = unassembled
                    else:
                        ix: np.ndarray[np.uint64] = h5[
                            f"UserDataCfg/{self._task_parameters.detname}/ix"
                        ][()]
                        iy: np.ndarray[np.uint64] = h5[
                            f"UserDataCfg/{self._task_parameters.detname}/iy"
                        ][()]

                        ix -= np.min(ix)
                        iy -= np.min(iy)

                        if (
                            unassembled.flatten().shape != ix.flatten().shape
                            or unassembled.flatten().shape != iy.flatten().shape
                        ):
                            raise RuntimeError(
                                "Shapes of detector image and pixel coordinates do not match!"
                            )

                        out_shape: Tuple[int, int] = (
                            int(np.max(ix) + 1),
                            int(np.max(iy) + 1),
                        )

                        powder = np.asarray(
                            sparse.coo_matrix(
                                (unassembled.flatten(), (ix.flatten(), iy.flatten())),
                                shape=out_shape,
                            ).todense()
                        )
        return powder

    def _get_pixel_size_and_wavelength(
        self, ds: psana.DataSource, det: psana.Detector
    ) -> Tuple[float, float]:
        """Extract pixel size in mm and wavelength in Angstroms."""
        pixel_size: float
        if self._task_parameters.detname.lower() == "rayonix":
            pixel_size = ds.env().configStore().get(psana.Rayonix.ConfigV2).pixelWidth()
        else:
            pixel_size = det.pixel_size(ds.env())
        pixel_size /= 1e3

        wavelength: float = ds.env().epicsStore().value("SIOC:SYS0:ML00:AO192") * 10
        return pixel_size, wavelength

    def _estimate_distance(self) -> float:
        exp: str = self._task_parameters.lute_config.experiment
        run: Union[int, str] = self._task_parameters.lute_config.run
        ds: psana.DataSource = psana.DataSource(f"exp={exp}:run={run}")
        det: psana.Detector = psana.Detector(self._task_parameters.detname, ds.env())
        return -1 * np.mean(det.coords_z(run)) / 1e3

    def _initial_image_center(
        self, powder: np.ndarray[np.float64]
    ) -> np.ndarray[np.float64]:
        return np.array(powder.shape) / 2.0

    def _center_guesses(
        self, powder: np.ndarray[np.float64]
    ) -> List[Tuple[float, float]]:
        """Return starting beam center points based on dx/dy parameters."""
        initial_center: np.ndarray[np.float64] = self._initial_image_center(powder)
        dx: Tuple[int, int, int] = self._task_parameters.dx
        dy: Tuple[int, int, int] = self._task_parameters.dy
        x_offsets: np.ndarray[np.float64] = np.linspace(dx[0], dx[1], dx[2])
        y_offsets: np.ndarray[np.float64] = np.linspace(dy[0], dy[1], dy[2])
        center_offsets: List[Tuple[float, float]] = list(
            itertools.product(x_offsets, y_offsets)
        )
        new_centers: List[Tuple[float, float]] = [
            (initial_center[0] + offset[0], initial_center[1] + offset[1])
            for offset in center_offsets
        ]
        return new_centers

    def _radial_profile(
        self,
        powder: np.ndarray[np.float64],
        mask: np.ndarray[np.float64],
        center: Tuple[float, float],
        threshold: float = 10.0,
        filter_profile: bool = False,
        filter_order: int = 2,
        filter_threshold: float = 0.25,
    ) -> np.ndarray[np.float64]:
        """Compute the radial intensity profile of an image.

        Args:
            powder (np.ndarray[np.float64]): 2-D assembled powder image.

            mask (np.ndarray[np.float64]): Corresponding binary mask for the
                powder image.

            center (Tuple[float, float]): Beam center in the image, in pixels.

            threshold (float): Default: 10. Below this intensity set the
                intensity of the radial profile to 0.

            filter_profile (bool): Default: False. If True apply a lowpass
                Butterworth filter to the profile.

            filter_order (int): Default: 2. If applying a filter, the order of
                the filter.

            filter_threshold (float): Default: 0.25. Critical frequency for the
                Butterworth filter, if applying it.

        Returns:
            radial_profile (np.ndarray[np.float64]): 1-D array of peak intensities
                for an azimuthally integrated powder image.
        """
        y: np.ndarray[np.int64]
        x: np.ndarray[np.int64]
        y, x = np.indices(powder.shape)
        radius_map: np.ndarray[np.float64] = (
            (x - center[0]) ** 2 + (y - center[1]) ** 2
        ) ** 0.5

        # r = get_radius_map(data.shape, center=center)
        if mask is not None:
            radius_map = np.where(mask == 1, radius_map, 0)

        radius_map_int: np.ndarray[np.int64] = radius_map.astype(np.int64)
        tbin: np.ndarray[np.float64] = np.bincount(
            radius_map_int.ravel(), powder.ravel()
        )
        nr: np.ndarray[np.int64] = np.bincount(radius_map_int.ravel())
        radial_profile: np.ndarray[np.float64] = np.divide(
            tbin, nr, out=np.zeros(nr.shape[0]), where=nr != 0
        )
        if filter_profile:
            from scipy.signal import sosfiltfilt, butter

            sos = butter(filter_order, filter_threshold, output="sos")
            radial_profile = sosfiltfilt(sos, radial_profile)
            radial_profile[radial_profile < threshold] = 0
        return radial_profile

    def _calc_and_score_ideal_rings(
        self, q_peaks: np.ndarray[np.float64]
    ) -> Tuple[np.ndarray[np.float64], float]:
        """Score inter-peak distances in q-space based on known behenate pattern.

        Relies on the equidistance of peaks in q-space for silver behenate powder.

        Args:
            q_peaks (np.ndarray[np.float64]): Positions of powder peaks in q-space
                (inverse Angstrom). 1 dimensional.

        Returns:
            rings (np.ndarray[np.float64]): Predicted positions of peaks based on
                the best fit q-spacing. 1 dimensional.

            final_score (float): Final score calculated from residuals associated
                with each q-spacin.
        """
        # Q-spacings between behenate peaks
        delta_qs: np.ndarray[np.float64] = np.arange(0.01, 0.03, 0.00005)
        order_max: int = 13
        qround: np.ndarray[np.int64] = np.round(q_peaks, 5)
        scores: List[float] = []
        dq: float
        for dq in delta_qs:
            order: np.ndarray[np.float64] = qround / dq
            remainder: np.ndarray[np.float64] = np.minimum(
                qround % dq, np.abs(dq - (qround % dq))
            )
            score: np.ndarray[np.float64] = (
                np.mean(remainder[np.where(order < order_max)]) / dq
            )  # %mod helps prevent half periods from scoring well
            scores.append(score)
        deltaq_current = delta_qs[np.argmin(scores)]
        rings: np.ndarray[np.float64] = np.arange(
            deltaq_current, deltaq_current * (order_max + 1), deltaq_current
        )

        final_score: float = (np.mean(scores) - np.min(scores)) / np.std(scores)
        return rings, final_score

    def _opt_distance(
        self,
        radial_profile: np.ndarray[np.float64],
        distance_guess: float,
        wavelength: float,
        pixel_size: float,
    ) -> Tuple[np.ndarray[np.int64], np.ndarray[np.float64], float, float]:
        """Optimize the detector distance.

        Args:
            radial_profile (np.ndarray[np.float64]): 1-D array of peak intensities
                for an azimuthally integrated powder image.

            distance_guess (float): Starting guess for the detector distance.

            wavelength (float): X-ray wavelength in angstroms.

            pixel_size (float): Size of detector pixels in mm.

        Returns:
            peak_indices (np.ndarray[np.int64]): 1-D array of peak indices.

            selected_peaks (np.ndarray[np.float64]): Array of selected peaks.

            new_distance (float): New, optimized, detector distance.

            final_score (float): Final score calculated from residuals associated
                with each q-spacing.
        """
        peak_indices: np.ndarray[np.int64]
        peak_indices, _ = find_peaks(radial_profile, prominence=1, distance=10)
        theta: np.ndarray[np.float64] = np.arctan(
            np.arange(radial_profile.shape[0]) * pixel_size / distance_guess
        )
        q_profile: np.ndarray[np.float64] = 2.0 * np.sin(theta / 2.0) / wavelength

        rings: np.ndarray[np.float64]
        final_score: float
        rings, final_score = self._calc_and_score_ideal_rings(q_profile[peak_indices])
        peaks_predicted: np.ndarray[np.float64] = (
            2 * distance_guess * np.arcsin(rings * wavelength / 2.0) / pixel_size
        )

        # FOR BEHENATE ONLY!!! Need other q0s
        q0: float = 0.1076
        new_distance: float = (
            peaks_predicted[0]
            * pixel_size
            / np.tan(2.0 * np.arcsin(wavelength * (q0 / (2 * np.pi)) / 2.0))
        )
        return peak_indices, radial_profile[peak_indices], new_distance, final_score

    def _generate_concentric_sample_pts(
        self,
        peak_radii: np.ndarray[np.int64],
        center: Tuple[float, float],
        num_pts: int = 200,
    ) -> np.ndarray[np.float64]:
        # X,Y labelling seems backwards
        cx: float = center[0]
        cy: float = center[1]
        # Reshape linear radii (peak indices) for broadcasting
        radii: np.ndarray[np.int64] = np.array([peak_radii]).reshape(-1, 1)
        theta: np.ndarray[np.float64] = np.linspace(0.0, 2 * np.pi, 200)

        coords_x: np.ndarray[np.float64] = radii * np.cos(theta) + cx
        coords_y: np.ndarray[np.float64] = radii * np.sin(theta) + cy

        # Reshape for optimization routines
        coords: np.ndarray[np.float64] = np.zeros((2, num_pts * len(peak_radii)))
        coords[1] = coords_x.reshape(-1)
        coords[0] = coords_y.reshape(-1)

        return coords

    def _opt_center(
        self,
        powder: np.ndarray[np.float64],
        indices: np.ndarray[np.int64],
        center_guess: Tuple[float, float],
    ) -> lmfit.minimizer.MinimizerResult:
        """Optimize the beam center position on the detector.

        Args:
            powder (np.ndarray[np.float64]): 2-D assembled powder image.

            indices (np.ndarray[np.float64]): Indices of peaks in a radial profile
                of the azimuthally integrated powder image.

            center_guess (Tuple[float, float]): Starting guess for the beam center
                in pixels.

        Returns:
            res (lmfit.minimizer.MinimizerResult): Result from the minimization.
                res.params["cx"] and res.params["cy"] contain the new beam center.
        """
        # Perform fitting
        params: lmfit.Parameters = lmfit.Parameters()
        params.add("cx", value=center_guess[0])
        params.add("cy", value=center_guess[1])
        for i in range(len(indices)):
            params.add(f"r{i:d}", value=indices[i])
        res: lmfit.minimizer.MinimizerResult = lmfit.minimize(
            geometry_optimize_residual,
            params,
            method="leastsq",
            nan_policy="omit",
            args=(powder,),
        )
        try:
            lmfit.report_fit(res)
        except TypeError as err:
            # This shouldn't happen but don't fail if it does
            logger.error(f"Unable to report fit! {err}")
        return res

    def _opt_geom(
        self,
        powder: np.ndarray[np.float64],
        mask: np.ndarray[np.uint64],
        params_guess: Tuple[int, Tuple[float, float], float],
        n_iterations: int,
        wavelength: float,
        pixel_size: float,
    ) -> Tuple[List[float], List[float], List[Tuple[float, float]]]:
        """Optimize the detector distance and beam center.

        Args:
            powder (np.ndarray[np.float64]): 2-D assembled powder image.

            mask (np.ndarray[np.float64]): Corresponding binary mask for the
                powder image.

            params_guess (Tuple[int, Tuple[float, float], float]): Initial guesses.
                In format: (n_peaks, (center_x, center_y), distance).

            n_iterations (int): Number of iterations to perform.

            wavelength (float): X-ray wavelength in angstroms.

            pixel_size (float): Size of detector pixels in mm.

        Returns:
            peak_indices (np.ndarray[np.int64]): 1-D array of peak indices.

            selected_peaks (np.ndarray[np.float64]): Array of selected peaks.

            final_scores (float): Final scores calculated from residuals associated
                with each q-spacing for each iteration of optimization.

            calc_distances (List[float]): Optimized distances associated with each
                score.

            calc_centers (List[Tuple[float, float]]): Optimized centers associated
                with each score.
        """
        n_peaks: int = self._task_parameters.n_peaks
        radial_profile: np.ndarray[np.float64] = self._radial_profile(
            powder, mask, params_guess[1]
        )
        indices: np.ndarray[np.int64]
        peaks: np.ndarray[np.float64]
        distance: float
        final_score: float
        indices, peaks, distance, final_score = self._opt_distance(
            radial_profile, params_guess[2], wavelength, pixel_size
        )

        final_scores: List[float] = [final_score]
        calc_distances: List[float] = [distance]
        calc_centers: List[Tuple[float, float]] = [params_guess[1]]

        center_guess: Tuple[float, float] = params_guess[1]
        for iter in range(n_iterations):
            # Select the highest intensity peaks from the first 8 in q
            selected_indices: np.ndarray[np.int64] = indices[
                np.argsort(peaks[:8])[::-1][:n_peaks]
            ]
            res: lmfit.minimizer.MinimizerResult = self._opt_center(
                powder, selected_indices, center_guess
            )
            center_guess = (res.params["cx"].value, res.params["cy"].value)
            logger.info(f"New center is: ({center_guess[0]}, {center_guess[1]})")
            radial_profile = self._radial_profile(powder, mask, center_guess)
            indices, peaks, distance, final_score = self._opt_distance(
                radial_profile, distance, wavelength, pixel_size
            )
            final_scores.append(final_score)
            calc_distances.append(distance)
            calc_centers.append(center_guess)
        return final_scores, calc_distances, calc_centers

    def _run(self) -> None:
        """Perform geometry optimization of detector distance and beam center.

        Requires a powder image from data acquired of Ag Behenate.
        """
        exp: str = self._task_parameters.lute_config.experiment
        run: Union[int, str] = self._task_parameters.lute_config.run
        ds: psana.DataSource = psana.DataSource(f"exp={exp}:run={run}")
        det: psana.Detector = psana.Detector(self._task_parameters.detname, ds.env())
        pixel_size_mm: float
        wavelength_angs: float
        pixel_size_mm, wavelength_angs = self._get_pixel_size_and_wavelength(ds, det)

        powder: np.ndarray[np.float64] = self._extract_powder(
            self._task_parameters.powder
        )
        # FIX THIS
        mask: Optional[np.ndarray[np.float64]] = None
        powder[powder > self._task_parameters.threshold] = 0

        starting_centers: List[Tuple[float, float]] = self._center_guesses(powder)
        starting_distance: float = self._estimate_distance()
        starting_scan_params: List[Tuple[int, Tuple[float, float], float]] = list(
            itertools.product(
                (self._task_parameters.n_peaks,),
                starting_centers,
                (starting_distance,),
            )
        )
        if self._mpi_rank == 0:
            quotient: int
            remainder: int
            quotient, remainder = divmod(len(starting_scan_params), self._mpi_size)
            parameters_per_rank = np.array(
                [
                    quotient + 1 if rank < remainder else quotient
                    for rank in range(self._mpi_size)
                ]
            )
            start_indices_per_rank: np.ndarray[np.int64] = np.zeros(
                self._mpi_size, dtype=np.int64
            )
            start_indices_per_rank[1:] = np.cumsum(parameters_per_rank[:-1])
        else:
            parameters_per_rank = np.zeros(self._mpi_size, dtype=np.int64)
            start_indices_per_rank = np.zeros(self._mpi_size, dtype=np.int64)

        self._mpi_comm.Bcast(parameters_per_rank, root=0)
        self._mpi_comm.Bcast(start_indices_per_rank, root=0)

        # Basic RANK-LOCAL variables - Each rank uses a subset of params
        ################################################################
        n_params: int = parameters_per_rank[self._mpi_rank]
        start_idx: int = start_indices_per_rank[self._mpi_rank]
        stop_idx: int = start_idx + n_params

        final_scores_by_rank: List[float] = []
        final_distances_by_rank: List[float] = []
        final_centers_by_rank: List[Tuple[float, float]] = []
        for params in starting_scan_params[start_idx:stop_idx]:
            scores: List[float]
            distances: List[float]
            centers: List[Tuple[float, float]]
            scores, distances, centers = self._opt_geom(
                powder,
                mask,
                params,
                self._task_parameters.n_iterations,
                wavelength_angs,
                pixel_size_mm,
            )
            final_scores_by_rank.extend(scores)
            final_distances_by_rank.extend(distances)
            final_centers_by_rank.extend(centers)

        self._mpi_comm.Barrier()

        # Gather all results
        final_scores: Union[List[float], List[List[float]]] = self._mpi_comm.gather(
            final_scores_by_rank, root=0
        )
        final_distances: Union[List[float], List[List[float]]] = self._mpi_comm.gather(
            final_distances_by_rank, root=0
        )
        final_centers: Union[
            List[Tuple[float, float], List[List[Tuple[float, float]]]]
        ] = self._mpi_comm.gather(final_centers_by_rank, root=0)

        # Flatten nested lists
        def flatten(nested_lists: List[List[Any]]) -> List[Any]:
            flattened: List[Any] = []
            for item in nested_lists:
                flattened.extend(item)
            return flattened

        if self._mpi_rank == 0:
            final_scores = flatten(final_scores)
            final_distances = flatten(final_distances)
            final_centers = flatten(final_centers)
            best_idx: int = np.argmax(final_scores)
            best_distance: float = final_distances[best_idx]
            best_center: Tuple[float, float] = final_centers[best_idx]
            # Calculate resolution at edge
            theta: np.ndarray[np.float64] = np.arctan(
                np.array((powder.shape[0] / 2)) * pixel_size_mm / best_distance
            )
            q: np.ndarray[np.float64] = 2.0 * np.sin(theta / 2.0) / wavelength_angs
            edge_resolution: float = 1.0 / q

            self._result.summary = {
                "Detector distance (mm)": best_distance,
                "Detector center (pixels)": best_center,
                "Detector edge resolution (A)": edge_resolution,
            }

    def _post_run(self) -> None:
        super()._post_run()
        self._result.task_status = TaskStatus.COMPLETED
