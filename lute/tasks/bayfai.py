"""
Classes for geometry optimization tasks.

Classes:
    OptimizePyFAIGeom: optimize detector geometry using PyFAI coupled with Bayesian Optimization

"""

__all__ = ["OptimizePyFAIGeometry"]
__author__ = "Louis Conreux"

from lute.io.models.bayfai import OptimizePyFAIGeometryParameters
from lute.tasks._bayfai import BayesGeomOpt
from lute.tasks.task import Task
from lute.tasks.dataclasses import TaskStatus, ElogSummaryPlots
from lute.execution.logging import get_logger

import psana  # type: ignore
import os
import logging
from typing import Optional, Tuple

import h5py  # type: ignore
import panel as pn  # type: ignore
import numpy as np
import numpy.typing as npt
import time  # type: ignore
from scipy.ndimage import gaussian_filter, convolve, gaussian_laplace  # type: ignore

from LCLSGeom.converter import (  # type: ignore
    PsanaToPyFAI,
    PyFAIToPsana,
    PsanaToCrystFEL,
)
from LCLSGeom.calib import fetch_template  # type: ignore
from LCLSGeom.geometry import get_beam_center  # type: ignore

logger: logging.Logger = get_logger(__name__)


class OptimizePyFAIGeometry(Task):
    """Optimize detector geometry using PyFAI coupled with Bayesian Optimization."""

    def __init__(
        self, *, params: OptimizePyFAIGeometryParameters, use_mpi: bool = True
    ) -> None:
        super().__init__(params=params, use_mpi=use_mpi)

    def _build_pyFAI_detector(self):
        """
        Fetch the geometry data and build a pyFAI detector object.
        """
        assert isinstance(self._task_parameters, OptimizePyFAIGeometryParameters)
        in_file = self._task_parameters.in_file
        det_type = self._task_parameters.det_type
        ds_args = f"exp={self._task_parameters.lute_config.experiment}:run={self._task_parameters.lute_config.run}:idx"
        self.ds = psana.DataSource(ds_args)
        self.det = psana.Detector(det_type, self.ds.env())
        self.shape = self.det.shape()
        if det_type.lower() == "rayonix":
            env = self.ds.env()
            cfg = env.configStore()
            pixel_size_um = cfg.get(psana.Rayonix.ConfigV2).pixelWidth()
            self.pixel_size = pixel_size_um * 1e-6
            if in_file == "":
                logger.info(
                    f"No geometry file found for exp {self._task_parameters.lute_config.experiment}",
                )
                logger.info(
                    f"Fetching default geometry for {det_type} detector with pixel size {pixel_size_um} µm and shape {self.shape}",
                )
                src = str(self.det.name)
                in_file = fetch_template(
                    self._task_parameters.lute_config.experiment,
                    det_type,
                    src,
                    pixel_size_um,
                    self.shape,
                )
                time.sleep(5)
                self._task_parameters.in_file = in_file
        else:
            self.pixel_size = self.det.pixel_size(self.ds.env()) * 1e-6
        psana_to_pyfai = PsanaToPyFAI(
            in_file=in_file,
            det_type=det_type,
            pixel_size=self.pixel_size,
            shape=self.shape,
        )
        detector = psana_to_pyfai.detector
        return detector

    def _check_path_and_type(self, string: str) -> Tuple[bool, Optional[str]]:
        """
        Check if a string is a valid path and determine the filetype.

        Parameters
        ----------
        string : str
            String that may be a file path.

        Returns
        -------
        is_valid_path : bool
            If it is a valid path.

        powder_type : Optional[str]
            If is_valid_path, the file type.
        """
        is_valid_path: bool = False
        powder_type: Optional[str] = None
        if os.path.exists(string):
            is_valid_path = True
        else:
            return is_valid_path, powder_type
        try:
            with h5py.File(string):
                powder_type = "smd"
                is_valid_path = True

            return is_valid_path, powder_type
        except Exception:
            ...

        try:
            np.load(string)
            powder_type = "numpy"
            is_valid_path = True
            return is_valid_path, powder_type
        except ValueError:
            ...

        return is_valid_path, powder_type

    def _extract_powder(
        self, powder_path: str, shape: Tuple
    ) -> Optional[npt.NDArray[np.float64]]:
        """
        Extract a powder image from either a smalldata file or numpy array.

        Parameters
        ----------
        powder_path : str
            Path to the object containing the powder image.
        shape : Tuple
            Stacked shape of the detector. Powder image has to be reshaped to match detector shape.

        Returns
        -------
        powder : Optional[npt.NDArray[np.float64]]
            The extracted powder image.
            Returns None if no powder could be extracted and no specific error was encountered.
        """
        powder: Optional[npt.NDArray[np.float64]] = None
        assert isinstance(self._task_parameters, OptimizePyFAIGeometryParameters)
        if isinstance(powder_path, str):
            is_valid: bool
            dtype: Optional[str]
            is_valid, dtype = self._check_path_and_type(powder_path)
            if is_valid and dtype == "numpy":
                powder = np.load(powder_path)
                if powder is not None and powder.shape != shape:
                    powder = np.reshape(powder, shape)
            elif is_valid and dtype == "smd":
                h5: h5py.File
                with h5py.File(powder_path) as h5:
                    try:
                        if self._task_parameters.det_type == "Rayonix":
                            powder = h5[
                                f"Sums/{self._task_parameters.det_type}_calib_skipFirst_max"
                            ][()]
                        else:
                            powder = h5[
                                f"Sums/{self._task_parameters.det_type}_calib_max"
                            ][()]
                    except KeyError:
                        logger.warning(
                            'No "Max" powder found in SmallData. Using "Sum" powder.'
                        )
                        powder = h5[f"Sums/{self._task_parameters.det_type}_calib"][()]
                    if powder is not None and powder.shape != shape:
                        powder = np.reshape(powder, shape)
        return powder

    def _preprocess_powder(
        self,
        powder: Optional[npt.NDArray[np.float64]],
        preprocess: Optional[str] = "Diagonal",
    ) -> Optional[npt.NDArray[np.float64]]:
        """
        Preprocess extracted powder for enhancing optimization

        Parameters
        ----------
        powder : npt.NDArray[np.float64]
            Powder image to use for calibration
        preprocess : str
            Type of preprocessing technique
                Available preprocessing:
                "Finite": Gradient Computation using Finite Differences
                "Central": Gradient Computation using Central Differences
                "Laplacian": Gradient Computation using Laplacian of Gaussian
                "Sobel": Gradient Computation using Sobel filter
        """
        assert isinstance(self._task_parameters, OptimizePyFAIGeometryParameters)
        if powder is not None:
            powder[powder < 0] = 0
            self.raw_powder = powder
            if preprocess is None:
                return powder
            elif preprocess == "Finite":
                sigma = 1
                calib = gaussian_filter(powder, sigma=sigma)
                gradx_calib = np.zeros_like(powder)
                grady_calib = np.zeros_like(powder)
                gradx_calib[:-1, :] = calib[1:, :] - calib[:-1, :]
                grady_calib[:, :-1] = calib[:, 1:] - calib[:, :-1]
                powder = np.sqrt(gradx_calib**2 + grady_calib**2)
            elif preprocess == "Central":
                sigma = 1
                calib = gaussian_filter(powder, sigma=sigma)
                gradx_calib = np.zeros_like(powder)
                grady_calib = np.zeros_like(powder)
                gradx_calib[1:-1, :] = (calib[2:, :] - calib[:-2, :]) / 2
                grady_calib[:, 1:-1] = (calib[:, 2:] - calib[:, :-2]) / 2
                powder = np.sqrt(gradx_calib**2 + grady_calib**2)
            elif preprocess == "Diagonal":
                sigma = 1
                calib = gaussian_filter(powder, sigma=sigma)
                gradx_calib = np.zeros_like(powder)
                grady_calib = np.zeros_like(powder)
                gradx_calib[:-1, :-1] = (
                    calib[1:, :-1] - calib[:-1, :-1] + calib[1:, 1:] - calib[:-1, 1:]
                ) / 2
                grady_calib[:-1, :-1] = (
                    calib[:-1, 1:] - calib[:-1, :-1] + calib[1:, 1:] - calib[1:, :-1]
                ) / 2
                powder = np.sqrt(gradx_calib**2 + grady_calib**2)
            elif preprocess == "Sobel":
                sigma = 1
                calib = gaussian_filter(powder, sigma=sigma)
                sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
                sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
                gradx_calib = convolve(calib, sobel_x, mode="reflect")
                grady_calib = convolve(calib, sobel_y, mode="reflect")
                powder = np.sqrt(gradx_calib**2 + grady_calib**2)
            elif preprocess == "Laplacian":
                sigma = 1
                powder = gaussian_laplace(powder, sigma=sigma)
            else:
                logger.warning(f"Preprocessing technique {preprocess} not recognized.")
                logger.warning("Using raw powder instead.")
        return powder

    def _update_geometry(self, optimizer):
        """
        Update the geometry and write a new .poni, .geom and .data file

        Parameters
        ----------
        optimizer : BayesGeomOpt
            Optimizer object
        """
        assert isinstance(self._task_parameters, OptimizePyFAIGeometryParameters)
        path = os.path.dirname(self._task_parameters.out_file)
        poni_file = os.path.join(
            path, f"r{self._task_parameters.lute_config.run:0>4}.poni"
        )
        optimizer.gr.save(poni_file)
        PyFAIToPsana(
            in_file=poni_file,
            psana_file=self._task_parameters.in_file,
            out_file=self._task_parameters.out_file,
        )
        geom_file = os.path.join(
            path, f"r{self._task_parameters.lute_config.run:0>4}.geom"
        )
        PsanaToCrystFEL(
            in_file=self._task_parameters.out_file,
            out_file=geom_file,
        )
        psana_to_pyfai = PsanaToPyFAI(
            in_file=self._task_parameters.out_file,
            det_type=self._task_parameters.det_type,
            pixel_size=self.pixel_size,
            shape=self.shape,
        )
        detector = psana_to_pyfai.detector
        return detector

    def _run(self) -> None:
        start_time = time.time()
        assert isinstance(self._task_parameters, OptimizePyFAIGeometryParameters)
        detector = self._build_pyFAI_detector()
        powder = self._extract_powder(self._task_parameters.powder, detector.shape)
        powder = self._preprocess_powder(powder, self._task_parameters.preprocess)
        if powder is None:
            raise RuntimeError("Unable to extract powder. Cannot continue.")
        optimizer = BayesGeomOpt(
            exp=self._task_parameters.lute_config.experiment,
            run=self._task_parameters.lute_config.run,
            det_type=self._task_parameters.det_type,
            detector=detector,
            calibrant=self._task_parameters.calibrant,
        )
        optimizer.bayes_opt_geom(
            powder=powder,
            bounds=self._task_parameters.bo_params.bounds,
            res=self._task_parameters.bo_params.res,
            max_rings=self._task_parameters.bo_params.max_rings,
            n_samples=self._task_parameters.bo_params.n_samples,
            n_iterations=self._task_parameters.bo_params.n_iterations,
            kernel=self._task_parameters.bo_params.kernel,
            af=self._task_parameters.bo_params.af,
            hyperparam=self._task_parameters.bo_params.hyperparams,
            prior=self._task_parameters.bo_params.prior,
            seed=self._task_parameters.bo_params.seed,
        )
        if optimizer.rank == 0:
            logger.info("Optimization complete")
            logger.info(f"Elapsed time: {time.time() - start_time:.2f} s")
            distance, cx, cy = get_beam_center(optimizer.params)
            logger.info(f"Detector Distance to Sample: {distance:.6f}")
            logger.info(f"Beam center: ({cx:.6f}, {cy:.6f})")
            logger.info(
                f"Rotations: \u03b8x = ({optimizer.params[3]:.2e}, \u03b8y = {optimizer.params[4]:.2e}, \u03b8z = {optimizer.params[5]:.2e})"
            )
            logger.info(f"Final Residual: {optimizer.residual:.2e}")
            fig_folder = os.path.join(
                self._task_parameters.lute_config.work_dir, "figs"
            )
            os.makedirs(fig_folder, exist_ok=True)
            plot = f"{fig_folder}/bayFAI_diagnostics_{optimizer.exp}_r{optimizer.run:0>4}.png"
            calib_detector = self._update_geometry(optimizer)
            powder_plot, low_q, low_res, high_q, high_res, border_q, border_res = (
                optimizer.create_interactive_powder(
                    powder=optimizer.powder,
                    detector=calib_detector,
                    distance=distance,
                )
            )
            diagnostics_plot = optimizer.create_diagnostics_panel(
                powder=optimizer.powder,
                bo_history=optimizer.bo_history,
                detector=calib_detector,
                distance=distance,
                low_resolution=low_res,
                high_resolution=high_res,
                border_resolution=border_res,
                plot=plot,
            )
            pn.extension("matplotlib", "bokeh")
            plots = pn.Row(
                pn.pane.Matplotlib(diagnostics_plot, sizing_mode="fixed"),
                powder_plot,
            )
            content = pn.Column(
                pn.pane.Markdown(
                    "### Detector Geometry Optimization Summary",
                    styles={
                        "font-size": "2em",
                        "font-weight": "bold",
                        "text-align": "center",
                        "margin": "0 auto",
                        "display": "block",
                    },
                ),
                plots,
                sizing_mode="stretch_width",
            )
            content.save(
                f"{fig_folder}/bayFAI_summary_{optimizer.exp}_r{optimizer.run:0>4}.html",
                embed=True,
            )
            plots = pn.Tabs(content)
            self._result.summary = []
            self._result.summary.append(
                {
                    "Detector distance (m)": f"{distance:.6f}",
                    "Detector center (pix)": (
                        f"{cx/self.pixel_size:.3f}",
                        f"{cy/self.pixel_size:.3f}",
                    ),
                    "Low q": f"{low_q:.3f} \u00c5-1 | {low_res:.3f} \u00c5",
                    "High q": f"{border_q:.3f} \u00c5-1 | {border_res:.3f} \u00c5 (detector edge)",
                    "Highest q": f"{high_q:.3f} \u00c5-1 | {high_res:.3f} \u00c5 (detector corner)",
                }
            )
            logger.info(f">>> Low q : {low_q:.3f} \u00c5-1 | {low_res:.3f} \u00c5")
            logger.info(
                f">>> High q : {border_q:.3f} \u00c5-1 | {border_res:.3f} \u00c5 (detector edge)"
            )
            logger.info(
                f">>> Highest q : {high_q:.3f} \u00c5-1 | {high_res:.3f} \u00c5 (detector corner)"
            )
            self._result.summary.append(
                ElogSummaryPlots(
                    f"Geometry_Fit/r{self._task_parameters.lute_config.run:0>4}", plots
                )
            )
            self._result.task_status = TaskStatus.COMPLETED
