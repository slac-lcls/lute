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
    Psana2ToPyFAI,
    PyFAIToPsana2,
    PyFAIToCrystFEL,
)
from LCLSGeom.geometry import get_beam_center  # type: ignore

from psana import DataSource  # type: ignore
from psana.pscalib.calib.MDB_CLI import *  # gu, mu, etc
import psana.pscalib.calib.MDBUtils as mu
import psana.pscalib.calib.MDBWebUtils as wu
import psana.detector.UtilsCalib as uc

cc = wu.cc

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
        exp = self._task_parameters.lute_config.experiment
        run = int(self._task_parameters.lute_config.run)
        det_type = self._task_parameters.det_type
        if det_type.lower() == "jungfrau16m":
            detname = "jungfrau"
        psana_to_pyfai = PsanaToPyFAI(
            exp=exp,
            run_num=run,
            detname=detname,
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
                        if self._task_parameters.det_type == "jungfrau16M":
                            powder = h5[f"Sums/jungfrau_calib_max"][()]
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

    def _update_database(self, dbsuffix: Optional[str] = None):
        """
        Update the geometry database with the new geometry parameters.
        """
        assert isinstance(self._task_parameters, OptimizePyFAIGeometryParameters)
        exp = self._task_parameters.lute_config.experiment
        run_num = int(self._task_parameters.lute_config.run)
        det_type = self._task_parameters.det_type
        if det_type.lower() == "jungfrau16m":
            detname = "jungfrau"

        fname = self._task_parameters.out_file
        ctype = "geometry"
        dtype = "str"
        ds = DataSource(exp=exp, run=run_num)
        runs = next(ds.runs())
        det = runs.Detector(detname)
        longname: str = det.raw._uniqueid
        shortname: str = uc.detector_name_short(longname)
        det_type: str = det._dettype

        data = mu.data_from_file(fname, ctype, dtype, verb="DEBUG")

        # Will just setup all keyword arguments
        run_orig: int = (
            run_num  # This will be the run used for processing I think (e.g. the AgBh run you processed)
        )
        run_beg: int = 0  # Don't actually know how this differs from "run" below
        run_end: str = "end"
        run: int = (
            run_beg  # This is the run for validity checking I think, can change validity ranges
        )

        kwa = {
            "iofname": fname,
            "experiment": exp,
            "ctype": ctype,
            "dtype": dtype,
            "detector": shortname,
            "shortname": shortname,
            "detname": detname,
            "longname": longname,
            # Need to check what all these are
            # "time_sec": "...",
            # "time_stamp": "...",
            # "tsshort": "...",
            # "tstamp_orig": "...",
            "run": run,
            "run_beg": run_beg,
            "run_end": run_end,
            "run_orig": run_orig,
            # Not sure if this is should be provided - probably leave these 3 off for now.
            # "version": "...",
            # "comment": "...",
            # "extpars": {"content": "extended parameters dict->json->str",},
            "dettype": det_type,
            "dbsuffix": dbsuffix,  # Exclude if not needed
        }

        _ = wu.deploy_constants(
            data,
            exp,
            longname,
            url=cc.URL_KRB,
            krbheaders=cc.KRBHEADERS,
            **kwa,
        )

    def _update_geometry(self, dbsuffix: Optional[str] = None):
        """
        Update the geometry by writing a new poni, geom and data files as well as updating the geometry database.

        Parameters
        ----------
        optimizer : BayesGeomOpt
            Optimizer object
        dbsuffix : Optional[str]
            Suffix to append to the database file name, if any.
        """
        assert isinstance(self._task_parameters, OptimizePyFAIGeometryParameters)
        exp = self._task_parameters.lute_config.experiment
        run_num = int(self._task_parameters.lute_config.run)
        det_type = self._task_parameters.det_type
        if det_type == "jungfrau16M":
            detname = "jungfrau"
        path = os.path.dirname(self._task_parameters.out_file)
        poni_file = os.path.join(
            path, f"r{self._task_parameters.lute_config.run:0>4}.poni"
        )

        PyFAIToPsana(
            in_file=poni_file,
            exp=exp,
            run_num=run_num,
            detname=detname,
            out_file=self._task_parameters.out_file,
            dbsuffix=dbsuffix,
        )
        PyFAIToCrystFEL(
            in_file=poni_file,
            exp=exp,
            run_num=run_num,
            detname=detname,
            out_file=self._task_parameters.out_file,
            dbsuffix=dbsuffix,
        )

        if dbsuffix is None:
            dbsuffix = "testgeom"

        print(f"BEFORE UPDATE DATABASE", flush=True)
        self._update_database(dbsuffix=dbsuffix)
        print(f"AFTER UPDATE DATABASE", flush=True)

        converter = PsanaToPyFAI(
            exp=exp,
            run=run_num,
            detname=detname,
            dbsuffix=dbsuffix,
        )
        detector = converter.detector
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
        print(f"BEFORE OPTIMIZATION: Process {optimizer.rank}", flush=True)
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
        print(f"AFTER OPTIMIZATION: Process {optimizer.rank}", flush=True)
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
            plot = f"{fig_folder}/bayFAI_{optimizer.exp}_r{optimizer.run:0>4}.png"
            print(f"BEFORE UPDATE GEOMETRY: Process {optimizer.rank}", flush=True)
            calib_detector = self._update_geometry()
            fig, low_q, low_res, high_q, high_res, border_q, border_res = (
                optimizer.visualize_results(
                    powder=optimizer.powder,
                    bo_history=optimizer.bo_history,
                    detector=calib_detector,
                    distance=distance,
                    plot=plot,
                )
            )

            plots = pn.Tabs(fig)
            self._result.summary = []
            self._result.summary.append(
                {
                    "Detector distance (m)": f"{distance:.6f}",
                    "Detector center (pix)": (
                        f"{cx/calib_detector.pixel_size:.3f}",
                        f"{cy/calib_detector.pixel_size:.3f}",
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
