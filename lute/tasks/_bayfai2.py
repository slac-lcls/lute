"""
Classes for geometry optimization tasks.

Classes:
    BayFAIOpt: optimize detector geometry using PyFAI coupled with Bayesian Optimization.

Functions:
    - build_detector: build a PyFAI detector from a .data file.
    - generate_powder: generate a powder diffraction image from a .data file.
    - min_intensity: find the minimum intensity in a powder diffraction image.
    - define_calibrant: define the calibrant for a powder diffraction image.
    - update_geometry: update the detector geometry based on the optimization results.
"""

__all__ = [
    "BayFAIOpt",
    "build_detector",
    "generate_powder",
    "min_intensity",
    "define_calibrant",
    "update_geometry",
]
__author__ = "Louis Conreux"

from lute.execution.logging import get_logger

import os
from psana import DataSource  # type: ignore
from psana.pscalib.calib.MDB_CLI import *  # type: ignore
import psana.pscalib.calib.MDBUtils as mu  # type: ignore
import psana.pscalib.calib.MDBWebUtils as wu  # type: ignore
import psana.detector.UtilsCalib as uc  # type: ignore
import logging
import numpy as np
import numpy.typing as npt
from typing import Optional, Union, Any
import matplotlib.pyplot as plt  # type: ignore
import matplotlib.patches as patches  # type: ignore
from matplotlib import lines  # type: ignore
from bokeh.plotting import figure  # type: ignore
from bokeh.models import ColorBar, LinearColorMapper, HoverTool, ColumnDataSource  # type: ignore
from bokeh.palettes import Viridis256  # type: ignore
from bokeh.models.annotations import Label  # type: ignore
import h5py  # type: ignore
from scipy.ndimage import gaussian_filter  # type: ignore
import pyFAI  # type: ignore
from pyFAI.geometry import Geometry  # type: ignore
from pyFAI.goniometer import SingleGeometry  # type: ignore
from pyFAI.geometryRefinement import GeometryRefinement  # type: ignore
from pyFAI.calibrant import CALIBRANT_FACTORY  # type: ignore
from pyFAI.units import RADIAL_UNITS  # type: ignore
from sklearn.gaussian_process import GaussianProcessRegressor  # type: ignore
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel  # type: ignore
from sklearn.utils._testing import ignore_warnings  # type: ignore
from sklearn.exceptions import ConvergenceWarning  # type: ignore
from mpi4py import MPI

from LCLSGeom.psana2.converter import PsanaToPyFAI, PsanaToCrystFEL, PyFAIToPsana  # type: ignore

pyFAI.use_opencl = False

cc = wu.cc

logger: logging.Logger = get_logger(__name__)


def extract_powder(powder_path: str, detname: str) -> npt.NDArray[np.float64]:
    """
    Extract a powder image from smalldata analysis.

    Parameters
    ----------
    powder_path : str
        Path to the h5 file containing the powder data.

    Returns
    -------
    powder : npt.NDArray[np.float64]
        The extracted powder image.
    """
    with h5py.File(powder_path) as h5:
        try:
            powder = h5[f"Sums/{detname}_calib_max"][()]
        except KeyError:
            logger.warning(
                f"Cannot find {detname} Max powder in {powder_path}, defaulting to {detname} Sum instead."
            )
            powder = h5[f"Sums/{detname}_calib"][()]
    return powder


def preprocess_powder(
    powder: npt.NDArray[np.float64], smooth: bool = False
) -> npt.NDArray[np.float64]:
    """
    Preprocess extracted powder for enhancing optimization

    Parameters
    ----------
    powder : npt.NDArray[np.float64]
        Powder image to use for calibration
    smooth : bool, optional
        If True, apply smoothing to the powder image.
    """
    powder[powder < 0] = 0
    if smooth:
        for p in range(powder.shape[0]):
            calib = gaussian_filter(powder[p], sigma=1)
            gradx = np.zeros_like(calib)
            grady = np.zeros_like(calib)
            gradx[:-1, :-1] = (
                calib[1:, :-1] - calib[:-1, :-1] + calib[1:, 1:] - calib[:-1, 1:]
            ) / 2
            grady[:-1, :-1] = (
                calib[:-1, 1:] - calib[:-1, :-1] + calib[1:, 1:] - calib[1:, :-1]
            ) / 2
            powder[p] = np.sqrt(gradx**2 + grady**2)
    return powder


def generate_powder(
    powder_path: str, detname: str, smooth: bool = False
) -> npt.NDArray[np.float64]:
    """
    Generate a preprocessed powder image from smalldata reduction.

    Parameters
    ----------
    powder_path : str
        Path to the h5 file containing the powder data.
    detname : str
        Name of the detector
    smooth : bool, optional
        If True, apply smoothing to the powder image.
    """
    powder = extract_powder(powder_path, detname)
    powder = preprocess_powder(powder, smooth)
    return powder


def min_intensity(powder: npt.NDArray[np.float64]) -> float:
    """
    Define minimal intensity for identifying Bragg peaks.

    The minimal intensity is chosen so that the Signal to Noise Ratio (SNR) is maximized
    Signal is defined as the standard deviation of the pixels above the threshold
    Noise is defined as the standard deviation of the pixels below the threshold

    Parameters
    ----------
    powder : np.ndarray
        Powder image
    """
    mean = np.mean(powder)
    threshold = mean + 5 * np.std(powder)
    nice_pix = powder < threshold
    SNRs = []
    Imins = np.arange(95, 100, 0.25)
    for Imin in Imins:
        threshold = np.percentile(powder[nice_pix], Imin)
        signal_pixels = powder[nice_pix][powder[nice_pix] > threshold]
        signal = np.std(signal_pixels)
        noise_pixels = powder[nice_pix][powder[nice_pix] <= threshold]
        noise = np.std(noise_pixels)
        SNRs.append(signal / noise)
    q = Imins[np.argmax(SNRs)]
    Imin = np.percentile(powder[nice_pix], q)
    return Imin


def build_LCLS2_detector(exp: str, run: int, detname: str) -> pyFAI.detectors.Detector:
    """
    Read the metrology data and build a pyFAI detector object.

    Parameters
    ----------
    in_file : str
        Path to the input file
    shape : tuple
        Shape of the detector (n_modules, fs_dim, ss_dim)

    Returns
    -------
    pyFAI.Detector
        Configured pyFAI detector object
    """
    psana_to_pyfai = PsanaToPyFAI(
        exp=exp,
        run_num=run,
        detname=detname,
    )
    detector = psana_to_pyfai.detector
    return detector


def update_geometry(optimizer: Any, out_file: str):
    """
    Update the geometry and write a new .poni, .geom and .data file

    Parameters
    ----------
    optimizer : BayFAIOpt
        Optimizer object
    out_file : str
        Path to the output file
    """
    path = os.path.dirname(out_file)
    poni_file = os.path.join(path, f"r{optimizer.run:0>4}.poni")
    optimizer.gr.save(poni_file)
    PyFAIToPsana(
        in_file=poni_file,
        detector=optimizer.detector,
        out_file=out_file,
    )
    geom_file = os.path.join(path, f"r{optimizer.run:0>4}.geom")
    PsanaToCrystFEL(
        in_file=out_file,
        out_file=geom_file,
    )
    psana_to_pyfai = PsanaToPyFAI(
        in_file=out_file,
        shape=optimizer.detector.raw_shape,
    )
    detector = psana_to_pyfai.detector
    return detector


def define_calibrant(
    calibrant_name: str, exp: str, run: Union[str, int]
) -> pyFAI.calibrant.Calibrant:
    """
    Define calibrant for optimization with appropriate wavelength

    Parameters
    ----------
    calibrant : str
        Name of the calibrant
    exp : str
        Name of the experiment
    run : int
        Run number
    """
    ds = DataSource(exp=exp, run=run)
    runs = next(ds.runs())
    evt = next(runs.events())
    calibrant = CALIBRANT_FACTORY(calibrant_name)
    try:
        det_photon_energy = runs.Detector("ebeamh")
        photon_energy = det_photon_energy.raw.ebeamPhotonEnergy(evt)
        wavelength = 1.23984197386209e-06 / photon_energy
    except Exception:
        det_wavelength = runs.Detector("SIOC:SYS0:ML00:AO192")
        wavelength = det_wavelength(evt) * 1e-9
        photon_energy = 1.23984197386209e-06 / wavelength
    calibrant.wavelength = wavelength
    return calibrant


def rotation_matrix(params: list) -> np.ndarray:
    """
    Compute and return the detector tilts as a single rotation matrix

    Parameters
    ----------
    params : list
        Detector parameters found by PyFAI calibration
    """
    cos_rot1 = np.cos(params[3])
    cos_rot2 = np.cos(params[4])
    cos_rot3 = np.cos(params[5])
    sin_rot1 = np.sin(params[3])
    sin_rot2 = np.sin(params[4])
    sin_rot3 = np.sin(params[5])
    # Rotation about vertical axis: Note this rotation is left-handed
    rot1 = np.array(
        [[1.0, 0.0, 0.0], [0.0, cos_rot1, sin_rot1], [0.0, -sin_rot1, cos_rot1]]
    )
    # Rotation about horizontal axis: Note this rotation is left-handed
    rot2 = np.array(
        [[cos_rot2, 0.0, -sin_rot2], [0.0, 1.0, 0.0], [sin_rot2, 0.0, cos_rot2]]
    )
    # Rotation about z-axis: Note this rotation is right-handed
    rot3 = np.array(
        [[cos_rot3, -sin_rot3, 0.0], [sin_rot3, cos_rot3, 0.0], [0.0, 0.0, 1.0]]
    )
    rotation_matrix = np.dot(np.dot(rot3, rot2), rot1)
    return rotation_matrix


def correct_geom(detector: pyFAI.detectors.Detector, params: Optional[list] = None):
    """
    Correct the geometry given a set of geometry parameters.

    Parameters
    ----------
    detector : pyFAI.detectors.Detector
        PyFAI detector object containing pixel coordinates.
    params : list, optional
        6 Geometry parameters: distance, x-shift, y-shift, Rx, Ry, Rz
    """
    x, y, z = detector.calc_cartesian_positions()
    if params is not None:
        dist = params[0]
        poni1 = params[1]
        poni2 = params[2]
        x = (x - (detector.pixel_size / 2) - poni1).ravel()
        y = (y - (detector.pixel_size / 2) - poni2).ravel()
        if z is None:
            z = np.zeros_like(x) + dist
        else:
            z = (z + dist).ravel()
        coord_det = np.vstack((x, y, z))
        x, y, z = np.dot(rotation_matrix(params), coord_det)
    x = np.reshape(x, detector.raw_shape)
    y = np.reshape(y, detector.raw_shape)
    z = np.reshape(z, detector.raw_shape)
    return x, y, z


def calculate_2theta(
    detector: pyFAI.detectors.Detector, params: Optional[list] = None
) -> np.ndarray:
    """
    Calculate the 2θ angles for the detector based on the geometry parameters.

    Parameters
    ----------
    detector : pyFAI.detectors.Detector
        PyFAI detector object containing pixel coordinates to be corrected.
    params : list, optional
        6 Geometry parameters: distance, x-shift, y-shift, Rx, Ry, Rz
    """
    x, y, z = correct_geom(detector, params)
    tth = np.zeros(detector.raw_shape)
    for p in range(detector.n_modules):
        tth[p] = np.arctan2(np.sqrt(x[p] * x[p] + y[p] * y[p]), z[p])
    return tth


def calculate_radius(
    detector: pyFAI.detectors.Detector, params: Optional[list] = None
) -> np.ndarray:
    """
    Calculate the radius for each pixel based on the geometry parameters.

    Parameters
    ----------
    detector  : pyFAI.Detector
        pyFAI detector object
    params : list, optional
        6 Geometry parameters: distance, x-shift, y-shift, Rx, Ry, Rz

    Returns
    -------
    r : numpy.ndarray, with input shape
        map of pixels' radii
    """
    x, y, _ = correct_geom(detector, params)
    r = np.zeros(detector.raw_shape)
    for p in range(detector.n_modules):
        r[p] = np.sqrt(x[p] ** 2 + y[p] ** 2)
    return r


def azimuthal_integration(
    powder: npt.NDArray[np.float64],
    detector: pyFAI.detectors.Detector,
    params: Optional[list] = None,
) -> tuple:
    """
    Compute the radial intensity profile of an image.

    Parameters
    ----------
    powder : numpy.ndarray, shape (n,m)
        detector image
    detector : pyFAI.Detector
        PyFAI detector object
    params : list, optional
        6 Geometry parameters: distance, x-shift, y-shift, Rx, Ry, Rz
    """
    r = calculate_radius(detector, params)
    intensity, bin_edges = np.histogram(
        r.ravel(), bins=1000, range=(r.min(), r.max()), weights=powder.ravel()
    )
    count, _ = np.histogram(r.ravel(), bins=bin_edges)
    radialprofile = np.divide(
        intensity, count, out=np.zeros_like(intensity), where=count != 0
    )
    r_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return radialprofile, r_centers


def r2q(radii: np.ndarray, distance: float, wavelength: float) -> np.ndarray:
    """
    Convert pixel radii to scattering vector magnitude q.

    Parameters
    ----------
    radii : numpy.ndarray, 1d
        radius in meter from beam center
    distance : float
        detector distance in meter
    wavelength : float
        X-ray wavelength in meter

    Returns
    -------
    qs: numpy.ndarray, 1d
        magnitude of q-vector in per Angstrom
    """
    theta = np.arctan2(radii, distance)
    qs = 4.0 * np.pi * np.sin(theta / 2.0) / (wavelength * 1e10)
    return qs


class BayFAIOpt:
    """
    Class to run BayFAI optimization on a powder image.

    Parameters
    ----------
    exp : str
        Experiment name
    run : int
        Run number
    detector : PyFAI.Detector
        PyFAI detector object
    powder : np.ndarray
        Powder pattern data
    calibrant : PyFAI.Calibrant
        Calibrant object
    fixed : list
        List of parameters to keep fixed during optimization
    """

    def __init__(
        self,
        exp,
        run,
        detector,
        powder,
        calibrant,
        fixed,
    ):
        self.exp = exp
        self.run = run
        self.detname = detector.name
        self.detector = detector
        self.powder = powder
        self.stacked_powder = np.reshape(powder, detector.shape)
        self.calibrant = calibrant
        self.calibrant_name = os.path.splitext(os.path.basename(calibrant.filename))[0][
            6:
        ]
        self.tth = np.array(self.calibrant.get_2th())
        self.fixed = fixed
        self.parallelized = ["dist"]
        self.order = ["dist", "poni1", "poni2", "rot1", "rot2", "rot3"]
        self.space = []
        for p in self.order:
            if p not in self.fixed and p not in self.parallelized:
                self.space.append(p)
        self.ds = DataSource(exp=exp, run=run)
        group: MPI.Group = self.ds.comms._bd_only_group
        self.comm = MPI.COMM_WORLD.Create_group(group)
        if self.comm != MPI.COMM_NULL:
            self.rank = self.comm.Get_rank()
            self.size = self.comm.Get_size()
            if self.rank == 0:
                logger.info(f"Getting {self.size} processes for BayFAIOpt psana2 task")
        else:
            self.rank = -1
            self.size = -1

    @staticmethod
    def UCB(X, gp_model, visited_idx, beta=1.96):
        y_pred, y_std = gp_model.predict(X, return_std=True)
        ucb = y_pred + beta * y_std
        ucb[visited_idx] = -np.inf
        next = np.argmax(ucb)
        return next

    @staticmethod
    def q_UCB(X, gp_model, q, visited_idx, beta=1.96):
        y_pred, y_std = gp_model.predict(X, return_std=True)
        ucb = y_pred + beta * y_std
        ucb[visited_idx] = -np.inf
        top_next = np.argsort(ucb)[-q:]
        return top_next

    def distribute_distances(self, center, res):
        """
        Distribute distances across MPI ranks.

        Parameters
        ----------
        center : dict
            Center values for each parameter
        res : float
            Resolution of the grid used to discretize the parameter search space

        Returns
        -------
        dist : float
            The distance assigned to this MPI rank
        """
        low = center["dist"] - res["dist"] * self.size / 2
        high = center["dist"] + res["dist"] * self.size / 2
        distances = np.linspace(low, high - res["dist"], self.size)
        self.distances = distances
        dist = distances[self.rank]
        return dist

    def create_search_space(self, dist, center, bounds, res):
        """
        Discretize the search space for the free parameters.

        Parameters
        ----------
        dist : float
            Distance on this MPI rank
        center : dict
            Center values for each parameter
        bounds : dict
            Bounds for each parameter, format: {param: (lower, upper)}
        res : dict
            Resolution per parameter

        Returns
        -------
        X : np.ndarray
            Full 6D geometry space (cartesian product)
        X_norm : np.ndarray
            Normalized search space (between-1 and 1)
        """
        center["dist"] = dist
        full_params = {}
        search_params = {}
        for p in self.order:
            if p in self.space:
                low = center[p] + bounds[p][0]
                high = center[p] + bounds[p][1]
                if high < low:
                    low, high = high, low
                step = res[p]
                full_params[p] = np.arange(low, high + step, step)
                search_params[p] = full_params[p]
            else:
                full_params[p] = np.array([center[p]])

        X = np.array(np.meshgrid(*[full_params[p] for p in self.order])).T.reshape(
            -1, len(self.order)
        )
        X_search = np.array(
            np.meshgrid(*[search_params[p] for p in self.space])
        ).T.reshape(-1, len(self.space))
        self.mins = np.min(X_search, axis=0)
        self.maxs = np.max(X_search, axis=0)
        X_norm = 2 * (X_search - self.mins) / (self.maxs - self.mins) - 1
        return X, X_norm

    def sample_initial_points(self, X, X_norm, center, bounds, n_samples, prior):
        """
        Sample initial points from the search space.

        Parameters
        ----------
        X : np.ndarray
            Search space
        X_norm : np.ndarray
            Normalized search space
        center : dict
            Center values for each parameter
        bounds : dict
            Bounds for each parameter
        n_samples : int
            Number of samples to draw
        prior : bool
            Use prior information for sampling

        Returns
        -------
        np.ndarray
            Sampled points
        """
        if prior:
            means = [center[p] for p in self.space]
            cov = np.diag(
                [(np.abs((bounds[p][1] - bounds[p][0])) / 5) ** 2 for p in self.space]
            )
            X_free = np.random.multivariate_normal(means, cov, n_samples)
            X_free = np.clip(X_free, self.mins, self.maxs)
            X_norm_samples = 2 * (X_free - self.mins) / (self.maxs - self.mins) - 1
            X_samples = np.tile([center[p] for p in self.order], (n_samples, 1))
            for i, p in enumerate(self.space):
                j = self.order.index(p)
                X_samples[:, j] = X_free[:, i]
            return X_samples, X_norm_samples
        else:
            idx_samples = np.random.choice(X.shape[0], n_samples)
            X_samples = X[idx_samples]
            X_norm_samples = X_norm[idx_samples]
            return X_samples, X_norm_samples

    def score(self, sample, Imin, max_rings):
        """
        Evaluate score at a given sampled geometry.

        Parameters
        ----------
        sample : array-like
            Geometry parameters
        Imin : float
            Minimum intensity threshold
        max_rings : int
            Maximum number of rings to consider

        Returns
        -------
        score : float
            Scalar score for Bayesian optimization
        """
        dist, poni1, poni2, rot1, rot2, rot3 = sample
        geom_sample = Geometry(
            dist=dist,
            poni1=poni1,
            poni2=poni2,
            rot1=rot1,
            rot2=rot2,
            rot3=rot3,
            detector=self.detector,
            wavelength=self.calibrant.wavelength,
        )
        sg = SingleGeometry(
            "Score Geometry",
            self.stacked_powder,
            calibrant=self.calibrant,
            detector=self.detector,
            geometry=geom_sample,
        )
        sg.extract_cp(max_rings=max_rings, pts_per_deg=1, Imin=Imin)
        score = len(sg.geometry_refinement.data)
        return score

    def pyFAI_score(self, best_param, Imin, max_rings):
        """
        Evaluate geometry found by BO on pyFAI refinement tool

        Parameters
        ----------
        best_param : list
            Best parameters found by Bayesian optimization
        Imin : float
            Minimum intensity threshold
        max_rings : int
            Maximum number of rings to consider
        rtol : float
            Relative tolerance for masking ring pixels

        Returns
        -------
        residual : float
            Residual error after refinement
        score : float
            BO Score of the refined parameters
        params : dict
            Refined parameters
        """
        dist, poni1, poni2, rot1, rot2, rot3 = best_param
        best_geom = Geometry(
            dist=dist,
            poni1=poni1,
            poni2=poni2,
            rot1=rot1,
            rot2=rot2,
            rot3=rot3,
            detector=self.detector,
            wavelength=self.calibrant.wavelength,
        )
        sg = SingleGeometry(
            "Best Geometry",
            self.stacked_powder,
            calibrant=self.calibrant,
            detector=self.detector,
            geometry=best_geom,
        )
        sg.extract_cp(max_rings=max_rings, pts_per_deg=1, Imin=Imin)
        self.sg = sg
        residual = 0
        if len(sg.geometry_refinement.data) > 0:
            residual = sg.geometry_refinement.refine3(fix=["wavelength"])
        params = sg.geometry_refinement.param
        score = self.score(params, Imin, max_rings)
        return residual, score, params

    @ignore_warnings(category=ConvergenceWarning)
    def bayes_opt_distance(
        self,
        dist,
        center,
        bounds,
        res,
        n_samples,
        n_iterations,
        Imin,
        max_rings,
        beta=1.96,
        prior=True,
        seed=0,
    ):
        """
        Run Bayesian Optimization on a subspace of fixed distance.

        Parameters
        ----------
        dist : float
            Distance on this MPI rank
        center : dict
            Dictionary of center values for each parameter
        bounds : dict
            Dictionary of bounds for each parameter
        res : dict
            Dictionary of resolution for each parameter
        n_samples : int
            Number of samples to initialize the Gaussian Process
        n_iterations : int
            Number of iterations of Bayesian Optimization
        Imin : float
            Minimum intensity threshold for identifying Bragg peaks
        max_rings : int
            Maximum number of rings to search for Bragg peaks
        beta : float
            Exploration-exploitation trade-off parameter for UCB acquisition function
        prior : bool
            Whether to sample initial points around the center or randomly
        seed : int
            Random seed for reproducibility
        """
        np.random.seed(seed)

        # 1. Create the search space
        X, X_norm = self.create_search_space(dist, center, bounds, res)

        # 2. Sample initial points
        X_samples, X_norm_samples = self.sample_initial_points(
            X, X_norm, center, bounds, n_samples, prior
        )

        # 3. Evaluate the initial points
        bo_history = {"params": [], "scores": []}
        y = np.zeros((n_samples))
        for i in range(n_samples):
            y[i] = self.score(X_samples[i], Imin, max_rings)
            bo_history["params"].append(X_samples[i])
            bo_history["scores"].append(y[i])

        if np.all(y == 0):
            result = {
                "bo_history": bo_history,
                "params": [dist, 0, 0, 0, 0, 0],
                "residual": 0,
                "score": 0,
                "best_idx": 0,
            }
            logger.warning(
                f"All samples have score 0 for dist={dist}. Skipping Bayesian Optimization."
            )
            return result

        y[np.isnan(y)] = 0
        if np.std(y) != 0:
            y_norm = (y - np.mean(y)) / np.std(y)
        else:
            y_norm = y - np.mean(y)

        # 4. Initialize the Gaussian Process model
        kernel = RBF(length_scale=0.3, length_scale_bounds=(0.2, 0.4)) * ConstantKernel(
            constant_value=1.0, constant_value_bounds=(0.5, 1.5)
        ) + WhiteKernel(noise_level=0.001, noise_level_bounds="fixed")
        gp_model = GaussianProcessRegressor(
            kernel=kernel, n_restarts_optimizer=10, random_state=0
        )
        gp_model.fit(X_norm_samples, y_norm)
        visited_idx = list([])

        # 5. Run the Bayesian Optimization loop
        for i in range(n_iterations):
            # 6. Select the next point to evaluate
            next = self.UCB(X_norm, gp_model, visited_idx, beta)
            next_sample = X[next]
            visited_idx.append(next)

            # 7. Compute the score of the next point
            score = self.score(next_sample, Imin, max_rings)
            y = np.append(y, [score], axis=0)
            bo_history["params"].append(next_sample)
            bo_history["scores"].append(score)
            X_samples = np.append(X_samples, [X[next]], axis=0)
            X_norm_samples = np.append(X_norm_samples, [X_norm[next]], axis=0)
            if np.std(y) != 0:
                y_norm = (y - np.mean(y)) / np.std(y)
            else:
                y_norm = y - np.mean(y)

            # 8. Update the Gaussian Process model
            gp_model.fit(X_norm_samples, y_norm)

        # 9. Gather results
        best_idx = np.argmax(y)
        best_param = X_samples[best_idx]
        residual, score, params = self.pyFAI_score(best_param, Imin, max_rings)
        logger.info(
            f"Rank {self.rank} dist={dist:.4f}m: score={score}, residual={residual:3e}"
        )
        result = {
            "bo_history": bo_history,
            "params": params,
            "residual": residual,
            "score": score,
            "best_idx": best_idx,
        }
        return result

    def bayfai_opt(
        self,
        center,
        bounds,
        res,
        n_samples,
        n_iterations,
        Imin,
        max_rings,
        beta=1.96,
        prior=True,
        seed=0,
    ):
        """
        Run BayFAI optimization.
        Split the distance parameter across MPI ranks.
        Run Bayesian Optimization on each rank with fixed distance.
        Perform pyFAI least-squares refinement for each rank's best geometry.
        Optimal geometry is chosen based on the lowest residual among ranks.

        Parameters
        ----------
        center : dict
            Dictionary of center values for each parameter
        bounds : dict
            Dictionary of bounds for each parameter
        res : dict
            Dictionary of resolution for each parameter
        n_samples : int
            Number of samples to initialize the Gaussian Process
        n_iterations : int
            Number of iterations of Bayesian Optimization
        Imin : float
            Minimum intensity threshold for identifying Bragg peaks
        max_rings : int
            Maximum number of rings to consider
        beta : float
            Exploration-exploitation trade-off parameter for UCB acquisition function
        prior : bool
            Whether to sample initial points around the center or randomly
        seed : int
            Random seed for reproducibility
        """
        if self.ds.comms.bd_comm.Get_rank() == 0:
            return

        if self.comm != MPI.COMM_NULL:
            dist = self.distribute_distances(center, res)
            logger.info(
                f"Rank {self.rank}: Running Bayesian Optimization on distance {dist:.4f} m"
            )

            bayfai_hyperparams = {
                "n_samples": n_samples,
                "n_iterations": n_iterations,
                "Imin": Imin,
                "max_rings": max_rings,
                "beta": beta,
                "prior": prior,
                "seed": seed,
            }

            results = self.bayes_opt_distance(
                dist,
                center,
                bounds,
                res,
                **bayfai_hyperparams,
            )

            self.comm.Barrier()

            self.scan = {}
            self.scan["bo_history"] = self.comm.gather(results["bo_history"], root=0)
            self.scan["params"] = self.comm.gather(results["params"], root=0)
            self.scan["residual"] = self.comm.gather(results["residual"], root=0)
            self.scan["score"] = self.comm.gather(results["score"], root=0)
            self.scan["best_idx"] = self.comm.gather(results["best_idx"], root=0)
            self.finalize()

    def finalize(self):
        if self.rank == 0:
            for key in self.scan.keys():
                self.scan[key] = np.array([item for item in self.scan[key]])
            non_zeros = np.where(self.scan["score"] > 0)[0]
            thrsh = np.percentile(self.scan["score"][non_zeros], 25)
            self.thrsh = thrsh
            score_indices = np.where(self.scan["score"] > thrsh)[0]
            shift_index = np.argmin(self.scan["residual"][score_indices])
            index = score_indices[shift_index]
            self.index = index
            self.bo_history = self.scan["bo_history"][index]
            self.params = self.scan["params"][index]
            self.residual = self.scan["residual"][index]
            self.best_score = self.scan["score"][index]
            self.best_idx = self.scan["best_idx"][index]
            self.gr = GeometryRefinement(
                calibrant=self.calibrant,
                dist=self.params[0],
                poni1=self.params[1],
                poni2=self.params[2],
                rot1=self.params[3],
                rot2=self.params[4],
                rot3=self.params[5],
                detector=self.detector,
                wavelength=self.calibrant.wavelength,
            )

    def plot_radial_integration(self, qs, radial, calibrant, ax=None):
        """
        Plot the radial integration of a powder image

        Parameters
        ----------
        qs : np.array
            q-space range covered by detector
        radial : np.array
            Radial intensity profile
        calibrant : Calibrant
            Calibrant object
        ax : plt.Axes
            Matplotlib axes
        """
        if ax is None:
            fig, ax = plt.subplots()

        unit = RADIAL_UNITS["q_A^-1"]
        ax.plot(qs, radial, color="black", linewidth=0.8)

        x_values = calibrant.get_peaks(unit)
        if x_values is not None:
            for x in x_values:
                line = lines.Line2D(
                    [x, x],
                    ax.axis()[2:4],
                    color="red",
                    linestyle="--",
                    linewidth=0.8,
                    alpha=0.7,
                )
                ax.add_line(line)

        ax.set_title("Radial Profile", fontsize=6)
        if unit:
            ax.set_xlabel(unit.label, fontsize=6)
        ax.set_ylabel("Intensity", fontsize=6)
        ax.tick_params(axis="x", labelsize=4)
        ax.tick_params(axis="y", labelsize=4)

    def plot_bo_history(self, ax):
        """
        Plot the Bayesian Optimization history across all ranks

        Parameters
        ----------
        bo_history : dict
            Dictionary containing the BO history with keys 'params' and 'scores' for each rank-distance
        ax : plt.Axes
            Matplotlib axes
        """
        bo_history = self.scan["bo_history"]
        iters = np.arange(len(bo_history[0]["scores"]))
        for r in range(len(bo_history)):
            score = bo_history[r]["scores"]
            ax.plot(iters, score, marker="o", markersize=1, linestyle="None", alpha=0.6)
        ax.plot(
            iters,
            self.bo_history["scores"],
            marker="o",
            markersize=3,
            linestyle="--",
            linewidth=0.8,
            color="black",
            markerfacecolor="red",
            markeredgecolor="black",
            label=f"Best Distance (m): {self.distances[self.index]:.3f}",
        )
        ax.legend(fontsize=6)
        ax.set_xlabel("Iteration", fontsize=6)
        ax.set_ylabel("Score", fontsize=6)
        ax.tick_params(axis="x", labelsize=4)
        ax.tick_params(axis="y", labelsize=4)
        ax.set_title("Bayesian Optimization History", fontsize=6)

    def plot_score_distance_scan(self, ax):
        """
        Plot the score scan over distance

        Parameters
        ----------
        ax : plt.Axes
            Matplotlib axes
        """
        ax.plot(self.distances, self.scan["score"], linewidth=0.8, color="black")
        ax.axhline(
            self.thrsh,
            color="red",
            linestyle="--",
            label=f"Threshold score: {self.thrsh}",
            linewidth=0.8,
        )
        ax.legend(fontsize=6)
        ax.set_xlabel("Distance (m)", fontsize=6)
        ax.set_ylabel("Score", fontsize=6)
        ax.tick_params(axis="x", labelsize=4)
        ax.tick_params(axis="y", labelsize=4)
        ax.set_title("Bragg Peaks Found vs Distance", fontsize=6)

    def plot_residual_distance_scan(self, refined_dist, ax):
        """
        Plot the residual scan over distance

        Parameters
        ----------
        refined_dist : float
            Refined distance
        ax : plt.Axes
            Matplotlib axes
        """
        ax.plot(self.distances, self.scan["residual"], linewidth=0.8, color="black")
        best_dist = self.distances[self.index]
        ax.axvline(
            best_dist,
            color="green",
            linestyle="--",
            label=f"Best distance (m): {best_dist:.3f}",
            linewidth=0.8,
        )
        ax.axvline(
            refined_dist,
            color="red",
            linestyle="--",
            label=f"Refined distance (m): {refined_dist:.3f}",
            linewidth=0.8,
        )
        ax.legend(fontsize=6)
        ax.set_xlabel("Distance (m)", fontsize=6)
        ax.set_ylabel("Residual", fontsize=6)
        ax.tick_params(axis="x", labelsize=4)
        ax.tick_params(axis="y", labelsize=4)
        ax.set_title("PyFAI Residual vs Distance", fontsize=6)

    def plot_intensity_hist(self, powder, Imin, ax):
        """
        Plot histogram of pixel intensities in the powder image

        Parameters
        ----------
        powder : np.ndarray
            Powder image
        exp : str
            Experiment name
        run : int
            Run number
        Imin : float
            Minimum intensity threshold for identifying Bragg peaks
        ax : plt.Axes
            Matplotlib axes
        """
        threshold = np.mean(powder) + 3 * np.std(powder)
        nice_pix = powder < threshold
        mean = np.mean(powder[nice_pix])
        std_dev = np.std(powder[nice_pix])
        _ = ax.hist(
            powder[nice_pix],
            bins=200,
            color="skyblue",
            edgecolor="black",
            alpha=0.7,
            label="Pixel Intensities",
            orientation="horizontal",
        )
        ax.axhline(
            mean,
            color="red",
            linestyle="--",
            label=f"Mean ({mean:.2f})",
        )
        ax.axhline(
            mean + std_dev,
            color="orange",
            linestyle="--",
            label=f"Mean + Std Dev ({mean + std_dev:.2f})",
        )
        ax.axhline(
            mean + 2 * std_dev,
            color="green",
            linestyle="--",
            label=f"Mean + 2 Std Dev ({mean + 2 * std_dev:.2f})",
        )
        ax.axhline(
            Imin,
            color="purple",
            linestyle=":",
            linewidth=2,
            label=f"Threshold ({Imin:.2f})",
        )
        ax.set_xlim([0, 100000])
        ax.set_ylim([0, mean + 3 * std_dev])
        ax.set_ylabel("Pixel Intensity", fontsize=6)
        ax.set_xlabel("Frequency", fontsize=6)
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.tick_params(axis="y", labelsize=4)
        ax.set_title(
            f"Histogram of Pixel Intensities \n for {self.exp} run {self.run}", fontsize=6
        )
        ax.legend(fontsize=6)

    def plot_powder_and_resolution(self, powder, detector, distance, ax=None):
        """
        Plot the powder image with calibrated overlapping 2θ rings.

        Parameters
        ----------
        powder : np.ndarray
            Powder image
        detector : PyFAI(Detector)
            Corrected PyFAI detector object
        distance : float
            Distance of the detector
        """
        if ax is None:
            _fig, ax = plt.subplots()
        y, x, _ = correct_geom(detector, params=[distance, 0, 0, 0, 0, 0])

        xmin, xmax = x.min(), x.max()
        ymin, ymax = y.min(), y.max()
        if xmin < 0 and ymin < 0 and xmax > 0 and ymax > 0:
            ax.set_xlim(xmin * 1.1, xmax * 1.1)
            ax.set_ylim(ymin * 1.1, ymax * 1.1)
        elif xmin < 0 and ymin < 0 and xmax < 0 and ymax < 0:
            ax.set_xlim(xmin * 1.1, xmax * 0.9)
            ax.set_ylim(ymin * 1.1, ymax * 0.9)
        elif xmin < 0 and ymin > 0 and xmax > 0 and ymax > 0:
            ax.set_xlim(xmin * 1.1, xmax * 1.1)
            ax.set_ylim(ymin * 0.9, ymax * 1.1)
        elif xmin > 0 and ymin < 0 and xmax > 0 and ymax > 0:
            ax.set_xlim(xmin * 0.9, xmax * 1.1)
            ax.set_ylim(ymin * 1.1, ymax * 1.1)
        elif xmin < 0 and ymin < 0 and xmax > 0 and ymax < 0:
            ax.set_xlim(xmin * 1.1, xmax * 1.1)
            ax.set_ylim(ymin * 1.1, ymax * 0.9)
        elif xmin < 0 and ymin < 0 and xmax < 0 and ymax > 0:
            ax.set_xlim(xmin * 1.1, xmax * 0.9)
            ax.set_ylim(ymin * 1.1, ymax * 1.1)
        elif xmin < 0 and ymin > 0 and xmax < 0 and ymax > 0:
            ax.set_xlim(xmin * 1.1, xmax * 0.9)
            ax.set_ylim(ymin * 0.9, ymax * 1.1)
        elif xmin > 0 and ymin < 0 and xmax > 0 and ymax < 0:
            ax.set_xlim(xmin * 0.9, xmax * 1.1)
            ax.set_ylim(ymin * 1.1, ymax * 0.9)
        elif xmin > 0 and ymin > 0 and xmax > 0 and ymax > 0:
            ax.set_xlim(xmin * 0.9, xmax * 1.1)
            ax.set_ylim(ymin * 0.9, ymax * 1.1)

        ax.scatter(
            x.ravel(),
            y.ravel(),
            c=powder.ravel(),
            s=3,
            edgecolors=None,
            linewidth=0,
            vmin=np.percentile(powder, 5),
            vmax=np.percentile(powder, 95),
        )

        ttha = calculate_2theta(detector, params=[distance, 0, 0, 0, 0, 0])
        for i in range(detector.n_modules):
            ax.contour(
                x[i],
                y[i],
                ttha[i],
                levels=self.tth,
                cmap="autumn",
                linewidths=1,
                linestyles="dashed",
            )

        radii = calculate_radius(detector)
        closest_pixel_index = np.argmin(radii)
        closest_pixel = radii.flatten()[closest_pixel_index]
        closest_q = r2q(closest_pixel, distance, self.calibrant.wavelength)
        closest_resol = 2 * np.pi / closest_q
        furthest_pixel_index = np.argmax(radii)
        furthest_pixel = radii.flatten()[furthest_pixel_index]
        furthest_q = r2q(furthest_pixel, distance, self.calibrant.wavelength)
        furthest_resol = 2 * np.pi / furthest_q
        d_left = abs(xmin)
        d_right = abs(xmax)
        d_bottom = abs(ymin)
        d_top = abs(ymax)
        border_distances = [d_left, d_right, d_bottom, d_top]
        border_pixel = max(border_distances)
        border_q = r2q(border_pixel, distance, self.calibrant.wavelength)
        border_resol = 2 * np.pi / border_q
        border_2_q = r2q(border_pixel / 2, distance, self.calibrant.wavelength)
        border_2_resol = 2 * np.pi / border_2_q

        radius_lvls = np.array(
            [closest_pixel, border_pixel / 2, border_pixel, furthest_pixel]
        )
        resol_lvls = np.array(
            [closest_resol, border_2_resol, border_resol, furthest_resol]
        )
        for i in range(detector.n_modules):
            ax.contour(
                x[i],
                y[i],
                radii[i],
                levels=radius_lvls,
                cmap="summer",
                linewidths=1,
                linestyles="dashed",
            )
        for radius, resol in zip(radius_lvls, resol_lvls):
            text_x = radius / np.sqrt(2)
            text_y = radius / np.sqrt(2)
            ax.text(
                text_x,
                text_y,
                f"{resol:.3f} \u00c5",
                color="red",
                fontsize=8,
                bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=1),
            )
        ax.set_xlabel("X-axis (m)", fontsize=8)
        ax.set_ylabel("Y-axis (m)", fontsize=8)
        ax.tick_params(axis="x", labelsize=6)
        ax.tick_params(axis="y", labelsize=6)
        ax.set_title(
            f"Run {self.run} - {self.detname} - {self.calibrant_name}", fontsize=8
        )
        ax.set_aspect("equal")

    def create_interactive_powder(
        self,
        powder,
        detector,
        distance,
    ):
        """
        Create an interactive powder image with calibrated overlapping 2θ rings.

        Parameters
        ----------
        powder : np.ndarray
            Powder image
        detector : PyFAI(Detector)
            Corrected PyFAI detector object
        distance : float
            Refined distance
        """
        y, x, _ = correct_geom(detector, params=[distance, 0, 0, 0, 0, 0])

        xmin, xmax = x.min(), x.max()
        ymin, ymax = y.min(), y.max()

        if xmin < 0 and ymin < 0 and xmax > 0 and ymax > 0:
            xlim = (xmin * 1.1, xmax * 1.1)
            ylim = (ymin * 1.1, ymax * 1.1)
        elif xmin < 0 and ymin < 0 and xmax < 0 and ymax < 0:
            xlim = (xmin * 1.1, xmax * 0.9)
            ylim = (ymin * 1.1, ymax * 0.9)
        elif xmin < 0 and ymin > 0 and xmax > 0 and ymax > 0:
            xlim = (xmin * 1.1, xmax * 1.1)
            ylim = (ymin * 0.9, ymax * 1.1)
        elif xmin > 0 and ymin < 0 and xmax > 0 and ymax > 0:
            xlim = (xmin * 0.9, xmax * 1.1)
            ylim = (ymin * 1.1, ymax * 1.1)
        elif xmin < 0 and ymin < 0 and xmax > 0 and ymax < 0:
            xlim = (xmin * 1.1, xmax * 1.1)
            ylim = (ymin * 1.1, ymax * 0.9)
        elif xmin < 0 and ymin < 0 and xmax < 0 and ymax > 0:
            xlim = (xmin * 1.1, xmax * 0.9)
            ylim = (ymin * 1.1, ymax * 1.1)
        elif xmin < 0 and ymin > 0 and xmax < 0 and ymax > 0:
            xlim = (xmin * 1.1, xmax * 0.9)
            ylim = (ymin * 0.9, ymax * 1.1)
        elif xmin > 0 and ymin < 0 and xmax > 0 and ymax < 0:
            xlim = (xmin * 0.9, xmax * 1.1)
            ylim = (ymin * 1.1, ymax * 0.9)
        elif xmin > 0 and ymin > 0 and xmax > 0 and ymax > 0:
            xlim = (xmin * 0.9, xmax * 1.1)
            ylim = (ymin * 0.9, ymax * 1.1)

        p = figure(
            title=f"Run {self.run} - {self.detname} - {self.calibrant_name}",
            x_axis_label="X-axis (m)",
            y_axis_label="Y-axis (m)",
            width=1200,
            height=1200,
            match_aspect=True,
            x_range=xlim,
            y_range=ylim,
        )

        vmin, vmax = np.percentile(powder, 5), np.percentile(powder, 95)
        color_mapper = LinearColorMapper(palette=Viridis256, low=vmin, high=vmax)

        source = ColumnDataSource(
            data={"x": x.ravel(), "y": y.ravel(), "intensity": powder.ravel()}
        )

        _ = p.scatter(
            x="x",
            y="y",
            size=3,
            color={"field": "intensity", "transform": color_mapper},
            line_color=None,
            source=source,
        )

        _ = ColorBar(
            color_mapper=color_mapper, width=8, location=(0, 0), title="Intensity"
        )

        ttha = calculate_2theta(detector, params=[distance, 0, 0, 0, 0, 0])
        for i in range(detector.n_modules):
            p.contour(
                x=x[i],
                y=y[i],
                z=ttha[i],
                levels=self.tth,
                line_color="red",
                line_width=3,
                line_dash="dashed",
            )

        radii = calculate_radius(detector, params=[distance, 0, 0, 0, 0, 0])
        closest_pixel_index = np.argmin(radii)
        closest_pixel = radii.flatten()[closest_pixel_index]
        closest_q = r2q(closest_pixel, distance, self.calibrant.wavelength)
        closest_resol = 2 * np.pi / closest_q

        furthest_pixel_index = np.argmax(radii)
        furthest_pixel = radii.flatten()[furthest_pixel_index]
        furthest_q = r2q(furthest_pixel, distance, self.calibrant.wavelength)
        furthest_resol = 2 * np.pi / furthest_q

        d_left = abs(xmin)
        d_right = abs(xmax)
        d_bottom = abs(ymin)
        d_top = abs(ymax)
        border_distances = [d_left, d_right, d_bottom, d_top]
        border_pixel = max(border_distances)
        border_q = r2q(border_pixel, distance, self.calibrant.wavelength)
        border_resol = 2 * np.pi / border_q
        border_2_q = r2q(border_pixel / 2, distance, self.calibrant.wavelength)
        border_2_resol = 2 * np.pi / border_2_q

        radius_lvls = np.array(
            [closest_pixel, border_pixel / 2, border_pixel, furthest_pixel]
        )
        resol_lvls = np.array(
            [closest_resol, border_2_resol, border_resol, furthest_resol]
        )
        for i in range(detector.n_modules):
            p.contour(
                x=x[i],
                y=y[i],
                z=radii[i],
                levels=radius_lvls,
                line_color="green",
                line_width=3,
                line_dash="dashed",
            )
        for radius, resol in zip(radius_lvls, resol_lvls):
            text_x = radius / np.sqrt(2)
            text_y = radius / np.sqrt(2)
            label_annotation = Label(
                x=text_x,
                y=text_y,
                text=f"{resol:.3f} Å",
                text_color="red",
                text_font_size="16pt",
            )
            p.add_layout(label_annotation)

        hover = HoverTool(
            tooltips=[
                ("x", "@x{0.000}"),
                ("y", "@y{0.000}"),
                ("Intensity", "@intensity{0.0}"),
            ]
        )
        p.add_tools(hover)

        p.title.text_font_size = "12pt"
        p.xaxis.axis_label_text_font_size = "10pt"
        p.yaxis.axis_label_text_font_size = "10pt"
        p.xaxis.major_label_text_font_size = "8pt"
        p.yaxis.major_label_text_font_size = "8pt"

        return (
            p,
            closest_q,
            closest_resol,
            furthest_q,
            furthest_resol,
            border_q,
            border_resol,
        )

    def create_diagnostics_panel(
        self,
        powder,
        Imin,
        detector,
        distance,
        low_resolution=None,
        high_resolution=None,
        border_resolution=None,
        plot="",
    ):
        """
        Create a diagnostics panel with the results of the Bayesian Optimization.

        Parameters
        ----------
        powder : np.ndarray
            Powder image
        detector : PyFAI(Detector)
            Corrected PyFAI detector object
        distance : float
            Refined distance
        low_resolution : float, optional
            Lowest resolution value, if available
        high_resolution : float, optional
            Highest resolution value, if available
        border_resolution : float, optional
            Border resolution value, if available
        plot : str
            Path to save plot
        """
        fig = plt.figure(figsize=(6, 9), dpi=100)
        nrow, ncol = 3, 2
        irow, icol = 0, 0

        # Labelling experiment and run number
        ax1 = plt.subplot2grid((nrow, ncol), (irow, icol))
        rect = patches.Rectangle(
            (0, 0),
            1,
            1,
            transform=ax1.transAxes,
            color="lightgrey",
            alpha=0.3,
        )
        ax1.add_patch(rect)
        ax1.text(
            0.05,
            0.9,
            f"Experiment {self.exp}",
            ha="left",
            va="center",
            fontsize=8,
        )
        ax1.text(0.05, 0.8, f"Run {self.run}", ha="left", va="center", fontsize=8)
        ax1.text(
            0.05, 0.7, f"Detector {self.detname}", ha="left", va="center", fontsize=8
        )
        ax1.text(
            0.05,
            0.6,
            f"Calibrant {self.calibrant_name}",
            ha="left",
            va="center",
            fontsize=8,
        )
        ax1.text(
            0.05,
            0.5,
            f"Distance = {distance:.4f} m",
            ha="left",
            va="center",
            fontsize=8,
        )
        if low_resolution is not None:
            ax1.text(
                0.05,
                0.4,
                f"{'Low-q Resolution':<30}",
                ha="left",
                va="center",
                fontsize=8,
                color="black",
            )
            ax1.text(
                0.50,
                0.4,
                f"{low_resolution:.3f} \u00c5",
                ha="left",
                va="center",
                fontsize=8,
                color="red",
            )
            ax1.text(
                0.05,
                0.3,
                f"{'Border Resolution':<30}",
                ha="left",
                va="center",
                fontsize=8,
                color="black",
            )
            ax1.text(
                0.50,
                0.3,
                f"{border_resolution:.3f} \u00c5",
                ha="left",
                va="center",
                fontsize=8,
                color="red",
            )
            ax1.text(
                0.05,
                0.2,
                f"{'Corner Resolution':<30}",
                ha="left",
                va="center",
                fontsize=8,
                color="black",
            )
            ax1.text(
                0.50,
                0.2,
                f"{high_resolution:.3f} \u00c5",
                ha="left",
                va="center",
                fontsize=8,
                color="red",
            )
        ax1.axis("off")
        icol += 1

        # Plotting histogram of pixel intensities
        ax2 = plt.subplot2grid((nrow, ncol), (irow, icol))
        self.plot_intensity_hist(powder, Imin, ax2)
        icol = 0
        irow += 1

        # Plotting radial profiles with peaks
        ax3 = plt.subplot2grid((nrow, ncol), (irow, icol), colspan=2)
        profile, radii = azimuthal_integration(powder, detector)
        qs = r2q(radii, distance, self.calibrant.wavelength)
        self.plot_radial_integration(qs, profile, self.calibrant, ax3)
        irow += 1

        # Plotting score scan over distance
        ax5 = plt.subplot2grid((nrow, ncol), (irow, icol))
        self.plot_bo_history(ax5)
        icol += 1

        # Plotting residual scan over distance
        ax6 = plt.subplot2grid((nrow, ncol), (irow, icol))
        self.plot_residual_distance_scan(distance, ax6)

        fig.tight_layout()

        if plot != "":
            fig.savefig(plot, dpi=100)
        return fig

    def create_summary_plot(
        self,
        powder,
        Imin,
        detector,
        distance,
        low_resolution=None,
        high_resolution=None,
        border_resolution=None,
        plot="",
    ):
        """
        Create a summary plot with the results of the Bayesian Optimization.

        Parameters
        ----------
        history : list
            List of BO history
        powder : np.ndarray
            Powder image
        Imin : float
            Minimum intensity threshold for identifying Bragg peaks
        detector : PyFAI(Detector)
            Corrected PyFAI detector object
        distance : float
            Refined distance
        low_resolution : float, optional
            Lowest resolution value, if available
        high_resolution : float, optional
            Highest resolution value, if available
        border_resolution : float, optional
            Border resolution value, if available
        plot : str
            Path to save plot
        """
        fig = plt.figure(figsize=(9, 12), dpi=100)
        nrow, ncol = 4, 3
        irow, icol = 0, 0

        # Labelling experiment and run number
        ax1 = plt.subplot2grid((nrow, ncol), (irow, icol))
        rect = patches.Rectangle(
            (0, 0),
            1,
            1,
            transform=ax1.transAxes,
            color="lightgrey",
            alpha=0.3,
        )
        ax1.add_patch(rect)
        ax1.text(
            0.05,
            0.9,
            f"Experiment {self.exp}",
            ha="left",
            va="center",
            fontsize=8,
        )
        ax1.text(0.05, 0.8, f"Run {self.run}", ha="left", va="center", fontsize=8)
        ax1.text(
            0.05, 0.7, f"Detector {self.detname}", ha="left", va="center", fontsize=8
        )
        ax1.text(
            0.05,
            0.6,
            f"Calibrant {self.calibrant_name}",
            ha="left",
            va="center",
            fontsize=8,
        )
        ax1.text(
            0.05,
            0.5,
            f"Distance = {distance:.4f} m",
            ha="left",
            va="center",
            fontsize=8,
        )
        if low_resolution is not None:
            ax1.text(
                0.05,
                0.4,
                f"{'Low-q Resolution':<30}",
                ha="left",
                va="center",
                fontsize=8,
                color="black",
            )
            ax1.text(
                0.50,
                0.4,
                f"{low_resolution:.3f} \u00c5",
                ha="left",
                va="center",
                fontsize=8,
                color="red",
            )
            ax1.text(
                0.05,
                0.3,
                f"{'Border Resolution':<30}",
                ha="left",
                va="center",
                fontsize=8,
                color="black",
            )
            ax1.text(
                0.50,
                0.3,
                f"{border_resolution:.3f} \u00c5",
                ha="left",
                va="center",
                fontsize=8,
                color="red",
            )
            ax1.text(
                0.05,
                0.2,
                f"{'Corner Resolution':<30}",
                ha="left",
                va="center",
                fontsize=8,
                color="black",
            )
            ax1.text(
                0.50,
                0.2,
                f"{high_resolution:.3f} \u00c5",
                ha="left",
                va="center",
                fontsize=8,
                color="red",
            )
        ax1.axis("off")
        icol += 1

        # Plotting radial profiles with peaks
        ax2 = plt.subplot2grid((nrow, ncol), (irow, icol), colspan=ncol - icol)
        masked_powder = powder
        profile, radii = azimuthal_integration(masked_powder, detector)
        qs = r2q(radii, distance, self.calibrant.wavelength)
        self.plot_radial_integration(qs, profile, self.calibrant, ax=ax2)
        irow += 1
        icol = 0

        # Plotting assembled powder with resolutions
        ax3 = plt.subplot2grid((nrow, ncol), (irow, icol), rowspan=2, colspan=2)
        self.plot_powder_and_resolution(powder, detector, distance, ax=ax3)
        icol = +2

        # Plotting histogram of pixel intensities
        ax4 = plt.subplot2grid((nrow, ncol), (irow, icol), rowspan=2)
        self.plot_intensity_hist(powder, Imin, ax4)
        irow += 2
        icol = 0

        # Plotting BO convergence
        ax5 = plt.subplot2grid((nrow, ncol), (irow, icol))
        self.plot_bo_history(ax5)
        icol += 1

        # Plotting score scan over distance
        ax6 = plt.subplot2grid((nrow, ncol), (irow, icol))
        self.plot_score_distance_scan(ax6)
        icol += 1

        # Plotting residual scan over distance
        ax7 = plt.subplot2grid((nrow, ncol), (irow, icol))
        self.plot_residual_distance_scan(distance, ax7)

        fig.tight_layout()

        if plot != "":
            fig.savefig(plot, dpi=100)
        return fig