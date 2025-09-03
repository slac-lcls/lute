"""
Classes for geometry optimization tasks.

Classes:
    BayesGeomOpt: optimize detector geometry using PyFAI coupled with Bayesian Optimization

"""

__all__ = ["BayesGeomOpt"]
__author__ = "Louis Conreux"

from lute.execution.logging import get_logger

import os
import psana  # type: ignore
import logging
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import matplotlib.patches as patches 
from bokeh.plotting import figure  # type: ignore
from bokeh.models import ColorBar, LinearColorMapper, HoverTool, ColumnDataSource  # type: ignore
from bokeh.palettes import Viridis256  # type: ignore
from bokeh.models.annotations import Label  # type: ignore
import h5py
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
import pyFAI  
from pyFAI.geometry import Geometry
from pyFAI.goniometer import SingleGeometry
from pyFAI.calibrant import CALIBRANT_FACTORY  # type: ignore
from pyFAI.units import RADIAL_UNITS  # type: ignore
from sklearn.gaussian_process import GaussianProcessRegressor 
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel  
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from mpi4py import MPI

from LCLSGeom.psana.converter import PsanaToPyFAI, PsanaToCrystFEL, PyFAIToPsana

pyFAI.use_opencl = False

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
            logger.warning(f"Cannot find {detname} Max powder in {powder_path}, defaulting to {detname} Sum instead.")
            powder = h5[f"Sums/{detname}_calib"][()]
    return powder

def preprocess_powder(powder, smooth=False):
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
        calib = gaussian_filter(powder, sigma=1)
        gradx_calib = np.zeros_like(powder)
        grady_calib = np.zeros_like(powder)
        for p in range(powder.shape[0]):
            gradx_calib[p, :-1, :-1] = (
                calib[p, 1:, :-1] - calib[p, :-1, :-1] + calib[p, 1:, 1:] - calib[p, :-1, 1:]
            ) / 2
            grady_calib[p, :-1, :-1] = (
                calib[p, :-1, 1:] - calib[p, :-1, :-1] + calib[p, 1:, 1:] - calib[p, 1:, :-1]
            ) / 2
        powder = np.sqrt(gradx_calib**2 + grady_calib**2)
    return powder

def min_intensity(powder, threshold):
    """
    Estimates minimal intensity for extracting key Bragg peaks

    Parameters
    ----------
    powder : np.ndarray
        Powder image
    threshold : float
        Percentile for intensity thresholding
    """
    mean = np.mean(powder)
    std = np.std(powder)
    outlier = mean + 5 * std
    nice_pix = powder < outlier
    Imin = np.percentile(powder[nice_pix], threshold)
    powder = np.clip(powder, 0, outlier)
    return Imin

def generate_powder(powder_path, detname, smooth=False):
    """
    Generate the assembled powder plot and cache it.

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
    Imin = min_intensity(powder, 95)
    return powder, Imin

def build_detector(in_file, shape):
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
        in_file=in_file,
        shape=shape,
    )
    detector = psana_to_pyfai.detector
    return detector

def update_geometry(optimizer, out_file):
    """
    Update the geometry and write a new .poni, .geom and .data file

    Parameters
    ----------
    optimizer : BayesGeomOpt
        Optimizer object
    out_file : str
        Path to the output file
    """
    path = os.path.dirname(out_file)
    poni_file = os.path.join(
        path, f"r{optimizer.run:0>4}.poni"
    )
    optimizer.gr.save(poni_file)
    PyFAIToPsana(
        in_file=poni_file,
        detector=optimizer.detector,
        out_file=out_file,
    )
    geom_file = os.path.join(
        path, f"r{optimizer.run:0>4}.geom"
    )
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

def define_calibrant(calibrant, exp, run):
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
    ds_args = f"exp={exp}:run={run}:idx"
    ds = psana.DataSource(ds_args)
    runner = next(ds.runs())
    evt = runner.event(runner.times()[0])
    photon_energy = None
    try:
        photon_energy = psana.Detector("EBeam").get(evt).ebeamPhotonEnergy()
        wavelength = 1.23984197386209e-06 / photon_energy
    except Exception:
        wavelength = ds.env().epicsStore().value("SIOC:SYS0:ML00:AO192") * 1e-9
        photon_energy = 1.23984197386209e-06 / wavelength
    calibrant = CALIBRANT_FACTORY(calibrant)
    calibrant.wavelength = wavelength
    return calibrant

def rotation_matrix(params):
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
    rot1 = np.array([[1.0, 0.0, 0.0],
                        [0.0, cos_rot1, sin_rot1],
                        [0.0, -sin_rot1, cos_rot1]])
    # Rotation about horizontal axis: Note this rotation is left-handed
    rot2 = np.array([[cos_rot2, 0.0, -sin_rot2],
                        [0.0, 1.0, 0.0],
                        [sin_rot2, 0.0, cos_rot2]])
    # Rotation about z-axis: Note this rotation is right-handed
    rot3 = np.array([[cos_rot3, -sin_rot3, 0.0],
                        [sin_rot3, cos_rot3, 0.0],
                        [0.0, 0.0, 1.0]])
    rotation_matrix = np.dot(np.dot(rot3, rot2), rot1)
    return rotation_matrix

def correct_geom(detector, params):
    """
    Correct the geometry based on the given parameters found by PyFAI calibration
    """
    p1, p2, p3 = detector.calc_cartesian_positions()
    dist = params[0]
    poni1 = params[1]
    poni2 = params[2]
    p1 = (p1 - (detector.pixel_size / 2) - poni1).ravel()
    p2 = (p2 - (detector.pixel_size / 2) - poni2).ravel()
    if p3 is None:
        p3 = np.zeros_like(p1) + dist
    else:
        p3 = (p3+dist).ravel()
    coord_det = np.vstack((p1, p2, p3))
    coord_sample = np.dot(rotation_matrix(params), coord_det)
    x, y, z = coord_sample
    x = np.reshape(x, detector.raw_shape)
    y = np.reshape(y, detector.raw_shape)
    z = np.reshape(z, detector.raw_shape)
    return x, y, z

def calculate_2theta(detector, params):
    """
    Calculate the 2theta angles for the detector based on the geometry parameters.

    Parameters
    ----------
    detector : pyFAI.detectors.Detector
        PyFAI detector object containing pixel index map and shape information.
    params : list
        6 Geometry parameters: distance, x-shift, y-shift, Rx, Ry, Rz
    """
    x, y, z = correct_geom(detector, params)
    ttha = np.zeros(detector.raw_shape)
    # loop through the panels
    for p in range(detector.n_modules):
        ttha[p, :] = np.arctan2(np.sqrt(x[p]*x[p]+y[p]*y[p]), z[p])
    return ttha

def get_radius_map(detector):
    """
    Compute each pixel's radius for an array with input shape and center.
    Detector is assumed to be calibrated.

    Parameters
    ----------
    detector  : pyFAI.Detector
        pyFAI detector object 

    Returns
    -------
    r : numpy.ndarray, with input shape
        map of pixels' radii
    """
    y, x, _ = detector.calc_cartesian_positions()
    r = np.sqrt(x ** 2 + y ** 2)
    return r

def radial_profile(powder, detector):
    """
    Compute the radial intensity profile of an image.
    Detector is assumed to be calibrated.

    Parameters
    ----------
    powder : numpy.ndarray, shape (n,m)
        detector image
    detector : pyFAI.Detector
        PyFAI detector object

    Returns
    -------
    radialprofile : numpy.ndarray, 1d
        radial intensity profile of input image
    """
    r = get_radius_map(detector)
    intensity, bin_edges = np.histogram(
        r.ravel(), bins=1000, range=(r.min(), r.max()), weights=powder.ravel()
    )
    count, _ = np.histogram(r.ravel(), bins=bin_edges)
    radialprofile = np.divide(
        intensity, count, out=np.zeros_like(intensity), where=count != 0
    )
    r_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return radialprofile, r_centers

def pix2q(pixels, distance, wavelength):
    """
    Convert distance from number of pixels from detector center to q-space.

    Parameters
    ----------
    pixels : numpy.ndarray, 1d
        distance in meter from beam center
    distance : float
        detector distance in meter
    wavelength : float
        X-ray wavelength in meter

    Returns
    -------
    qs: numpy.ndarray, 1d
        magnitude of q-vector in per Angstrom
    """
    theta = np.arctan2(pixels, distance)
    qs = 4.0 * np.pi * np.sin(theta / 2.0) / (wavelength * 1e10)
    return qs

class BayesGeomOpt:
    """
    Class to perform Geometry Optimization using Bayesian Optimization wrapped over PyFAI

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
        self.det_name = detector.name
        self.detector = detector
        self.powder = powder
        self.stacked_powder = np.reshape(powder, detector.shape)
        self.calibrant = calibrant
        self.calibrant_name = os.path.splitext(os.path.basename(calibrant.filename))[0]
        self.fixed = fixed
        self.order = ["dist", "poni1", "poni2", "rot1", "rot2", "rot3"]
        self.tth = np.array(calibrant.get_2th())
        self.space = []
        for p in self.order:
            if p not in self.fixed:
                self.space.append(p)
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

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

    def create_search_space(self, bounds, center, res):
        """
        Dynamically discretize the search space based on bounds.
        
        Parameters
        ----------
        bounds : dict
            Bounds for each parameter, format: {param: (lower, upper)}
        center : dict
            Center values for each parameter
        res : dict
            Resolution per parameter
        
        Returns
        -------
        X : np.ndarray
            Full 6D geometry space (cartesian product)
        X_norm : np.ndarray
            Normalized search space (between-1 and 1)
        """
        full_params = {}
        search_params = {}
        for p in self.order:
            if p in self.space:
                low = center[p] + bounds[p][0]
                high = center[p] + bounds[p][1]
                step = res[p]
                full_params[p] = np.arange(low, high + step, step)
                search_params[p] = full_params[p]
            else:
                full_params[p] = np.array([center[p]])

        X = np.array(np.meshgrid(*[full_params[p] for p in self.order])).T.reshape(-1, len(self.order))
        X_search = np.array(np.meshgrid(*[search_params[p] for p in self.space])).T.reshape(-1, len(self.space))
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
            cov = np.diag([((bounds[p][1] - bounds[p][0]) / 5) ** 2 for p in self.space])
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

    def score(self, sample, Imin, max_rings, rtol=1e-2):
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
        rtol : float
            Relative tolerance for masking ring pixels

        Returns
        -------
        score : float
            Scalar score for Bayesian optimization
        """
        ttha = calculate_2theta(self.detector, sample)
        min_ttha = np.min(ttha)
        max_ttha = np.max(ttha)
        valid_ttha = self.tth[(self.tth >= min_ttha) & (self.tth <= max_ttha)]

        score = 0.0
        ring = 0
        for tth_i in valid_ttha:
            if ring >= max_rings:
                return score / max_rings
            mask = np.abs(ttha - tth_i) <= rtol * tth_i
            pixels = self.powder[mask]
            pixels = pixels[pixels >= Imin]
            if len(pixels) != 0:
                score += np.sum(pixels)
            ring += 1
        score /= max_rings
        return score

    def pyFAI_score(self, best_param, Imin, max_rings, rtol):
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
        geom_initial = Geometry(
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
            geometry=geom_initial,
        )
        sg.extract_cp(max_rings=max_rings, pts_per_deg=1, Imin=Imin)
        self.sg = sg
        score = self.score(best_param, Imin, max_rings, rtol)
        residual = 0
        if len(sg.geometry_refinement.data) > 0:
            residual = sg.geometry_refinement.refine3(fix=["rot3", "wavelength"])
        params = sg.geometry_refinement.param
        return residual, score, params


    @ignore_warnings(category=ConvergenceWarning)
    def sync_bayes_opt(
        self,
        center,
        bounds,
        res,
        n_samples,
        n_iterations,
        Imin,
        max_rings,
        rtol,
        beta=1.96,
        prior=True,
        seed=0,
    ):
        """
        Perform Bayesian Optimization on 5 geometric parameters.

        Parameters
        ----------
        center : dict
            Dictionary of the center values for each parameter
        bounds : dict
            Dictionary of the per-parameter bounds for the search space
        res : dict
            Dictionary of the per-parameter resolutions for the search space
        n_samples : int
            Number of initial samples to draw
        n_iterations : int
            Number of optimization iterations
        Imin : float
            Minimum intensity threshold
        max_rings : int
            Maximum number of rings to consider
        rtol : float
            Relative tolerance in q-space for masking ring pixels
        beta : float
            Exploration-exploitation trade-off parameter for UCB
        prior : bool
            Use prior information for optimization
        seed : int
            Random seed for reproducibility
        """
        np.random.seed(seed+self.rank)

        # 1. Create Search Space
        X, X_norm = self.create_search_space(bounds, center, res)
        print(f"Rank {self.rank}: Search space size: {X.shape[0]}")

        # 2. Sample Initial Points
        # Rank 0 will sample from a Gaussian prior on center
        # Other ranks will sample uniformly within search space
        if self.rank == 0:
            prior = True
        else:
            prior = False
        X_samples, X_norm_samples = self.sample_initial_points(X, X_norm, center, bounds, n_samples, prior)

        bo_history = {}
        y = np.zeros((n_samples))

        # 3. Evaluate initial points
        for i in range(n_samples):
            y[i] = self.score(X_samples[i], Imin, max_rings, rtol)
            bo_history[f"init_{i+1}"] = {"param": X_samples[i], "score": y[i]}

        self.comm.Barrier()

        X_samples_all = self.comm.gather(X_samples, root=0)
        X_norm_samples_all = self.comm.gather(X_norm_samples, root=0)
        y_all = self.comm.gather(y, root=0)

        if self.rank == 0:
            X_samples_all = np.vstack(X_samples_all)
            X_norm_samples_all = np.vstack(X_norm_samples_all)
            y_all = np.concatenate(y_all)
            y_all[np.isnan(y_all)] = 0
            if np.std(y_all) != 0:
                y_norm = (y_all - np.mean(y_all)) / np.std(y_all)
            else:
                y_norm = y_all - np.mean(y_all)

            kernel = RBF(
                length_scale=0.3, length_scale_bounds=(0.2, 0.4)
            ) * ConstantKernel(
                constant_value=1.0, constant_value_bounds=(0.5, 1.5)
            ) + WhiteKernel(
                noise_level=0.001, noise_level_bounds="fixed"
            )
            gp_model = GaussianProcessRegressor(kernel=kernel, random_state=seed)
            gp_model.fit(X_norm_samples_all, y_norm)
            visited_idx = list([])

        for i in tqdm(range(n_iterations)):
            # 4. Rank 0 selects next points with q-UCB
            if self.rank == 0:
                nexts = self.q_UCB(X_norm, gp_model, self.size, visited_idx, beta)
                next_points = X[nexts]
                visited_idx.extend(nexts)
            else:
                next_points = None

            # 5. Scatter points to all ranks
            next_point = self.comm.scatter(next_points, root=0)

            # 6. Compute score locally
            score = self.score(next_point, Imin, max_rings, rtol)
            bo_history[f"iter_{i+1}"] = {"param": next_point, "score": score}

            self.comm.Barrier()

            # 7. Gather scores on Rank 0
            score_all = self.comm.gather(score, root=0)

            if self.rank == 0:
                scores = np.array(score_all)
                scores[np.isnan(scores)] = 0
                y_all = np.concatenate([y_all, scores])
                X_samples = np.vstack([X_samples, X[nexts]])
                X_norm_samples = np.vstack([X_norm_samples, X_norm[nexts]])
                if np.std(y_all) != 0:
                    y_norm = (y_all - np.mean(y_all)) / np.std(y_all)
                else:
                    y_norm = y_all - np.mean(y_all)

                # 8. Update Gaussian Process
                gp_model.fit(X_norm_samples, y_norm)

        self.comm.Barrier()

        # 8. Collect BO history from each rank
        bo_histories = self.comm.gather(bo_history, root=0)
        if self.rank == 0:
            for _ in range(self.size):
                history = [bo_histories[r] for r in range(self.size)]

        # 9. Evaluate best geometry using PyFAI refinement tool
        if self.rank == 0:
            best_idx = np.argmax(y_all)
            best_param = X_samples[best_idx]
            residual, score, params = self.pyFAI_score(best_param, Imin, max_rings, rtol)
            result = {
                "history": history,
                "params": params,
                "residual": residual,
                "score": score,
                "best_idx": best_idx,
            }
            return result
        
    @ignore_warnings(category=ConvergenceWarning)
    def async_bayes_opt(
        self,
        center,
        bounds,
        res,
        n_samples,
        n_iterations,
        Imin,
        max_rings,
        rtol,
        beta=1.96,
        prior=True,
        seed=0,
    ):
        """
        Perform Bayesian Optimization on 5 geometric parameters.

        Parameters
        ----------
        center : dict
            Dictionary of the center values for each parameter
        bounds : dict
            Dictionary of the per-parameter bounds for the search space
        res : dict
            Dictionary of the per-parameter resolutions for the search space
        n_samples : int
            Number of initial samples to draw
        n_iterations : int
            Number of optimization iterations
        Imin : float
            Minimum intensity threshold
        max_rings : int
            Maximum number of rings to consider
        rtol : float
            Relative tolerance in q-space for masking ring pixels
        beta : float
            Exploration-exploitation trade-off parameter for UCB
        prior : bool
            Use prior information for optimization
        seed : int
            Random seed for reproducibility
        """
        np.random.seed(seed+self.rank)

        # 1. Create Search Space
        X, X_norm = self.create_search_space(bounds, center, res)
        print(f"Rank {self.rank}: Search space size: {X.shape[0]}")

        # 2. Sample Initial Points
        # Rank 0 will sample from a Gaussian prior on center
        # Other ranks will sample uniformly within search space
        if self.rank == 0:
            prior = True
        else:
            prior = False
        X_samples, X_norm_samples = self.sample_initial_points(X, X_norm, center, bounds, n_samples, prior)

        bo_history = {}
        y = np.zeros((n_samples))

        # 3. Evaluate initial points
        for i in range(n_samples):
            y[i] = self.score(X_samples[i], Imin, max_rings, rtol)
            bo_history[f"init_{i+1}"] = {"param": X_samples[i], "score": y[i]}

        if np.std(y) != 0:
            y_norm = (y - np.mean(y)) / np.std(y)
        else:
            y_norm = y - np.mean(y)

        kernel = RBF(
            length_scale=0.3, length_scale_bounds=(0.2, 0.4)
        ) * ConstantKernel(
            constant_value=1.0, constant_value_bounds=(0.5, 1.5)
        ) + WhiteKernel(
            noise_level=0.001, noise_level_bounds="fixed"
        )
        gp_model = GaussianProcessRegressor(kernel=kernel, random_state=seed)
        gp_model.fit(X_norm_samples, y_norm)
        visited_idx = list([])

        for i in tqdm(range(n_iterations)):
            # 4. Rank 0 selects next points with q-UCB
            next = self.UCB(X_norm, gp_model, visited_idx, beta)
            next_point = X[next]
            visited_idx.append(next)

            # 5. Compute score
            score = self.score(next_point, Imin, max_rings, rtol)
            bo_history[f"iter_{i+1}"] = {"param": next_point, "score": score}

            y = np.concatenate([y, [score]])
            X_samples = np.vstack([X_samples, [X[next]]])
            X_norm_samples = np.vstack([X_norm_samples, [X_norm[next]]])
            if np.std(y) != 0:
                y_norm = (y - np.mean(y)) / np.std(y)
            else:
                y_norm = y - np.mean(y)

            # 6. Update Gaussian Process
            gp_model.fit(X_norm_samples, y_norm)

        self.comm.Barrier()

        # 7. Collect BO history from each rank
        bo_histories = self.comm.gather(bo_history, root=0)
        y_all = self.comm.gather(y, root=0)
        X_samples_all = self.comm.gather(X_samples, root=0)
        X_norm_samples_all = self.comm.gather(X_norm_samples, root=0)
        if self.rank == 0:
            for rank in range(self.size):
                history = [bo_histories[r] for r in range(self.size)]
            y = np.concatenate(y_all)
            X_samples = np.vstack(X_samples_all)
            X_norm_samples = np.vstack(X_norm_samples_all)

            # 8. Evaluate best geometry using PyFAI refinement tool
            best_idx = np.argmax(y)
            best_param = X_samples[best_idx]
            residual, score, params = self.pyFAI_score(best_param, Imin, max_rings, rtol)
            result = {
                "history": history,
                "params": params,
                "residual": residual,
                "score": score,
                "best_idx": best_idx,
            }
            return result

    def plot_bo_history(self, history, ax=None):
        """
        Plot the Bayesian Optimization history.

        Parameters
        ----------
        history : list
            List of dictionaries containing the optimization history.
        ax : plt.Axes, optional
            Matplotlib axes to plot on.
        """
        if ax is None:
            fig, ax = plt.subplots()

        for entry in history:
            ax.plot(range(len(entry["score"])), entry["score"], "o-", linewidth=0.8)

        ax.set_title("Bayesian Optimization History", fontsize=8)
        ax.set_xlabel("Iterations", fontsize=8)
        ax.set_ylabel("Score", fontsize=8)
        ax.set_yscale("log")
        ax.tick_params(axis="x", labelsize=6)
        ax.tick_params(axis="y", labelsize=6)

    def plot_radial_integration(
        self, q, profile, error, calibrant=None, label=None, ax=None
    ):
        """
        Plot the radial integration of a powder image

        Parameters
        ----------
        q : np.array
            Array of q values
        profile : np.array
            Array of intensity values
        error : np.array
            Array of intensity errors if provided
        calibrant : Calibrant
            Calibrant object
        label : str
            Name of the curve
        ax : plt.Axes
            Matplotlib axes
        """
        from matplotlib import lines

        if ax is None:
            fig, ax = plt.subplots()

        unit = RADIAL_UNITS["q_A^-1"]
        if error is not None:
            ax.errorbar(q, profile, error, label=label)
        else:
            ax.plot(q, profile, label=label, color="black", linewidth=0.8)

        if label:
            ax.legend(fontsize=8)
        if calibrant and unit:
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

        ax.set_title("Radial Profile", fontsize=8)
        if unit:
            ax.set_xlabel(unit.label, fontsize=8)
        ax.set_ylabel("Intensity", fontsize=8)
        ax.tick_params(axis="x", labelsize=6)
        ax.tick_params(axis="y", labelsize=6)

    def plot_hist_and_compute_stats(self, powder, exp, run, Imin, ax):
        """
        Plot histogram of pixel intensities and compute statistics

        Parameters
        ----------
        powder : np.ndarray
            Powder image
        exp : str
            Experiment name
        run : int
            Run number
        Imin : float
            Minimum intensity value
        ax : plt.Axes
            Matplotlib axes
        """
        threshold = np.mean(powder) + 5 * np.std(powder)
        nice_pix = powder < threshold
        mean = np.mean(powder[nice_pix])
        std_dev = np.std(powder[nice_pix])
        _ = ax.hist(
            powder[nice_pix],
            bins=500,
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
            linewidth=1.5,
            label=f"95th Percentile ({Imin:.2f})",
        )
        ax.set_ylim([0, mean + 5 * std_dev])
        ax.set_ylabel("Pixel Intensity", fontsize=8)
        ax.set_xlabel("Frequency", fontsize=8)
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.tick_params(axis="y", labelsize=6)
        ax.set_title(
            f"Histogram of Pixel Intensities \n for {exp} run {run}", fontsize=8
        )
        ax.legend(fontsize=8)

    def create_interactive_powder(
        self,
        powder,
        detector,
        distance,
    ):
        """
        Create an interactive powder image with control points and calibrated rings.

        Parameters
        ----------
        powder : np.ndarray
            Powder image
        detector : PyFAI(Detector)
            Corrected PyFAI detector object
        distance : float
            Refined distance
        """
        y, x, z = detector.calc_cartesian_positions()
        if z is None:
            z = np.zeros_like(x)
        z += distance

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
            title=f"Run {self.run} - {detector.detname} - {self.calibrant_name}",
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

        _ = p.circle(
            x="x",
            y="y",
            size=3,
            color={"field": "intensity", "transform": color_mapper},
            line_color=None,
            source=source,
        )

        color_bar = ColorBar(
            color_mapper=color_mapper, width=8, location=(0, 0), title="Intensity"
        )
        p.add_layout(color_bar, "right")

        tth = self.calibrant.get_2th()
        x = np.reshape(x, detector.raw_shape)
        y = np.reshape(y, detector.raw_shape)
        z = np.reshape(z, detector.raw_shape)

        for i in range(detector.n_modules):
            ttha = np.arctan2(np.sqrt(x[i] * x[i] + y[i] * y[i]), z[i])
            p.contour(
                x=x[i],
                y=y[i],
                z=ttha,
                levels=tth,
                line_color="red",
                line_width=3,
                line_dash="dashed",
            )

        cx, cy = 0, 0
        d = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        closest_pixel_index = np.argmin(d)
        closest_pixel = d.flatten()[closest_pixel_index]
        closest_q = pix2q(closest_pixel, distance, self.calibrant.wavelength)
        closest_resol = 2 * np.pi / closest_q

        furthest_pixel_index = np.argmax(d)
        furthest_pixel = d.flatten()[furthest_pixel_index]
        furthest_q = pix2q(furthest_pixel, distance, self.calibrant.wavelength)
        furthest_resol = 2 * np.pi / furthest_q

        d_left = abs(cx - xmin)
        d_right = abs(cx - xmax)
        d_bottom = abs(cy - ymin)
        d_top = abs(cy - ymax)
        border_distances = [d_left, d_right, d_bottom, d_top]
        border_pixel = max(border_distances)
        border_q = pix2q(border_pixel, distance, self.calibrant.wavelength)
        border_resol = 2 * np.pi / border_q
        border_2_q = pix2q(border_pixel / 2, distance, self.calibrant.wavelength)
        border_2_resol = 2 * np.pi / border_2_q

        circles_data = [
            (closest_pixel, closest_resol),
            (furthest_pixel, furthest_resol),
            (border_pixel, border_resol),
            (border_pixel / 2, border_2_resol),
        ]

        for radius, resol in circles_data:
            theta = np.linspace(0, 2 * np.pi, 100)
            circle_x = cx + radius * np.cos(theta)
            circle_y = cy + radius * np.sin(theta)
            p.line(
                circle_x, circle_y, line_color="green", line_dash="dashed", line_width=3
            )
            text_x = cx + radius / np.sqrt(2)
            text_y = cy + radius / np.sqrt(2)

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
        history,
        powder,
        detector,
        distance,
        Imin,
        low_resolution=None,
        high_resolution=None,
        border_resolution=None,
        plot="",
    ):
        """
        Create a diagnostics panel with the results of the Bayesian Optimization.

        Parameters
        ----------
        history : list
            List of BO history
        powder : np.ndarray
            Powder image
        detector : PyFAI(Detector)
            Corrected PyFAI detector object
        distance : float
            Refined distance
        Imin : float
            Minimum intensity value
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
            0.05, 0.7, f"Detector {detector.detname}", ha="left", va="center", fontsize=8
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
        ax2 = plt.subplot2grid((nrow, ncol), (irow, icol), rowspan=2)
        self.plot_hist_and_compute_stats(powder, self.exp, self.run, Imin, ax2)
        icol = 0
        irow += 1

        # Plotting BO iterations
        ax3 = plt.subplot2grid((nrow, ncol), (irow, icol))
        self.plot_bo_history(history, ax3)

        # Plotting radial profiles with peaks
        ax4 = plt.subplot2grid((nrow, ncol), (irow, icol), colspan=2)
        profile, radii = radial_profile(powder, detector)
        qs = pix2q(radii, distance, self.calibrant.wavelength)
        self.plot_radial_integration(
            qs, profile, error=None, calibrant=self.calibrant, ax=ax4
        )

        fig.tight_layout()

        if plot != "":
            fig.savefig(plot, dpi=100)
        return fig
