"""
Classes for geometry optimization tasks.

Classes:
    BayesGeomOpt: optimize detector geometry using PyFAI coupled with Bayesian Optimization

"""

__all__ = ["BayesGeomOpt"]
__author__ = "Louis Conreux"

from lute.execution.logging import get_logger

import psana  # type: ignore
import logging
import numpy as np
import matplotlib.pyplot as plt  # type: ignore
import matplotlib.patches as patches  # type: ignore
import pyFAI  # type: ignore
from pyFAI.geometry import Geometry  # type: ignore
from pyFAI.goniometer import SingleGeometry  # type: ignore
from pyFAI.calibrant import CALIBRANT_FACTORY  # type: ignore
from pyFAI.units import RADIAL_UNITS  # type: ignore
from sklearn.gaussian_process import GaussianProcessRegressor  # type: ignore
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel, Matern  # type: ignore
from sklearn.utils._testing import ignore_warnings  # type: ignore
from sklearn.exceptions import ConvergenceWarning  # type: ignore
from scipy.stats import norm  # type: ignore
from mpi4py import MPI

pyFAI.use_opencl = False

logger: logging.Logger = get_logger(__name__)


class BayesGeomOpt:
    """
    Class to perform Geometry Optimization using Bayesian Optimization on pyFAI

    Parameters
    ----------
    exp : str
        Experiment name
    run : int
        Run number
    det_type : str
        Detector type
    detector : PyFAI(Detector)
        PyFAI detector object
    calibrant : str
        Calibrant name
    """

    def __init__(
        self,
        exp,
        run,
        det_type,
        detector,
        calibrant,
    ):
        self.exp = exp
        self.run = run
        self.det_type = det_type
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.detector = detector
        self.calibrant = calibrant
        self.order = ["dist", "poni1", "poni2", "rot1", "rot2", "rot3"]
        self.space = ["poni1", "poni2"]
        self.values = {
            "dist": 0.1,
            "poni1": 0,
            "poni2": 0,
            "rot1": 0,
            "rot2": 0,
            "rot3": 0,
        }

    @staticmethod
    def expected_improvement(X, gp_model, best_y, epsilon=0):
        y_pred, y_std = gp_model.predict(X, return_std=True)
        z = (y_pred - best_y + epsilon) / y_std
        ei = y_pred - best_y * norm.cdf(z) + y_std * norm.pdf(z)
        return ei

    @staticmethod
    def upper_confidence_bound(X, gp_model, best_y=None, beta=1.96):
        y_pred, y_std = gp_model.predict(X, return_std=True)
        ucb = y_pred + beta * y_std
        return ucb

    @staticmethod
    def probability_of_improvement(X, gp_model, best_y, epsilon=0):
        y_pred, y_std = gp_model.predict(X, return_std=True)
        z = (y_pred - best_y + epsilon) / y_std
        pi = norm.cdf(z)
        return pi

    @staticmethod
    def contextual_improvement(X, gp_model, best_y, hyperparam=None):
        y_pred, y_std = gp_model.predict(X, return_std=True)
        cv = np.mean(y_std**2) / best_y
        z = (y_pred - best_y + cv) / y_std
        ci = y_pred - best_y * norm.cdf(z) + y_std * norm.pdf(z)
        return ci

    def set_wavelength_calibrant(self):
        """
        Define calibrant for optimization with appropriate wavelength

        Parameters
        ----------
        wavelength : float
            Wavelength of the experiment
        """
        self.calibrant_name = self.calibrant
        calibrant = CALIBRANT_FACTORY(self.calibrant)
        ds_args = f"exp={self.exp}:run={self.run}:idx"
        ds = psana.DataSource(ds_args)
        runner = next(ds.runs())
        evt = runner.event(runner.times()[0])
        photon_energy = None
        try:
            photon_energy = psana.Detector("EBeam").get(evt).ebeamPhotonEnergy()
        except AttributeError:
            logger.warning("Event lacking an ebeamPhotonEnergy value.")
        if photon_energy is None or np.isinf(photon_energy):
            self.wavelength = ds.env().epicsStore().value("SIOC:SYS0:ML00:AO192") * 1e-9
        else:
            self.wavelength = 1.23984197386209e-06 / photon_energy
        self.photon_energy = photon_energy
        calibrant.wavelength = self.wavelength
        self.calibrant = calibrant

    def build_mask(self, central=True, edges=True):
        """
        Mask pixels marked as false status, edges and central pixels of panels

        Parameters
        ----------
        central : bool
            Mask central pixel of panels
        edges : bool
            Mask edges of panels
        """
        ds_args = f"exp={self.exp}:run={self.run}:idx"
        ds = psana.DataSource(ds_args)
        det = psana.Detector(self.det_type, ds.env())
        runner = next(ds.runs())
        evt = runner.event(runner.times()[0])
        runnum = evt.run()
        try:
            mask = det.mask_v2(par=runnum, central=central, edges=edges)
        except AttributeError:
            mask = None
        if mask is not None:
            if len(mask.shape) != 2:
                mask = np.reshape(mask, (mask.shape[0] * mask.shape[1], mask.shape[2]))
        return mask

    def min_intensity(self, powder):
        """
        Define minimal intensity for control point extraction

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
        if self.rank == 0:
            logger.info(f"Threshold for pixel outliers: {threshold}")
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
        self.q = round(q, 2)
        self.Imin = Imin
        self.powder = powder
        return Imin

    @ignore_warnings(category=ConvergenceWarning)
    def bayes_opt_center(
        self,
        powder,
        dist,
        bounds,
        res,
        Imin,
        max_rings=6,
        n_samples=20,
        n_iterations=80,
        kernel="RBF",
        af="ucb",
        hyperparam=None,
        prior=True,
        seed=None,
    ):
        """
        Perform Bayesian Optimization on PONI center parameters, for a fixed distance

        Parameters
        ----------
        powder : np.ndarray
            Powder image
        dist : float
            Fixed distance
        bounds : dict
            Dictionary of bounds for each parameter
        res : float
            Resolution of the grid used to discretize the parameter search space
        Imin : float
            Minimum intensity threshold for control point extraction based on intensity distribution percentile
        max_rings : int
            Maximum number of rings to use for control point extraction
        n_samples : int
            Number of samples to initialize the GP model
        n_iterations : int
            Number of iterations for optimization
        kernel : str
            Kernel to use for the Gaussian Process Regressor
            'RBF' for Radial Basis Function kernel
            'Matern' for Matern kernel
        af : str
            Acquisition function to use for optimization
        hyperparam : dict
            Dictionary of hyperparameters for the acquisition function
        prior : bool
            Use prior information for optimization
        seed : int
            Random seed for reproducibility
        """

        if seed is not None:
            np.random.seed(seed)
        self.values["dist"] = dist
        if res is None:
            res = self.detector.pixel_size

        if bounds["poni1"][0] > bounds["poni1"][1]:
            bounds["poni1"] = (bounds["poni1"][1], bounds["poni1"][0])
        if bounds["poni2"][0] > bounds["poni2"][1]:
            bounds["poni2"] = (bounds["poni2"][1], bounds["poni2"][0])

        inputs = {}
        norm_inputs = {}
        for p in self.order:
            if p in self.space:
                inputs[p] = np.arange(bounds[p][0], bounds[p][1] + res, res)
                norm_inputs[p] = inputs[p]
            else:
                inputs[p] = np.array([self.values[p]])
        X = np.array(np.meshgrid(*[inputs[p] for p in self.order])).T.reshape(
            -1, len(self.order)
        )
        X_space = np.array(
            np.meshgrid(*[norm_inputs[p] for p in self.space])
        ).T.reshape(-1, len(self.space))
        X_norm = (X_space - np.mean(X_space, axis=0)) / (
            np.max(X_space, axis=0) - np.min(X_space, axis=0)
        )
        if prior:
            means = np.mean(X_space, axis=0)
            cov = np.diag(
                [
                    ((bounds[param][1] - bounds[param][0]) / 5) ** 2
                    for param in self.space
                ]
            )
            X_samples = np.random.multivariate_normal(means, cov, n_samples)
            X_norm_samples = (X_samples - np.mean(X_space, axis=0)) / (
                np.max(X_space, axis=0) - np.min(X_space, axis=0)
            )
            for p in self.order:
                if p not in self.space:
                    idx = self.order.index(p)
                    X_samples = np.insert(X_samples, idx, self.values[p], axis=1)
        else:
            idx_samples = np.random.choice(X.shape[0], n_samples)
            X_samples = X[idx_samples]
            X_norm_samples = X_norm[idx_samples]

        bo_history = {}
        y = np.zeros((n_samples))
        for i in range(n_samples):
            dist, poni1, poni2, rot1, rot2, rot3 = X_samples[i]
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
                "extract_cp",
                powder,
                calibrant=self.calibrant,
                detector=self.detector,
                geometry=geom_initial,
            )
            sg.extract_cp(max_rings=max_rings, pts_per_deg=1, Imin=Imin)
            y[i] = len(sg.geometry_refinement.data)
            bo_history[f"init_sample_{i+1}"] = {"param": X_samples[i], "score": y[i]}

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
        best_score = np.max(y_norm)

        if kernel == "RBF":
            kernel = RBF(
                length_scale=0.3, length_scale_bounds=(0.2, 0.4)
            ) * ConstantKernel(
                constant_value=1.0, constant_value_bounds=(0.5, 1.5)
            ) + WhiteKernel(
                noise_level=0.001, noise_level_bounds="fixed"
            )
        elif kernel == "Matern":
            kernel = Matern(
                length_scale=0.3, length_scale_bounds=(0.2, 0.4), nu=2.5
            ) * ConstantKernel(
                constant_value=1.0, constant_value_bounds=(0.5, 1.5)
            ) + WhiteKernel(
                noise_level=0.001, noise_level_bounds="fixed"
            )
        gp_model = GaussianProcessRegressor(
            kernel=kernel, n_restarts_optimizer=10, random_state=0
        )
        gp_model.fit(X_norm_samples, y_norm)
        visited_idx = list([])

        if af == "ucb":
            if hyperparam is None:
                hyperparam = {"beta": 1.96}
            hyperparam = hyperparam["beta"]
            af = self.upper_confidence_bound
        elif af == "ei":
            if hyperparam is None:
                hyperparam = {"epsilon": 0}
            hyperparam = hyperparam["epsilon"]
            af = self.expected_improvement
        elif af == "pi":
            if hyperparam is None:
                hyperparam = {"epsilon": 0}
            hyperparam = hyperparam["epsilon"]
            af = self.probability_of_improvement
        elif af == "ci":
            af = self.contextual_improvement

        for i in range(n_iterations):
            # 1. Generate the Acquisition Function values using the Gaussian Process Regressor
            af_values = af(X_norm, gp_model, best_score, hyperparam)
            af_values[visited_idx] = -np.inf

            # 2. Select the next set of parameters based on the Acquisition Function
            new_idx = np.argmax(af_values)
            new_input = X[new_idx]
            visited_idx.append(new_idx)

            # 3. Compute the score of the new set of parameters
            dist, poni1, poni2, rot1, rot2, rot3 = new_input
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
                "extract_cp",
                powder,
                calibrant=self.calibrant,
                detector=self.detector,
                geometry=geom_initial,
            )
            sg.extract_cp(max_rings=max_rings, pts_per_deg=1, Imin=Imin)
            score = len(sg.geometry_refinement.data)
            if np.isnan(score):
                score = 0
            y = np.append(y, [score], axis=0)
            bo_history[f"iteration_{i+1}"] = {
                "param": X[new_idx],
                "score": score,
            }
            X_samples = np.append(X_samples, [X[new_idx]], axis=0)
            X_norm_samples = np.append(X_norm_samples, [X_norm[new_idx]], axis=0)
            if np.std(y) != 0:
                y_norm = (y - np.mean(y)) / np.std(y)
            else:
                y_norm = y - np.mean(y)
            best_score = np.max(y_norm)

            # 4. Update the Gaussian Process Regressor
            gp_model.fit(X_norm_samples, y_norm)

        best_idx = np.argmax(y_norm)
        best_param = X_samples[best_idx]
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
            "extract_cp",
            powder,
            calibrant=self.calibrant,
            detector=self.detector,
            geometry=geom_initial,
        )
        sg.extract_cp(max_rings=max_rings, pts_per_deg=1, Imin=Imin)
        self.sg = sg
        score = len(sg.geometry_refinement.data)
        residual = 0
        if score != 0:
            residual = sg.geometry_refinement.refine3(fix=["wavelength"])
        params = sg.geometry_refinement.param
        result = {
            "bo_history": bo_history,
            "params": params,
            "residual": residual,
            "score": score,
            "best_idx": best_idx,
        }
        return result

    def bayes_opt_geom(
        self,
        powder,
        bounds,
        res,
        max_rings=6,
        n_samples=20,
        n_iterations=80,
        kernel="RBF",
        af="ucb",
        hyperparam=None,
        prior=True,
        seed=None,
    ):
        """
        From guessed initial geometry, optimize the geometry using Bayesian Optimization on pyFAI package

        Parameters
        ----------
        powder : str
            Path to powder image to use for calibration
        bounds : dict
            Dictionary of bounds and resolution for search parameters
        res : float
            Resolution of the grid used to discretize the parameter search space
        max_rings : int
            Maximum number of rings to use for control point extraction
        n_samples : int
            Number of samples to initialize the GP model
        n_iterations : int
            Number of iterations for optimization
        kernel : str
            Kernel to use for the Gaussian Process Regressor
        af : str
            Acquisition function to use for optimization
        hyperparam : dict
            Dictionary of hyperparameters for the acquisition function
        prior : bool
            Use prior information for optimization
        seed : int
            Random seed for reproducibility
        """

        if seed is not None:
            np.random.seed(seed)

        self.set_wavelength_calibrant()

        mask = self.build_mask()
        if mask is not None:
            powder = powder * mask

        self.max_rings = max_rings
        Imin = self.min_intensity(powder)

        self.bounds = bounds
        if self.rank == 0:
            logger.info(
                f"Optimizing geometry for exp {self.exp} run {self.run} with {self.det_type} detector with minimal intensity threshold {Imin:.2e}"
            )
            if isinstance(bounds["dist"], float):
                distances = np.linspace(
                    bounds["dist"] - 0.05, bounds["dist"] + 0.05, self.size
                )
            else:
                distances = np.linspace(bounds["dist"][0], bounds["dist"][1], self.size)
            self.distances = distances
        else:
            distances = None

        dist = self.comm.scatter(distances, root=0)

        results = self.bayes_opt_center(
            powder,
            dist,
            bounds,
            res,
            Imin,
            max_rings,
            n_samples,
            n_iterations,
            kernel,
            af,
            hyperparam,
            prior,
            seed,
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
            if isinstance(self.bounds["dist"], float):
                thrsh = np.percentile(self.scan["score"], 25)
            else:
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
            self.score = self.scan["score"][index]
            self.best_idx = self.scan["best_idx"][index]

    def get_radius_map(self, detector, center=None):
        """
        Compute each pixel's radius for an array with input shape and center.

        Parameters
        ----------
        detector  : pyFAI Detector object
            detector object containing pixel infos
        center : 2d tuple or array
            (cx,cy) detector center in meters; if None, choose image center

        Returns
        -------
        r : numpy.ndarray, with input shape
            map of pixels' radii
        """
        y, x, z = detector.calc_cartesian_positions()
        if center is None:
            center = (0, 0)
        r = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
        return r

    def radial_profile(self, powder, detector, center=None):
        """
        Compute the radial intensity profile of an image.

        Parameters
        ----------
        powder : numpy.ndarray, shape (n,m)
            detector image
        center : 2d tuple or array
            (cx,cy) beam center in meter; if None, choose detector origin

        Returns
        -------
        radialprofile : numpy.ndarray, 1d
            radial intensity profile of input image
        """
        if center is None:
            center = (0, 0)
        r = self.get_radius_map(detector, center=center)
        intensity, bin_edges = np.histogram(
            r.ravel(), bins=1000, range=(0, r.max()), weights=powder.ravel()
        )
        count, _ = np.histogram(r.ravel(), bins=bin_edges)
        radialprofile = np.divide(
            intensity, count, out=np.zeros_like(intensity), where=count != 0
        )
        r_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        return radialprofile, r_centers

    def pix2q(self, pixels, distance):
        """
        Convert distance from number of pixels from detector center to q-space.

        Parameters
        ----------
        pixels : numpy.ndarray, 1d
            distance in meter from beam center
        distance : float
            detector distance in meter
        Returns
        -------
        qvals : numpy.ndarray, 1d
            magnitude of q-vector in per Angstrom
        """
        theta = np.arctan2(pixels, distance)
        return 4.0 * np.pi * np.sin(theta / 2.0) / (self.wavelength * 1e10)

    def plot_powder_and_resolution(self, sg, distance, ax=None):
        """
        Display an image with the control points and the calibrated rings as well as detector resolutions

        Parameters
        ----------
        sg : SingleGeometry
            SingleGeometry object containing powder and geometry data
        distance : float
            Distance of the detector
        beam_center : Tuple(float)
            Beam center coordinates
        """
        if ax is None:
            _fig, ax = plt.subplots()
        powder = sg.image
        label = sg.label
        detector = sg.detector
        y, x, z = detector.calc_cartesian_positions()
        if z is None:
            z = np.zeros_like(x)
        z += distance

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

        img = ax.scatter(
            x.ravel(),
            y.ravel(),
            c=powder.ravel(),
            s=1,
            edgecolors=None,
            linewidth=0,
            vmin=np.percentile(powder, 5),
            vmax=np.percentile(powder, 95),
        )
        cbar = plt.colorbar(img, ax=ax, orientation="vertical")
        cbar.set_label("Intensity", fontsize=8)
        cbar.ax.tick_params(labelsize=6)
        tth = self.calibrant.get_2th()
        if self.det_type.lower() != "rayonix":
            x = np.reshape(x, detector.raw_shape)
            y = np.reshape(y, detector.raw_shape)
            z = np.reshape(z, detector.raw_shape)
            ttha = np.arctan2(np.sqrt(x * x + y * y), z)
            for i in range(detector.n_modules):
                ax.contour(
                    x[i],
                    y[i],
                    ttha[i],
                    levels=tth,
                    cmap="autumn",
                    linewidths=1,
                    linestyles="dashed",
                )
        else:
            ttha = np.arctan2(np.sqrt(x * x + y * y), z)
            ax.contour(
                x,
                y,
                ttha,
                levels=tth,
                cmap="autumn",
                linewidths=1,
                linestyles="dashed",
            )

        cx, cy = 0, 0
        sign_x = np.sign(np.mean(x))
        sign_y = np.sign(np.mean(y))
        d = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        closest_pixel_index = np.argmin(d)
        closest_pixel = d.flatten()[closest_pixel_index]
        if self.det_type == "Rayonix":
            closest_pixel = 0.009  # Beam Stop Radius Rayonix = 9 mm
        closest_q = self.pix2q(closest_pixel, distance)
        closest_resol = 2 * np.pi / closest_q

        furthest_pixel_index = np.argmax(d)
        furthest_pixel = d.flatten()[furthest_pixel_index]
        furthest_q = self.pix2q(furthest_pixel, distance)
        furthest_resol = 2 * np.pi / furthest_q

        d_left = abs(cx - xmin)
        d_right = abs(cx - xmax)
        d_bottom = abs(cy - ymin)
        d_top = abs(cy - ymax)
        border_distances = [d_left, d_right, d_bottom, d_top]
        border_pixel = max(border_distances)
        border_q = self.pix2q(border_pixel, distance)
        border_resol = 2 * np.pi / border_q
        border_2_q = self.pix2q(border_pixel / 2, distance)
        border_2_resol = 2 * np.pi / border_2_q

        circle_closest = plt.Circle(
            (cx, cy), closest_pixel, color="green", linestyle="dashed", fill=False
        )
        ax.add_artist(circle_closest)
        ax.text(
            cx + sign_x * closest_pixel / np.sqrt(2),
            -cy + sign_y * closest_pixel / np.sqrt(2),
            f"{closest_resol:.3f} \u00c5",
            color="red",
            fontsize=8,
            ha="left",
        )

        circle_furthest = plt.Circle(
            (cx, cy), furthest_pixel, color="green", linestyle="dashed", fill=False
        )
        ax.add_artist(circle_furthest)
        ax.text(
            cx + sign_x * furthest_pixel / np.sqrt(2),
            cy + sign_y * furthest_pixel / np.sqrt(2),
            f"{furthest_resol:.3f} \u00c5",
            color="red",
            fontsize=8,
            ha="left",
        )

        circle_border = plt.Circle(
            (cx, cy), border_pixel, color="green", linestyle="dashed", fill=False
        )
        ax.add_artist(circle_border)
        ax.text(
            cx + sign_x * border_pixel / np.sqrt(2),
            cy + sign_y * border_pixel / np.sqrt(2),
            f"{border_resol:.3f} \u00c5",
            color="red",
            fontsize=8,
            ha="left",
        )

        circle_border_2 = plt.Circle(
            (cx, cy), border_pixel / 2, color="green", linestyle="dashed", fill=False
        )
        ax.add_artist(circle_border_2)
        ax.text(
            cx + sign_x * border_pixel / (2 * np.sqrt(2)),
            cy + sign_y * border_pixel / (2 * np.sqrt(2)),
            f"{border_2_resol:.3f} \u00c5",
            color="red",
            fontsize=8,
            ha="left",
        )

        ax.set_xlabel("X-axis (m)", fontsize=8)
        ax.set_ylabel("Y-axis (m)", fontsize=8)
        ax.tick_params(axis="x", labelsize=6)
        ax.tick_params(axis="y", labelsize=6)
        ax.set_title(label, fontsize=8)
        ax.set_aspect("equal")
        return (
            closest_q,
            closest_resol,
            furthest_q,
            furthest_resol,
            border_q,
            border_resol,
        )

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

    def plot_score_distance_scan(self, distances, ax):
        """
        Plot the score scan over distance

        Parameters
        ----------
        distances : np.array
            Array of distances
        ax : plt.Axes
            Matplotlib axes
        """
        scores = self.scan["score"]
        ax.plot(distances, scores)
        ax.axhline(
            self.thrsh,
            color="red",
            linestyle="--",
            label=f"Threshold score: {self.thrsh}",
        )
        ax.legend(fontsize=8)
        ax.set_xlabel("Distance (m)", fontsize=8)
        ax.set_ylabel("Score", fontsize=8)
        ax.tick_params(axis="x", labelsize=6)
        ax.tick_params(axis="y", labelsize=6)
        ax.set_title("Number of Control Points vs Distance", fontsize=8)

    def plot_residual_distance_scan(self, distances, refined_dist, ax):
        """
        Plot the residual scan over distance

        Parameters
        ----------
        distances : np.array
            Array of distances
        refined_dist : float
            Refined distance
        ax : plt.Axes
            Matplotlib axes
        """
        residuals = self.scan["residual"]
        ax.plot(distances, residuals)
        best_dist = distances[self.index]
        ax.axvline(
            best_dist,
            color="green",
            linestyle="--",
            label=f"Best distance (m): {best_dist:.3f}",
        )
        ax.axvline(
            refined_dist,
            color="red",
            linestyle="--",
            label=f"Refined distance (m): {refined_dist:.3f}",
        )
        ax.legend(fontsize=4)
        ax.set_yscale("log")
        ax.set_xlabel("Distance (m)", fontsize=8)
        ax.set_ylabel("Residual", fontsize=8)
        ax.tick_params(axis="x", labelsize=6)
        ax.tick_params(axis="y", labelsize=6)
        ax.set_title("Residual vs Distance", fontsize=8)

    def plot_hist_and_compute_stats(self, powder, exp, run, ax):
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
            self.Imin,
            color="purple",
            linestyle=":",
            linewidth=1.5,
            label=f"{self.q} th Percentile ({self.Imin:.2f})",
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

    def visualize_results(
        self,
        powder,
        bo_history,
        detector,
        distance,
        plot="",
    ):
        """
        Visualize fit, plotting (1) the BO convergence, (2) the radial profile and (3) the powder image.

        Parameters
        ----------
        powder : np.ndarray
            Powder image
        bo_history : dict
            Dictionary containing the history of optimization
        detector : PyFAI(Detector)
            Corrected PyFAI detector object
        distance : float
            Refined distance
        plot : str
            Path to save plot
        """
        fig = plt.figure(figsize=(9, 12), dpi=300)
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
            0.05, 0.7, f"Detector {self.det_type}", ha="left", va="center", fontsize=8
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
        ax1.axis("off")
        icol += 1

        # Plotting radial profiles with peaks
        ax2 = plt.subplot2grid((nrow, ncol), (irow, icol), colspan=ncol - icol)
        masked_powder = powder
        if self.det_type.lower() == "rayonix":
            radius = np.sqrt(2) * powder.shape[0] / 4
            row, col = np.ogrid[: powder.shape[0], : powder.shape[1]]
            center = (powder.shape[0] / 2, powder.shape[1] / 2)
            mask = ((row - center[0]) ** 2 + (col - center[1]) ** 2) <= radius**2
            masked_powder = powder * mask
        profile, radii = self.radial_profile(masked_powder, detector)
        q = self.pix2q(radii, distance)
        self.plot_radial_integration(
            q, profile, error=None, calibrant=self.calibrant, ax=ax2
        )
        irow += 1
        icol = 0

        # Plotting assembled powder with resolutions
        ax3 = plt.subplot2grid((nrow, ncol), (irow, icol), rowspan=2, colspan=2)
        geometry = Geometry(dist=distance)
        sg = SingleGeometry(
            f"Run {self.run} {self.calibrant_name}",
            powder,
            calibrant=self.calibrant,
            detector=detector,
            geometry=geometry,
        )
        sg.extract_cp(max_rings=self.max_rings, pts_per_deg=1, Imin=self.Imin)
        low_q, low_res, high_q, high_res, border_q, border_res = (
            self.plot_powder_and_resolution(sg=sg, distance=distance, ax=ax3)
        )
        icol = +2

        # Plotting histogram of pixel intensities
        ax4 = plt.subplot2grid((nrow, ncol), (irow, icol), rowspan=2)
        self.plot_hist_and_compute_stats(powder, self.exp, self.run, ax4)
        irow += 2
        icol = 0

        # Plotting BO convergence
        ax5 = plt.subplot2grid((nrow, ncol), (irow, icol))
        scores = [bo_history[key]["score"] for key in bo_history.keys()]
        ax5.plot(scores)
        ax5.set_xticks(np.arange(len(scores), step=20))
        ax5.axvline(
            self.scan["best_idx"][self.index],
            color="green",
            linestyle="--",
            label=f"Best score at n={self.scan['best_idx'][self.index]}",
        )
        ax5.set_xlabel("Iteration", fontsize=8)
        ax5.set_ylabel("Number of Control Points", fontsize=8)
        ax5.legend(fontsize=8)
        ax5.tick_params(axis="x", labelsize=6)
        ax5.tick_params(axis="y", labelsize=6)
        ax5.set_title(
            f"Convergence Plot, best score: {self.scan['score'][self.index]}",
            fontsize=8,
        )
        icol += 1

        # Plotting score scan over distance
        ax6 = plt.subplot2grid((nrow, ncol), (irow, icol))
        self.plot_score_distance_scan(self.distances, ax6)
        icol += 1

        # Plotting residual scan over distance
        ax7 = plt.subplot2grid((nrow, ncol), (irow, icol), colspan=ncol - icol)
        self.plot_residual_distance_scan(self.distances, distance, ax7)

        fig.tight_layout()

        if plot != "":
            fig.savefig(plot, dpi=300)
        return fig, low_q, low_res, high_q, high_res, border_q, border_res
