"""
Classes for geometry optimization tasks.

Classes:
    OptimizePyFAIGeom: optimize detector geometry using PyFAI coupled with Bayesian Optimization 

"""

__all__ = ["OptimizePyFAIGeometry"]
__author__ = "Louis Conreux"

from lute.io.models.geom_opt import OptimizePyFAIGeometryParameters
from lute.tasks.task import Task
from lute.tasks.dataclasses import TaskStatus, ElogSummaryPlots
from lute.execution.logging import get_logger

import psana  # type: ignore
import os
import logging
from typing import Optional, Tuple
import sys

sys.path.append("/sdf/home/l/lconreux/LCLSGeom")
from LCLSGeom.swap_geom import (  # type: ignore
    PsanaToPyFAI,
    PyFAIToCrystFEL,
    CrystFELToPsana,
    get_beam_center,
)

import h5py  # type: ignore
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt  # type: ignore
import pyFAI  # type: ignore
from pyFAI.geometry import Geometry  # type: ignore
from pyFAI.goniometer import SingleGeometry  # type: ignore
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator  # type: ignore
from pyFAI.calibrant import CALIBRANT_FACTORY  # type: ignore
from sklearn.gaussian_process import GaussianProcessRegressor  # type: ignore
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel  # type: ignore
from sklearn.utils._testing import ignore_warnings  # type: ignore
from sklearn.exceptions import ConvergenceWarning  # type: ignore
from scipy.stats import norm  # type: ignore
from mpi4py import MPI

pyFAI.use_opencl = False  # type: ignore

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
            logger.info(f"Threshold for pixel outliers: {threshold:.2e}")
        nice_pix = powder < threshold
        SNRs = []
        Imins = np.arange(99.5, 100, 0.01)
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
        Imin=90,
        max_rings=5,
        n_samples=50,
        n_iterations=50,
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
        y_norm = (y - np.mean(y)) / np.std(y)
        best_score = np.max(y_norm)

        kernel = RBF(length_scale=0.3, length_scale_bounds=(0.2, 0.4)) * ConstantKernel(
            constant_value=1.0, constant_value_bounds=(0.5, 1.5)
        ) + WhiteKernel(noise_level=0.001, noise_level_bounds="fixed")
        gp_model = GaussianProcessRegressor(
            kernel=kernel, n_restarts_optimizer=10, random_state=42
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
            y = np.append(y, [score], axis=0)
            bo_history[f"iteration_{i+1}"] = {
                "param": X[new_idx],
                "score": score,
            }
            X_samples = np.append(X_samples, [X[new_idx]], axis=0)
            X_norm_samples = np.append(X_norm_samples, [X_norm[new_idx]], axis=0)
            y_norm = (y - np.mean(y)) / np.std(y)
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
        Imin=99,
        max_rings=5,
        n_samples=50,
        n_iterations=50,
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
        Imin : float
            Minimum intensity threshold for control point extraction based on intensity distribution percentile
        max_rings : int
            Maximum number of rings to use for control point extraction
        n_samples : int
            Number of samples to initialize the GP model
        n_iterations : int
            Number of iterations for optimization
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

        if self.rank == 0:
            logger.info(
                f"Optimizing geometry for exp {self.exp} run {self.run} with {self.det_type} detector with minimal intensity threshold {Imin:.2e}"
            )
            logger.info(f"Number of distances to scan: {self.size}")
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
            index = np.argmin(self.scan["residual"])
            self.index = index
            self.bo_history = self.scan["bo_history"][index]
            self.params = self.scan["params"][index]
            self.residual = self.scan["residual"][index]
            self.score = self.scan["score"][index]
            self.best_idx = self.scan["best_idx"][index]

    def display(self, powder=None, cp=None, ai=None, label=None, sg=None, ax=None):
        """
        Display an image with the control points and the calibrated rings

        Parameters
        ----------
        powder : np.ndarray
        """
        if ax is None:
            _fig, ax = plt.subplots()
        if sg is not None:
            if powder is None:
                powder = sg.image
            if cp is None:
                cp = sg.control_points
            if ai is None:
                ai = sg.geometry_refinement
            if label is None:
                label = sg.label
        img = ax.imshow(
            powder.T,
            origin="lower",
            cmap="viridis",
            vmin=np.percentile(powder, 5),
            vmax=np.percentile(powder, 95),
        )
        cbar = plt.colorbar(img, ax=ax, orientation="vertical")
        cbar.set_label("Intensity")
        if ai is not None and cp.calibrant is not None:
            tth = cp.calibrant.get_2th()
            ttha = ai.twoThetaArray()
            ax.contour(
                ttha.T, levels=tth, cmap="autumn", linewidths=0.5, linestyles="dashed"
            )
        return ax

    def radial_integration(self, result, calibrant=None, label=None, ax=None):
        """
        Display the powder diffraction pattern

        Parameters
        ----------
        result : np.ndarray
            Powder diffraction pattern
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

        try:
            unit = result.unit
        except AttributeError:
            unit = None
        if len(result) == 3:
            ax.errorbar(*result, label=label)
        else:
            ax.plot(*result, label=label)

        if label:
            ax.legend()
        if calibrant and unit:
            x_values = calibrant.get_peaks(unit)
            if x_values is not None:
                for x in x_values:
                    line = lines.Line2D(
                        [x, x],
                        ax.axis()[2:4],
                        color="red",
                        linestyle="--",
                        linewidth=0.5,
                    )
                    ax.add_line(line)

        ax.set_title("Radial Profile")
        if unit:
            ax.set_xlabel(unit.label)
        ax.set_ylabel("Intensity")

    def score_distance_scan(self, distances, ax):
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
        ax.set_xlabel("Distance (m)")
        ax.set_ylabel("Score")
        ax.set_title("Number of Control Points vs Distance")

    def residual_distance_scan(self, distances, ax):
        """
        Plot the residual scan over distance

        Parameters
        ----------
        distances : np.array
            Array of distances
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
            label=f"Best distance: {best_dist:.2e}",
        )
        ax.legend(fontsize="x-small")
        ax.set_yscale("log")
        ax.set_xlabel("Distance (m)")
        ax.set_ylabel("Residual")
        ax.set_title("Residual vs Distance")

    def hist_and_compute_stats(self, powder, exp, run, ax):
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
        nice_pix = powder < threshold
        _ = ax.hist(
            powder[nice_pix],
            bins=1000,
            color="skyblue",
            edgecolor="black",
            alpha=0.7,
            label="Pixel Intensities",
        )
        ax.axvline(
            mean,
            color="red",
            linestyle="--",
            label=f"Mean ({mean:.2f})",
        )
        ax.axvline(
            mean + std_dev,
            color="orange",
            linestyle="--",
            label=f"Mean + Std Dev ({mean + std_dev:.2f})",
        )
        ax.axvline(
            mean + 2 * std_dev,
            color="green",
            linestyle="--",
            label=f"Mean + 2 Std Dev ({mean + 2 * std_dev:.2f})",
        )
        ax.axvline(
            self.Imin,
            color="purple",
            linestyle=":",
            linewidth=1.5,
            label=f"{self.q} th Percentile ({self.Imin:.2f})",
        )
        ax.set_xlim([0, mean + 5 * std_dev])
        ax.set_xlabel("Pixel Intensity")
        ax.set_ylabel("Frequency")
        ax.set_title(f"Histogram of Pixel Intensities \n for {exp} run {run}")
        ax.legend(fontsize="x-small")

    def visualize_results(self, powder, bo_history, detector, params, plot=""):
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
        params : list
            List of parameters for the best fit
        plot : str
            Path to save plot
        """
        fig = plt.figure(figsize=(12, 16), dpi=180)
        nrow, ncol = 3, 2
        irow, icol = 0, 0

        # Plotting BO convergence
        ax1 = plt.subplot2grid((nrow, ncol), (irow, icol))
        scores = [bo_history[key]["score"] for key in bo_history.keys()]
        ax1.plot(np.maximum.accumulate(scores))
        ax1.set_xticks(np.arange(len(scores), step=20))
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Best score so far")
        ax1.set_title(f"Convergence Plot, best score: {int(self.score)}")
        icol += 1

        # Plotting histogram of pixel intensities
        ax2 = plt.subplot2grid((nrow, ncol), (irow, icol), colspan=ncol - icol)
        self.hist_and_compute_stats(powder, self.exp, self.run, ax2)
        irow += 1
        icol = 0

        # Plotting radial profiles with peaks
        ax3 = plt.subplot2grid((nrow, ncol), (irow, icol))
        ai = AzimuthalIntegrator(
            dist=params[0], detector=detector, wavelength=self.calibrant.wavelength
        )
        masked_powder = powder
        if self.det_type.lower() == "rayonix":
            radius = np.sqrt(2) * powder.shape[0] / 4
            row, col = np.ogrid[: powder.shape[0], : powder.shape[1]]
            center = (powder.shape[0] / 2, powder.shape[1] / 2)
            mask = ((row - center[0]) ** 2 + (col - center[1]) ** 2) <= radius**2
            masked_powder = powder * mask
        res = ai.integrate1d(masked_powder, 1000)
        self.radial_integration(res, calibrant=self.calibrant, ax=ax3)
        icol += 1

        # Plotting stacked powder
        ax4 = plt.subplot2grid((nrow, ncol), (irow, icol), colspan=ncol - icol)
        geometry = Geometry(dist=params[0])
        sg = SingleGeometry(
            f"Max {self.calibrant_name}",
            powder,
            calibrant=self.calibrant,
            detector=detector,
            geometry=geometry,
        )
        sg.extract_cp(max_rings=self.max_rings, pts_per_deg=1, Imin=self.Imin)
        self.display(sg=sg, ax=ax4)
        irow += 1
        icol = 0

        # Plotting score scan over distance
        ax5 = plt.subplot2grid((nrow, ncol), (irow, icol))
        self.score_distance_scan(self.distances, ax5)
        icol += 1

        # Plotting residual scan over distance
        ax6 = plt.subplot2grid((nrow, ncol), (irow, icol), colspan=ncol - icol)
        self.residual_distance_scan(self.distances, ax6)

        if plot != "":
            fig.savefig(plot, dpi=180)


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
        ds_args = f"exp={self._task_parameters.exp}:run={self._task_parameters.run}:idx"
        self.ds = psana.DataSource(ds_args)
        self.det = psana.Detector(det_type, self.ds.env())
        self.shape = self.det.shape()
        if det_type.lower() == "rayonix":
            env = self.ds.env()
            cfg = env.configStore()
            self.pixel_size = cfg.get(psana.Rayonix.ConfigV2).pixelWidth() * 1e-6
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

    def _check_if_path_and_type(self, string: str) -> Tuple[bool, Optional[str]]:
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
            is_valid, dtype = self._check_if_path_and_type(powder_path)
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

    def _update_geometry(self, optimizer):
        """
        Update the geometry and write a new .geom file and .data file

        Parameters
        ----------
        optimizer : BayesGeomOpt
            Optimizer object
        """
        assert isinstance(self._task_parameters, OptimizePyFAIGeometryParameters)
        PyFAIToCrystFEL(
            detector=optimizer.detector,
            params=optimizer.params,
            psana_file=self._task_parameters.in_file,
            out_file=self._task_parameters.out_file.replace(
                f"{self._task_parameters.run}-end.data",
                f"r{self._task_parameters.run:0>4}.geom",
            ),
        )
        CrystFELToPsana(
            in_file=self._task_parameters.out_file.replace(
                f"{self._task_parameters.run}-end.data",
                f"r{self._task_parameters.run:0>4}.geom",
            ),
            det_type=optimizer.det_type,
            out_file=self._task_parameters.out_file,
            pixel_size=self.pixel_size,
            shape=self.shape,
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
        assert isinstance(self._task_parameters, OptimizePyFAIGeometryParameters)
        detector = self._build_pyFAI_detector()
        powder = self._extract_powder(self._task_parameters.powder, detector.shape)
        if powder is None:
            raise RuntimeError("Unable to extract powder. Cannot continue.")
        optimizer = BayesGeomOpt(
            exp=self._task_parameters.exp,
            run=self._task_parameters.run,
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
            af=self._task_parameters.bo_params.af,
            hyperparam=self._task_parameters.bo_params.hyperparams,
            prior=self._task_parameters.bo_params.prior,
            seed=self._task_parameters.bo_params.seed,
        )
        if optimizer.rank == 0:
            logger.info("Optimization complete")
            distance, cx, cy = get_beam_center(optimizer.params)
            logger.info(f"Detector Distance to Sample: {distance:.2e}")
            logger.info(f"Beam center: ({cx:.2e}, {cy:.2e})")
            logger.info(
                f"Rotations: \u03B8x = ({optimizer.params[3]:.2e}, \u03B8y = {optimizer.params[4]:.2e}, \u03B8z = {optimizer.params[5]:.2e})"
            )
            logger.info(f"Final Residual: {optimizer.residual:.2e}")
            fig_folder = os.path.join(self._task_parameters.work_dir, "figs")
            os.makedirs(fig_folder, exist_ok=True)
            plot = (
                f"{fig_folder}/bayes_opt_geom_{optimizer.exp}_r{optimizer.run:0>4}.png"
            )
            calib_detector = self._update_geometry(optimizer)
            optimizer.visualize_results(
                powder=optimizer.powder,
                bo_history=optimizer.bo_history,
                detector=calib_detector,
                params=optimizer.params,
                plot=plot,
            )
            p1, p2, _ = calib_detector.calc_cartesian_positions()
            cx_pix = np.abs(cx - np.min(p1)) / calib_detector.pixel1
            cy_pix = np.abs(cy - np.min(p2)) / calib_detector.pixel2
            theta: float = np.arctan(cx / distance)
            q: float = (
                2.0 * np.sin(theta / 2.0) / (optimizer.calibrant.wavelength * 1e10)
            )
            edge_resolution: float = 1.0 / q
            self._result.payload = plot
            self._result.summary = []
            self._result.summary.append(
                {
                    "Detector distance (m)": distance,
                    "Detector center (pixels)": (cx_pix, cy_pix),
                    "Detector edge resolution (A)": edge_resolution,
                }
            )
            logger.info(f"Beam center (pixels): ({cx_pix}, {cy_pix})")
            logger.info(f"Detector edge resolution (A): {edge_resolution}")
            self._result.summary.append(
                ElogSummaryPlots(f"Geometry_Fit/r{self._task_parameters.run:0>4}", plot)
            )
            self._result.task_status = TaskStatus.COMPLETED
