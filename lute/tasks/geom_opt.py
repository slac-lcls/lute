"""
Classes for geometry optimization tasks.

Classes:
    OptimizePyFAIGeom: optimize detector geometry using PyFAI coupled with Bayesian Optimization 

"""

__all__ = ["OptimizePyFAIGeometry"]
__author__ = "Louis Conreux"

from lute.execution.ipc import Message
from lute.io.models.geom_opt import *
from lute.tasks.task import *
from lute.tasks.dataclasses import *

import psana

import sys

sys.path.append("/sdf/home/l/lconreux/LCLSGeom")
from LCLSGeom.swap_geom import (
    PsanaToPyFAI,
    PyFAIToCrystFEL,
    CrystFELToPsana,
    get_beam_center,
)

import logging
from lute.execution.logging import get_logger

logger: logging.Logger = get_logger(__name__)

import numpy as np
import matplotlib.pyplot as plt
from pyFAI.geometry import Geometry
from pyFAI.goniometer import SingleGeometry
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator
from pyFAI.calibrant import CALIBRANT_FACTORY
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from scipy.stats import norm
from scipy.signal import find_peaks
from mpi4py import MPI


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

    def build_calibrant(self):
        """
        Define calibrant for optimization

        Parameters
        ----------
        wavelength : float
            Wavelength of the experiment
        """
        self.calibrant_name = self.calibrant
        calibrant = CALIBRANT_FACTORY(self.calibrant)
        ds_args = f"exp={self.exp}:run={self.run}:idx"
        ds = psana.DataSource(ds_args)
        self.wavelength = ds.env().epicsStore().value("SIOC:SYS0:ML00:AO192") * 1e-9
        photon_energy = 1.23984197386209e-09 / self.wavelength
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
        except:
            mask = None
        if mask is not None:
            if len(mask.shape) != 2:
                mask = np.reshape(mask, (mask.shape[0] * mask.shape[1], mask.shape[2]))
        return mask

    def min_intensity(self, Imin, powder):
        """
        Define minimal intensity for control point extraction
        Note: this is a heuristic that has been found to work well but may need some tuning.

        Parameters
        ----------
        Imin : float
            Minimum intensity to use for control point extraction based on intensity distribution
        powder : np.ndarray
            Powder image
        """
        masked_powder = np.ma.masked_array(powder, 0)
        mean = np.mean(masked_powder)
        std = np.std(masked_powder)
        threshold = mean + 3 * std
        nice_pix = masked_powder < threshold
        Imin = np.percentile(masked_powder[nice_pix], Imin)
        self.Imin = Imin
        self.powder = powder
        return Imin, powder

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
            ypred = gp_model.predict(X_norm, return_std=False)
            bo_history[f"iteration_{i+1}"] = {
                "param": X[new_idx],
                "score": score,
                "pred": ypred,
                "af": af_values,
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
        residual = sg.geometry_refinement.refine3(fix=["wavelength"])
        score = len(sg.geometry_refinement.data)
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

        powder = np.load(powder)

        self.build_calibrant()

        mask = self.build_mask()
        if mask is not None:
            powder = powder * mask

        self.max_rings = max_rings
        Imin, powder = self.min_intensity(Imin, powder)

        if self.rank == 0:
            logger.info(f"Number of distances to scan: {self.size}")
            self.bounds = bounds
            distances = np.linspace(bounds["dist"][0], bounds["dist"][1], self.size)
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
            non_zero_scores = np.where(self.scan["score"] > 0)[0]
            percentile_10 = np.percentile(self.scan["score"][non_zero_scores], 10)
            mean = np.mean(self.scan["score"][non_zero_scores])
            std_dev = np.std(self.scan["score"][non_zero_scores])
            logger.info(f"Mean Score: {mean:.2e}")
            logger.info(f"Score Std Dev: {std_dev:.2e}")
            logger.info(f"10th Score Percentile: {percentile_10:.2e}")
            peaks, _ = find_peaks(self.scan["score"], distance=2)
            shift_index = np.argmin(self.scan["residual"][peaks])
            index = peaks[shift_index]
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
        except:
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

    def score_distance_scan(self, bounds, ax):
        """
        Plot the score scan over distance

        Parameters
        ----------
        bounds : dict
            Dictionary of bounds for each parameter
        ax : plt.Axes
            Matplotlib axes
        """
        scores = self.scan["score"]
        non_zero_scores = np.where(scores > 0)[0]
        mean = np.mean(scores[non_zero_scores])
        std_dev = np.std(scores[non_zero_scores])
        percentile_10 = np.percentile(scores[non_zero_scores], 10)
        peaks, _ = find_peaks(scores, distance=2)
        distances = np.linspace(bounds["dist"][0], bounds["dist"][1], len(scores))
        ax.plot(distances, scores)
        ax.axhline(
            percentile_10,
            color="purple",
            linestyle="--",
            label=f"10th Percentile: {percentile_10:.2e}",
        )
        ax.axhline(
            mean, color="red", linestyle="--", linewidth=1.5, label=f"Mean ({mean:.2f})"
        )
        ax.plot(distances[peaks], scores[peaks], "x")
        ax.set_xlabel("Distance (m)")
        ax.set_ylabel("Score")
        ax.legend(fontsize="x-small")
        ax.set_title("Number of Control Points vs Distance")

    def residual_distance_scan(self, bounds, ax):
        """
        Plot the residual scan over distance

        Parameters
        ----------
        bounds : dict
            Dictionary of bounds for each parameter
        ax : plt.Axes
            Matplotlib axes
        """
        residuals = self.scan["residual"]
        distances = np.linspace(bounds["dist"][0], bounds["dist"][1], len(residuals))
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
        mean = np.mean(powder)
        threshold = np.mean(powder) + 2 * np.std(powder)
        nice_pix = powder < threshold
        mean = np.mean(powder[nice_pix])
        std_dev = np.std(powder[nice_pix])
        percentile_99 = np.percentile(powder[nice_pix], 99)
        _ = ax.hist(
            powder[nice_pix],
            bins=1000,
            color="skyblue",
            edgecolor="black",
            alpha=0.7,
            label="Pixel Intensities",
        )
        ax.axvline(
            mean, color="red", linestyle="--", linewidth=1.5, label=f"Mean ({mean:.2f})"
        )
        ax.axvline(
            mean + std_dev,
            color="orange",
            linestyle="--",
            linewidth=1.5,
            label=f"Mean + 1 Std ({mean + std_dev:.2f})",
        )
        ax.axvline(
            mean + 2 * std_dev,
            color="green",
            linestyle="--",
            linewidth=1.5,
            label=f"Mean + 2 Std ({mean + 2 * std_dev:.2f})",
        )
        ax.axvline(
            mean + 3 * std_dev,
            color="blue",
            linestyle="--",
            linewidth=1.5,
            label=f"Mean + 3 Std ({mean + 3 * std_dev:.2f})",
        )
        ax.axvline(
            percentile_99,
            color="purple",
            linestyle=":",
            linewidth=1.5,
            label=f"99th Percentile ({percentile_99:.2f})",
        )
        ax.set_xlim(0, mean + 5 * std_dev)
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
        self.score_distance_scan(self.bounds, ax5)
        icol += 1

        # Plotting residual scan over distance
        ax6 = plt.subplot2grid((nrow, ncol), (irow, icol), colspan=ncol - icol)
        self.residual_distance_scan(self.bounds, ax6)

        if plot != "":
            fig.savefig(plot, dpi=180)


class OptimizePyFAIGeometry(Task):
    """Optimize detector geometry using PyFAI coupled with Bayesian Optimization."""

    def __init__(
        self, *, params: OptimizePyFAIGeometryParameters, use_mpi: bool = True
    ) -> None:
        super().__init__(params=params, use_mpi=use_mpi)

    def _run(self) -> None:
        detector = self.build_pyFAI_detector()
        optimizer = BayesGeomOpt(
            exp=self._task_parameters.exp,
            run=self._task_parameters.run,
            det_type=self._task_parameters.det_type,
            detector=detector,
            calibrant=self._task_parameters.calibrant,
        )
        optimizer.bayes_opt_geom(
            powder=self._task_parameters.powder,
            bounds=self._task_parameters.bo_params.bounds,
            res=self._task_parameters.bo_params.res,
            Imin=self._task_parameters.bo_params.Imin,
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
            logger.info(f"Final Residuals: {optimizer.residual:.2e}")
            plot = f"{self._task_parameters.work_dir}/figs/bayes_opt_geom_{optimizer.exp}_r{optimizer.run:0>4}.png"
            detector = self.update_geometry(optimizer)
            optimizer.visualize_results(
                powder=optimizer.powder,
                bo_history=optimizer.bo_history,
                detector=detector,
                params=optimizer.params,
                plot=plot,
            )
            self._result.payload = plot
            self._result.task_status = TaskStatus.COMPLETED

    def build_pyFAI_detector(self):
        """
        Fetch the geometry data and build a pyFAI detector object.
        """
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

    def update_geometry(self, optimizer):
        """
        Update the geometry and write a new .geom file and .data file
        """
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
