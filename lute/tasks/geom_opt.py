"""
Classes for geometry optimization tasks.

Classes:
    OptimizePyFAIGeom: optimize detector geometry using PyFAI coupled with Bayesian Optimization 

"""

__all__ = ["OptimizePyFAIGeometry"]
__author__ = "Louis Conreux"

import sys
from pathlib import Path
from typing import Any, Dict, List, Literal, TextIO, Tuple, Optional

from lute.execution.ipc import Message
from lute.io.models.geom_opt import OptimizePyFAIGeometryParameters
from lute.tasks.task import Task

sys.path.append('/sdf/home/l/lconreux/LCLSGeom')
from LCLSGeom.swap_geom import PsanaToPyFAI, PyFAIToCrystFEL, CrystFELToPsana

import numpy as np
import matplotlib.pyplot as plt
from pyFAI.geometry import Geometry
from pyFAI.goniometer import SingleGeometry
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator
from pyFAI.calibrant import CALIBRANT_FACTORY
from mpi4py import MPI
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from scipy.stats import norm

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
    wavelength : float
        Wavelength of the experiment
    """

    def __init__(
        self,
        exp,
        run,
        det_type,
        detector,
        calibrant,
        wavelength,
    ):
        self.exp = exp
        self.run = run
        self.det_type = det_type.lower()
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.detector = detector
        self.calibrant = calibrant
        self.wavelength = wavelength
        self.order = ["dist", "poni1", "poni2", "rot1", "rot2", "rot3"]
        self.space = ["poni1", "poni2"]
        self.values = {'dist': 0.1,'poni1':0, 'poni2':0, 'rot1':0, 'rot2':0, 'rot3':0}

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
        photon_energy = 1.23984197386209e-09 / self.wavelength
        self.photon_energy = photon_energy
        calibrant.wavelength = self.wavelength
        self.calibrant = calibrant

    def min_intensity(self, Imin):
        """
        Define minimal intensity for control point extraction
        Note: this is a heuristic that has been found to work well but may need some tuning.

        Parameters
        ----------
        Imin : int or str
            Minimum intensity to use for control point extraction based on photon energy or max intensity
        """
        if type(Imin) == str:
            if 'rayonix' not in self.det_type: 
                Imin = np.max(self.powder_img) * 0.01
            else:
                self.powder_img = self.powder_img[self.powder_img > 1e3]
                Imin = np.max(self.powder_img) * 0.01
        else:
            Imin = Imin * self.photon_energy
        self.Imin = Imin

    @ignore_warnings(category=ConvergenceWarning)
    def bayes_opt_center(self, powder_img, dist, bounds, res, n_samples=50, num_iterations=50, af="ucb", hyperparam=None, prior=True, seed=None):
        """
        Perform Bayesian Optimization on PONI center parameters, for a fixed distance
        
        Parameters
        ----------
        powder_img : np.ndarray
            Powder image
        dist : float
            Fixed distance
        bounds : dict
            Dictionary of bounds for each parameter
        res : float
            Resolution of the grid used to discretize the parameter search space
        values : dict
            Dictionary of values for fixed parameters
        n_samples : int
            Number of samples to initialize the GP model
        num_iterations : int
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

        self.values['dist'] = dist

        if res is None:
            res = self.detector.pixel_size

        inputs = {}
        norm_inputs = {}
        for p in self.order:
            if p in self.space:
                inputs[p] = np.arange(bounds[p][0], bounds[p][1]+res, res)
                norm_inputs[p] = inputs[p]
            else:
                inputs[p] = np.array([self.values[p]])
        X = np.array(np.meshgrid(*[inputs[p] for p in self.order])).T.reshape(-1, len(self.order))
        X_space = np.array(np.meshgrid(*[norm_inputs[p] for p in self.space])).T.reshape(-1, len(self.space))
        X_norm = (X_space - np.mean(X_space, axis=0)) / (np.max(X_space, axis=0) - np.min(X_space, axis=0))
        if prior:
            means = np.mean(X_space, axis=0)
            cov = np.diag([((bounds[param][1] - bounds[param][0]) / 5)**2 for param in self.space])
            X_samples = np.random.multivariate_normal(means, cov, n_samples)
            X_norm_samples = (X_samples - np.mean(X_space, axis=0)) / (np.max(X_space, axis=0) - np.min(X_space, axis=0))
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
            geom_initial = Geometry(dist=dist, poni1=poni1, poni2=poni2, rot1=rot1, rot2=rot2, rot3=rot3, detector=self.detector, wavelength=self.calibrant.wavelength)
            sg = SingleGeometry("extract_cp", powder_img, calibrant=self.calibrant, detector=self.detector, geometry=geom_initial)
            sg.extract_cp(max_rings=5, pts_per_deg=1, Imin=self.Imin)
            y[i] = len(sg.geometry_refinement.data)
            bo_history[f'init_sample_{i+1}'] = {'param':X_samples[i], 'score': y[i]}

        y_norm = (y - np.mean(y)) / np.std(y)
        best_score = np.max(y_norm)

        kernel = RBF(length_scale=0.3, length_scale_bounds=(0.2, 0.4)) \
                * ConstantKernel(constant_value=1.0, constant_value_bounds=(0.5, 1.5)) \
                + WhiteKernel(noise_level=0.001, noise_level_bounds = 'fixed')
        gp_model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=42)
        gp_model.fit(X_norm_samples, y_norm)
        visited_idx = list([])

        if af == "ucb":
            if hyperparam is None:
                hyperparam = {'beta': 1.96}
            hyperparam = hyperparam['beta']
            af = self.upper_confidence_bound
        elif af == "ei":
            if hyperparam is None:
                hyperparam = {'epsilon': 0}
            hyperparam = hyperparam['epsilon']
            af = self.expected_improvement
        elif af == "pi":
            if hyperparam is None:
                hyperparam = {'epsilon': 0}
            hyperparam = hyperparam['epsilon']
            af = self.probability_of_improvement
        elif af == "ci":
            af = self.contextual_improvement

        for i in range(num_iterations):
            # 1. Generate the Acquisition Function values using the Gaussian Process Regressor
            af_values = af(X_norm, gp_model, best_score, hyperparam)
            af_values[visited_idx] = -np.inf         
            
            # 2. Select the next set of parameters based on the Acquisition Function
            new_idx = np.argmax(af_values)
            new_input = X[new_idx]
            visited_idx.append(new_idx)

            # 3. Compute the score of the new set of parameters
            dist, poni1, poni2, rot1, rot2, rot3 = new_input
            geom_initial = Geometry(dist=dist, poni1=poni1, poni2=poni2, rot1=rot1, rot2=rot2, rot3=rot3, detector=self.detector, wavelength=self.calibrant.wavelength)
            sg = SingleGeometry("extract_cp", powder_img, calibrant=self.calibrant, detector=self.detector, geometry=geom_initial)
            sg.extract_cp(max_rings=5, pts_per_deg=1, Imin=self.Imin)
            score = len(sg.geometry_refinement.data)
            y = np.append(y, [score], axis=0)
            ypred = gp_model.predict(X_norm, return_std=False)
            bo_history[f'iteration_{i+1}'] = {'param':X[new_idx], 'score': score, 'pred': ypred, 'af': af_values}
            X_samples = np.append(X_samples, [X[new_idx]], axis=0)
            X_norm_samples = np.append(X_norm_samples, [X_norm[new_idx]], axis=0)
            y_norm = (y - np.mean(y)) / np.std(y)
            best_score = np.max(y_norm)
            # 4. Update the Gaussian Process Regressor
            gp_model.fit(X_norm_samples, y_norm)
        
        best_idx = np.argmax(y_norm)
        best_param = X_samples[best_idx]
        dist, poni1, poni2, rot1, rot2, rot3 = best_param
        geom_initial = Geometry(dist=dist, poni1=poni1, poni2=poni2, rot1=rot1, rot2=rot2, rot3=rot3, detector=self.detector, wavelength=self.calibrant.wavelength)
        sg = SingleGeometry("extract_cp", powder_img, calibrant=self.calibrant, detector=self.detector, geometry=geom_initial)
        sg.extract_cp(max_rings=5, pts_per_deg=1, Imin=self.Imin)
        self.sg = sg
        residuals = sg.geometry_refinement.refine3(fix=['wavelength'])
        params = sg.geometry_refinement.param
        result = {'bo_history': bo_history, 'params': params, 'residuals': residuals, 'best_idx': best_idx}
        return result

    def bayes_opt_geom(self, powder, bounds, res, Imin='max', n_samples=50, num_iterations=50, af="ucb", hyperparam=None, prior=True, seed=None):
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
        Imin : int or str
            Minimum intensity to use for control point extraction based on photon energy or max intensity
        values : dict
            Dictionary of values for fixed parameters
        n_samples : int
            Number of samples to initialize the GP model
        num_iterations : int
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

        self.min_intensity(Imin)

        if self.rank == 0:
            distances = np.linspace(bounds['dist'][0], bounds['dist'][1], self.size)
        else:
            distances = None

        dist = self.comm.scatter(distances, root=0)
        print(f"Rank {self.rank} is working on distance {dist}")

        results = self.bayes_opt_center(powder, dist, bounds, res, n_samples, num_iterations, af, hyperparam, prior, seed)
        self.comm.Barrier()

        self.scan = {}
        self.scan['bo_history'] = self.comm.gather(results['bo_history'], root=0)
        self.scan['params'] = self.comm.gather(results['params'], root=0)
        self.scan['residuals'] = self.comm.gather(results['residuals'], root=0)
        self.scan['best_idx'] = self.comm.gather(results['best_idx'], root=0)
        self.finalize()

    def finalize(self):
        if self.rank == 0:
            for key in self.scan.keys():
                self.scan[key] = np.array([item for item in self.scan[key]])  
            index = np.argmin(self.scan['residuals']) 
            self.bo_history = self.scan['bo_history'][index]
            self.params = self.scan['params'][index]
            self.residuals = self.scan['residuals'][index]
            self.best_idx = self.scan['best_idx'][index]

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
        ax.imshow(powder.T,
                origin="lower",
                cmap="viridis",
                vmax=self.Imin)
        ax.set_title(label)
        if ai is not None and cp.calibrant is not None:
            tth = cp.calibrant.get_2th()
            ttha = ai.twoThetaArray()
            ax.contour(ttha.T, levels=tth, cmap="autumn", linewidths=0.5, linestyles="dashed")
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
                    line = lines.Line2D([x, x], ax.axis()[2:4],
                                        color='red', linestyle='--', linewidth=0.5)
                    ax.add_line(line)

        ax.set_title("Radial Profile")
        if unit:
            ax.set_xlabel(unit.label)
        ax.set_ylabel("Intensity")

    def visualize_results(self, powder, bo_history, detector, params, plot=''):
        """
        Visualize fit, plotting (1) the BO convergence, (2) the radial profile and (3) the powder image.

        Parameters
        ----------
        powder : np.ndarray
            Powder image
        bo_history : dict
            Dictionary containing the history of optimization
        detector : PyFAI(Detector)
            PyFAI detector object
        params : list
            List of parameters for the best fit
        plot : str
            Path to save plot
        """
        fig = plt.figure(figsize=(8,8),dpi=120)
        nrow,ncol=2,2
        irow,icol=0,0
        
        # Plotting BO convergence
        ax1 = plt.subplot2grid((nrow, ncol), (irow, icol))
        scores = [bo_history[key]['score'] for key in bo_history.keys()]
        ax1.plot(np.maximum.accumulate(scores))
        ax1.set_xticks(np.arange(len(scores), step=20))
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Best score so far')
        ax1.set_title('Convergence Plot')
        icol += 1

        # Plotting radial profiles with peaks
        ax2 = plt.subplot2grid((nrow, ncol), (irow, icol), colspan=ncol-icol)
        ai = AzimuthalIntegrator(dist=params[0], detector=detector, wavelength=self.calibrant.wavelength)
        res = ai.integrate1d(powder, 1000)
        self.radial_integration(res, calibrant=self.calibrant, ax=ax2)
        irow += 1

        # Plotting stacked powder
        geometry = Geometry(dist=params[0])
        sg = SingleGeometry(f'Max {self.calibrant_name}', powder, calibrant=self.calibrant, detector=detector, geometry=geometry)
        sg.extract_cp(max_rings=5, pts_per_deg=1, Imin=self.Imin)
        ax3 = plt.subplot2grid((nrow, ncol), (irow, 0), rowspan=nrow-irow, colspan=ncol)
        self.display(sg=sg, ax=ax3)

        if plot != '':
            fig.savefig(plot, dpi=300)

class OptimizePyFAIGeometry(Task):
    """Optimize detector geometry using PyFAI coupled with Bayesian Optimization."""

    def __init__(self, *, params: OptimizePyFAIGeometryParameters, use_mpi: bool = True) -> None:
        super().__init__(params=params, use_mpi=use_mpi)

    def _run(self) -> None:
        detector = self.build_pyFAI_detector()
        optimizer = BayesGeomOpt(
            exp=self._task_parameters.exp,
            run=self._task_parameters.run,
            det_type=self._task_parameters.det_type,
            detector=detector,
            calibrant=self._task_parameters.calibrant,
            wavelength=self._task_parameters.wavelength,
        )
        optimizer.bayes_opt_geom(
            powder=self._task_parameters.powder,
            bounds=self._task_parameters.bo_params.bounds,
            res=self._task_parameters.bo_params.res,
            Imin=self._task_parameters.bo_params.Imin,
            n_samples=self._task_parameters.bo_params.n_samples,
            num_iterations=self._task_parameters.bo_params.n_iterations,
            af=self._task_parameters.bo_params.af,
            hyperparam=self._task_parameters.bo_params.hyperparams,
            prior=self._task_parameters.bo_params.prior,
            seed=self._task_parameters.bo_params.seed,
        )
        if optimizer.rank == 0:
            detector = self.update_geometry(optimizer)
            plot = f'{self._task_parameters.work_dir}/figs/bayes_opt_geom_r{optimizer.run:04}.png'
            optimizer.visualize_results(
                powder=optimizer.powder_img,
                bo_history=optimizer.bo_history,
                detector=detector,
                params=optimizer.params,
                plot=plot,
            )

    def build_pyFAI_detector(self):
        """
        Fetch the geometry data and build a pyFAI detector object.        
        """
        in_file = self._task_parameters.in_file
        det_type = self._task_parameters.det_type
        psana_to_pyfai = PsanaToPyFAI(in_file=in_file, det_type=det_type)
        detector = psana_to_pyfai.detector
        return detector
    
    def update_geometry(self, optimizer):
        """
        Update the geometry and write a new .geom file and .data file
        """
        PyFAIToCrystFEL(detector=optimizer.detector, params=optimizer.params, psana_file=self._task_parameters.in_file, out_file=self._task_parameters.out_file.replace("0-end.data", f"r{optimizer.run:04}.geom"))
        CrystFELToPsana(in_file=self._task_parameters.out_file.replace("0-end.data", f"r{optimizer.run:04}.geom"), det_type=optimizer.det_type, out_file=self._task_parameters.out_file)
        psana_to_pyfai = PsanaToPyFAI(in_file=self.in_file.replace("0-end.data", f"{optimizer.run}-end.data"), det_type=optimizer.det_type)
        detector = psana_to_pyfai.detector
        return detector