"""
Classes for geometry optimization tasks.

Classes:
    OptimizePyFAIGeom: optimize detector geometry using PyFAI coupled with Bayesian Optimization

"""

__all__ = ["OptimizePyFAIGeometry"]
__author__ = "Louis Conreux"

import os
from lute.io.models.bayfai import OptimizePyFAIGeometryParameters
from lute.tasks.task import Task
from lute.tasks.dataclasses import TaskStatus, ElogSummaryPlots
from lute.tasks._bayfai import BayesGeomOpt, generate_powder, build_detector, define_calibrant, update_geometry
from lute.execution.logging import get_logger
import logging
from typing import Optional

import h5py  # type: ignore
import panel as pn  # type: ignore
import numpy as np
import numpy.typing as npt
import time  # type: ignore

logger: logging.Logger = get_logger(__name__)

class OptimizePyFAIGeometry(Task):
    """Optimize detector geometry using PyFAI coupled with Bayesian Optimization."""

    def __init__(
        self, *, params: OptimizePyFAIGeometryParameters, use_mpi: bool = True
    ) -> None:
        super().__init__(params=params, use_mpi=use_mpi)

    def _run(self) -> None:
        start_time = time.time()
        assert isinstance(self._task_parameters, OptimizePyFAIGeometryParameters)
        exp = self._task_parameters.lute_config.experiment
        run = self._task_parameters.lute_config.run

        # Process Diffraction Powder
        powder, Imin = generate_powder(self._task_parameters.powder, self._task_parameters.det_name, self._task_parameters.preprocess)
        if powder is None:
            raise RuntimeError("Unable to extract powder. Cannot continue.")

        # Build Detector
        detector = build_detector(self._task_parameters.in_file, powder.shape)

        # Setup Calibrant
        calibrant = define_calibrant(self._task_parameters.calibrant, exp, run)

        # Initialize Optimizer
        optimizer = BayesGeomOpt(
            exp=exp,
            run=run,
            detector=detector,
            powder=powder,
            calibrant=calibrant,
            fixed=self._task_parameters.fixed,
        )

        # Run Bayesian Optimization
        optim_params = {
            "center": self._task_parameters.center,
            "bounds": self._task_parameters.bounds,
            "res": self._task_parameters.resolution,
            "n_samples": self._task_parameters.n_init,
            "n_iterations": self._task_parameters.n_iter,
            "Imin": Imin,
            "max_rings": self._task_parameters.max_rings,
            "rtol": self._task_parameters.rtol,
            "prior": self._task_parameters.prior,
            "seed": self._task_parameters.seed,
        }
        result = optimizer.sync_bayes_opt(**optim_params)
        if optimizer.rank == 0:
            logger.info("Optimization complete")
            logger.info(f"Elapsed time: {time.time() - start_time:.2f} s")
            params = result['params']
            residual = result['residual']
            logger.info(f"Detector Distance to Sample: {params[0]:.6f}")
            logger.info(f"Beam center: ({params[1]:.6f}, {params[2]:.6f})")
            logger.info(
                f"Rotations: \u03b8x = ({params[3]:.2e}, \u03b8y = {params[4]:.2e}, \u03b8z = {params[5]:.2e})"
            )
            logger.info(f"Final Residual: {residual:.2e}")
            fig_folder = os.path.join(
                self._task_parameters.lute_config.work_dir, "figs"
            )
            os.makedirs(fig_folder, exist_ok=True)
            plot = f"{fig_folder}/bayFAI_{exp}_r{run:0>4}.png"
            calib_detector = update_geometry(optimizer, self._task_parameters.out_file)
            powder_plot, low_q, low_res, high_q, high_res, border_q, border_res = (
                optimizer.create_interactive_powder(
                    powder=optimizer.powder,
                    detector=calib_detector,
                    distance=params[0],
                )
            )
            diagnostics_plot = optimizer.create_diagnostics_panel(
                history=result['history'],
                powder=optimizer.powder,
                detector=calib_detector,
                distance=params[0],
                Imin=Imin,
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