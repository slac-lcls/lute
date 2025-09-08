"""
Classes for geometry optimization tasks.

Classes:
    OptimizePyFAIGeom: optimize detector geometry using PyFAI coupled with Bayesian Optimization

"""

__all__ = ["OptimizePyFAIGeometry"]
__author__ = "Louis Conreux"

from lute.io.models.bayfai import OptimizePyFAIGeometryParameters
from lute.tasks._bayfai import BayFAIOpt
from lute.tasks._bayfai import build_detector, generate_powder, min_intensity, define_calibrant, update_geometry
from lute.tasks.task import Task
from lute.tasks.dataclasses import TaskStatus, ElogSummaryPlots
from lute.execution.logging import get_logger

import os
import logging
import panel as pn  # type: ignore
import numpy as np
import numpy.typing as npt
import time  # type: ignore

from LCLSGeom.common.geometry import get_beam_center  # type: ignore

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
        detname = self._task_parameters.det_name
        powder = generate_powder(
            powder_path=self._task_parameters.powder,
            detname=detname,
            smooth=self._task_parameters.preprocess,
        )
        detector = build_detector(
            in_file=self._task_parameters.in_file,
            shape=powder.shape,
        )
        calibrant = define_calibrant(
            calibrant=self._task_parameters.calibrant,
            exp=exp,
            run=run,
        )
        Imin = min_intensity(powder)
        optimizer = BayFAIOpt(
            exp=exp,
            run=run,
            detector=detector,
            powder=powder,
            calibrant=calibrant,
            fixed=self._task_parameters.fixed,
        )
        bayfai_hyperparams = {
            "n_samples": self._task_parameters.bo_params.n_samples,
            "n_iterations": self._task_parameters.bo_params.n_iterations,
            "Imin": Imin,
            "max_rings": self._task_parameters.bo_params.max_rings,
            "prior": self._task_parameters.bo_params.prior,
            "beta": self._task_parameters.bo_params.beta,
            "seed": self._task_parameters.bo_params.seed,
        }
        optimizer.bayfai_opt(
            center=self._task_parameters.center,
            bounds=self._task_parameters.bounds,
            res=self._task_parameters.resolution,
            **bayfai_hyperparams,
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
            calib_detector = update_geometry(optimizer, self._task_parameters.out_file)
            powder_plot, low_q, low_res, high_q, high_res, border_q, border_res = (
                optimizer.create_interactive_powder(
                    powder=optimizer.powder,
                    detector=calib_detector,
                    distance=distance,
                )
            )
            diagnostics_plot = optimizer.create_diagnostics_panel(
                powder=optimizer.powder,
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
                        f"{cx/detector.pixel_size:.3f}",
                        f"{cy/detector.pixel_size:.3f}",
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
