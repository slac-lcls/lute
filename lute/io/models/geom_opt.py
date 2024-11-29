"""Models for optimizing detector geometry using PyFAI and Bayesian optimization.

Classes:
    - OptimizePyFAIGeometryParameters(TaskParameters):
        Parameters for optimizing detector geometry using PyFAI and Bayesian optimization.
"""

__all__ = ["OptimizePyFAIGeometryParameters"]
__author__ = "Louis Conreux"

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union, Tuple

from pydantic import BaseModel, Field, validator

from lute.io.models.base import TaskParameters
from lute.io.models.validators import validate_smd_path

import psana
from PSCalib.CalibFileFinder import CalibFileFinder


class OptimizePyFAIGeometryParameters(TaskParameters):
    """Parameters for optimizing detector geometry using PyFAI and Bayesian optimization.

    The Bayesian Optimization has default hyperparameters that can be overriden by the user.
    """

    class Config(TaskParameters.Config):
        set_result: bool = True
        """Whether the Executor should mark a specified parameter as a result."""

    class BayesGeomOptParameters(BaseModel):
        """Bayesian optimization hyperparameters."""

        bounds: Dict[str, Union[float, Tuple[float, float]]] = Field(
            {
                "dist": (0.02, 0.6),
                "poni1": (-0.01, 0.01),
                "poni2": (-0.01, 0.01),
            },
            description="Bounds defining the parameter search space for the Bayesian optimization.",
        )

        res: float = Field(
            None,
            description="Resolution of the grid used to discretize the parameter search space.",
        )

        max_rings: int = Field(
            5,
            description="Maximum number of rings to be used for the Bayesian optimization.",
        )

        n_samples: Optional[int] = Field(
            50,
            description="Number of random starts to initialize the Bayesian optimization.",
        )

        n_iterations: Optional[int] = Field(
            50,
            description="Number of iterations to run the Bayesian optimization.",
        )

        prior: Optional[bool] = Field(
            True,
            description="Whether to use a gaussian prior centered on the search space for the Bayesian optimization or randomly pick samples.",
        )

        af: Optional[str] = Field(
            "ucb",
            description="Acquisition function to be used by the Bayesian optimization.",
        )

        hyperparams: Optional[Dict[str, float]] = Field(
            {
                "beta": 1.96,
                "epsilon": 0.01,
            },
            description="Hyperparameters for the acquisition function.",
        )

        seed: Optional[int] = Field(
            None,
            description="Seed for the random number generator for potential reproducibility.",
        )

    _find_smd_path = validate_smd_path("powder")

    exp: str = Field(
        "",
        description="Experiment name.",
    )

    run: Union[str, int] = Field(
        None,
        description="Run number.",
    )

    det_type: str = Field(
        "",
        description="Detector type. Currently supported: 'ePix10k2M', 'ePix10kaQuad', 'Rayonix', 'Jungfrau1M', 'Jungfrau4M'",
    )

    date: str = Field(
        "",
        description="Start date of analysis",
    )

    work_dir: str = Field(
        "",
        description="Main working directory for LUTE.",
    )

    in_file: str = Field(
        "",
        description="Path to the input .data file containing the detector geometry info to be calibrated.",
    )

    powder: str = Field(
        "",
        description="Powder diffraction pattern to be used for the calibration.",
    )

    calibrant: str = Field(
        "",
        description="Calibrant used for the calibration supported by pyFAI: https://github.com/silx-kit/pyFAI/tree/main/src/pyFAI/resources/calibration",
    )

    out_file: str = Field(
        "",
        description="Path to the output .data file containing the optimized detector geometry.",
        is_result=True,
    )

    bo_params: BayesGeomOptParameters = Field(
        BayesGeomOptParameters(),
        description="Bayesian optimization parameters containing bounds and resolution for defining space search and hyperparameters.",
    )

    @validator("exp", always=True)
    def validate_exp(cls, exp: str, values: Dict[str, Any]) -> str:
        if not exp:
            print("exp")
            print(values.keys())
            exp: str = values["lute_config"].experiment
        return exp

    @validator("run", always=True)
    def validate_run(
        cls, run: Union[str, int], values: Dict[str, Any]
    ) -> Union[str, int]:
        if not run:
            print("run")
            print(values.keys())
            run: Union[str, int] = values["lute_config"].run
        return run

    @validator("date", always=True)
    def validate_date(cls, date: str, values: Dict[str, Any]) -> str:
        if not date:
            print("date")
            print(values.keys())
            date: str = values["lute_config"].date
        return date

    @validator("work_dir", always=True)
    def validate_work_dir(cls, work_dir: str, values: Dict[str, Any]) -> str:
        if not work_dir:
            print("work_dir")
            print(values.keys())
            work_dir: str = values["lute_config"].work_dir
        return work_dir

    @validator("in_file", always=True)
    def validate_in_file(cls, in_file: str, values: Dict[str, Any]) -> str:
        if not in_file:
            print("in_file")
            print(values.keys())
            exp = values["exp"]
            run = values["run"]
            cdir = f"/sdf/data/lcls/ds/{exp[:3]}/{exp}/calib"
            dsname = f"exp={exp}:run={run}"
            ds = psana.DataSource(dsname)
            det = psana.Detector(values["det_type"], ds.env())
            src = det.name
            type = "geometry"
            cff = CalibFileFinder(cdir)
            in_file: str = cff.findCalibFile(src, type, run)
            print('in_file', in_file)
        return in_file

    @validator("out_file", always=True)
    def validate_out_file(cls, out_file: str, values: Dict[str, Any]) -> str:
        if not out_file:
            print("out_file")
            print(values.keys())
            in_file = values["in_file"]
            run = values["run"]
            out_file: str = in_file.replace("0-end.data", f"{run}-end.data")
        return out_file
