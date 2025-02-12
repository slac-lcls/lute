"""Models for optimizing detector geometry using PyFAI and Bayesian optimization.

Classes:
    - OptimizePyFAIGeometryParameters(TaskParameters):
        Parameters for optimizing detector geometry using PyFAI and Bayesian optimization.
"""

__all__ = ["OptimizePyFAIGeometryParameters"]
__author__ = "Louis Conreux"

from typing import Any, Dict, Optional, Union, Tuple

from pydantic import BaseModel, Field, validator

from lute.io.models.base import TaskParameters
from lute.io.models.validators import validate_smd_path, validate_calib_path


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
            6,
            description="Maximum number of rings to be used for the Bayesian optimization.",
        )

        n_samples: Optional[int] = Field(
            20,
            description="Number of random starts to initialize the Bayesian optimization.",
        )

        n_iterations: Optional[int] = Field(
            80,
            description="Number of iterations to run the Bayesian optimization.",
        )

        kernel: Optional[str] = Field(
            "RBF",
            description="Kernel to be used by the Gaussian Process for the Bayesian optimization.",
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

    _find_in_file_path = validate_calib_path("in_file")

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

    preprocess: Optional[str] = Field(
        None,
        description="Preprocessing method to be used for the calibration.",
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
            exp = values["lute_config"].experiment
        return exp

    @validator("run", always=True)
    def validate_run(
        cls, run: Union[str, int], values: Dict[str, Any]
    ) -> Union[str, int]:
        if not run:
            run = values["lute_config"].run
        return run

    @validator("work_dir", always=True)
    def validate_work_dir(cls, work_dir: str, values: Dict[str, Any]) -> str:
        if not work_dir:
            work_dir = values["lute_config"].work_dir
        return work_dir

    @validator("out_file", always=True)
    def validate_out_file(cls, out_file: str, values: Dict[str, Any]) -> str:
        if not out_file:
            in_file = values["in_file"]
            run = values["run"]
            out_file = in_file.replace("0-end.data", f"{run}-end.data")
        return out_file
