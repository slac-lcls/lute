"""Models for optimizing detector geometry using PyFAI and Bayesian optimization.

Classes:
    - OptimizePyFAIGeometryParameters(TaskParameters):
        Parameters for optimizing detector geometry using PyFAI and Bayesian optimization.
"""

__all__ = ["OptimizePyFAIGeometryParameters"]
__author__ = "Louis Conreux"

from typing import List, Dict, Optional, Union, Tuple
from pydantic import BaseModel, Field

from lute.io.models.base import TaskParameters
from lute.io.models.validators import (
    validate_smd_path,
    validate_calib_path,
    validate_output_path,
)


class OptimizePyFAIGeometryParameters(TaskParameters):
    """Parameters for optimizing detector geometry using PyFAI and Bayesian optimization.

    The Bayesian Optimization has default hyperparameters that can be overriden by the user.
    """

    class Config(TaskParameters.Config):
        set_result: bool = True
        """Whether the Executor should mark a specified parameter as a result."""

    class BayesGeomOptParameters(BaseModel):
        """Bayesian optimization hyperparameters."""

        n_samples: int = Field(
            20,
            description="Number of random starts to initialize the Bayesian optimization.",
        )

        n_iterations: int = Field(
            80,
            description="Number of iterations to run the Bayesian optimization.",
        )

        max_rings: int = Field(
            10,
            description="Maximum number of rings to consider during the score calculation.",
        )

        rtol: float = Field(
            1e-2,
            description="Relative tolerance for diffraction angle pixel masking.",
        )

        beta: float = Field(
            1.96,
            description="Exploration parameter for the Upper Confidence Bound acquisition function.",
        )

        seed: Optional[int] = Field(
            0,
            description="Seed for the random number generator for reproducibility.",
        )

    fixed: List[str] = Field(
        ["rot3"],
        description="List of parameters to be fixed during the optimization.",
    )

    center: Dict[str, float] = Field(
        {
            "dist": 0.1,
            "poni1": 0.0,
            "poni2": 0.0,
            "rot1": 0.0,
            "rot2": 0.0,
            "rot3": 0.0,
        },
        description="Center values for the parameters to be optimized.",
    )

    bounds: Dict[str, Union[float, Tuple[float, float]]] = Field(
        {
            "dist": (-0.05, 0.05),
            "poni1": (-0.01, 0.01),
            "poni2": (-0.01, 0.01),
            "rot1": (-1, 1),
            "rot2": (-1, 1),
            "rot3": (-1, 1),
        },
        description="Bounds defining the parameter search space for the Bayesian optimization. Bound values are in meters for translations and radians for rotations.",
    )

    resolution: Dict[str, float] = Field(
        {
            "dist": 0.001,
            "poni1": 0.0001,
            "poni2": 0.0001,
            "rot1": 0.1,
            "rot2": 0.1,
            "rot3": 0.1,
        },
        description="Resolution of the grid used to discretize the parameter search space. Resolution is defined in meters for translation and radians for rotations.",
    )

    detname: str = Field(
        "",
        description="Detector name. Currently supported: 'ePix10k2M', 'ePix10kaQuad', 'Jungfrau05M', 'Jungfrau1M', 'Jungfrau4M', 'Jungfrau16M'",
    )

    in_file: str = Field(
        "",
        description="Path to the input .data file containing the detector geometry info to be calibrated.",
    )

    powder: str = Field(
        "",
        description="Powder diffraction image path to be used for the calibration.",
    )

    preprocess: bool = Field(
        True,
        description="Flag to enable or disable preprocessing for the calibration.",
    )

    calibrant: str = Field(
        "",
        description="Calibrant used for the calibration supported by pyFAI: https://github.com/silx-kit/pyFAI/tree/main/src/pyFAI/resources/calibration, \n e.g. Silver Behenate 'AgBh', LaB6 'LaB6', etc.",
    )

    out_file: str = Field(
        "",
        description="Path to the output .data file that will contain the optimized detector geometry.",
        is_result=True,
    )

    bo_params: BayesGeomOptParameters = Field(
        BayesGeomOptParameters(),
        description="Bayesian optimization hyperparameters.",
    )

    _find_in_file_path = validate_calib_path("in_file")

    _find_smd_path = validate_smd_path("powder")

    _find_out_file_path = validate_output_path("out_file")
