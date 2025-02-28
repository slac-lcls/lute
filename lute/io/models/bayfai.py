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

from lute.logging import get_logger

logger = get_logger(__name__)

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
            description="Bounds defining the parameter search space for the Bayesian optimization. All bound values are in meters.",
        )

        res: float = Field(
            None,
            description="Resolution of the grid used to discretize the parameter search space. Resolution is defined in meters. If None, set to the detector pixel size.",
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
            description="Kernel to be used by the Gaussian Process for the Bayesian optimization. Currently supported: 'RBF', 'Matern'",
        )

        prior: Optional[bool] = Field(
            True,
            description="Flag to use a gaussian prior centered on the search space for the Bayesian optimization or randomly pick samples to initialize the Gaussian Process.",
        )

        af: Optional[str] = Field(
            "ucb",
            description="Acquisition function to be used by the Bayesian optimization. \n Currently supported: Upper Confidence Bound 'ucb', \n Expected Improvement 'ei', \n Probability of Improvement 'poi'",
        )

        hyperparams: Optional[Dict[str, float]] = Field(
            {
                "beta": 1.96,
                "epsilon": 0.01,
            },
            description="Hyperparameters for the acquisition function. \n beta is the exploration parameter for the Upper Confidence Bound acquisition function. \n epsilon is the exploration parameter for the Expected Improvement and Probability of Improvement acquisition functions.",
        )

        seed: Optional[int] = Field(
            None,
            description="Seed for the random number generator for reproducibility.",
        )

    _find_in_file_path = validate_calib_path("in_file")

    _find_smd_path = validate_smd_path("powder")

    det_type: str = Field(
        "",
        description="Detector type. Currently supported: 'ePix10k2M', 'ePix10kaQuad', 'Rayonix', 'Jungfrau1M', 'Jungfrau4M'",
    )

    in_file: str = Field(
        "",
        description="Path to the input .data file containing the detector geometry info to be calibrated.",
    )

    powder: str = Field(
        "",
        description="Powder diffraction image path to be used for the calibration.",
    )

    preprocess: Optional[str] = Field(
        "Diagonal",
        description="Preprocessing method to be used for the calibration. \nAvailable methods: Finite Differences Gradient Computation 'Finite', \n Diagonal Differences Gradient Computation 'Diagonal', \n Central Differences Gradient Computation 'Central', \n Laplacian of Gaussian Filtering 'Laplacian', \n No Preprocessing 'None', \n PyPCA Filtering yet to be implemented",
    )

    calibrant: str = Field(
        "",
        description="Calibrant used for the calibration supported by pyFAI: https://github.com/silx-kit/pyFAI/tree/main/src/pyFAI/resources/calibration, \n e.g. Silver Behenate 'AgBh', LaB6 'LaB6', etc.",
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

    @validator("out_file", always=True)
    def validate_out_file(cls, out_file: str, values: Dict[str, Any]) -> str:
        if not out_file:
            logger.info("No output file provided.")
            logger.info(f"{values}")
            run = values["lute_config"].run
            in_file = values["in_file"]
            out_file = in_file.replace("0-end.data", f"{run}-end.data")
        return out_file
