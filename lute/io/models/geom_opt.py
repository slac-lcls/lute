import os
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Union, Tuple

from pydantic import BaseModel, Field

from .base import TaskParameters, validator
from ..db import read_latest_db_entry

from PSCalib.CalibFileFinder import CalibFileFinder

class BayesGeomOptParameters(BaseModel):
    """Bayesian optimization hyperparameters."""

    bounds: Dict[str, Tuple[float, float]] = Field(
        {
            "dist": (0.0, 0.0),
            "poni1": (0.0, 0.0),
            "poni2": (0.0, 0.0),
        },
        description="Bounds defining the parameter search space for the Bayesian optimization.",
    )

    res: float = Field(
        None,
        description="Resolution of the grid used to discretize the parameter search space.",
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

    hyperparams : Optional[Dict[str, float]] = Field(
        {
            "beta": 1.96,
            "epsilon": 0.01,
        },
        description="Hyperparameters for the acquisition function.",
    )

    seed : Optional[int] = Field(
        None,
        description="Seed for the random number generator for potential reproducibility.",
    )

class OptimizePyFAIGeometryParameters(TaskParameters):
    """Parameters for optimizing detector geometry using PyFAI and Bayesian optimization.
    
    The Bayesian Optimization has default hyperparameters that can be overriden by the user.
    """
    class Config(TaskParameters.Config):
            set_result: bool = True
            """Whether the Executor should mark a specified parameter as a result."""
    
    exp : str = Field(
        "",
        description="Experiment name.",
    )

    run : int = Field(
        None,
        description="Run number.",
    )

    det_type : str = Field(
        "",
        description="Detector type. Currently supported: 'ePix10k2M', 'ePix10kaQuad', 'Rayonix', 'Rayonix2', 'Jungfrau1M', 'Jungfrau4M'",
    )

    date : str = Field(
        "",
        description="Start date of analysis",
    )

    work_dir : str = Field(
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
        description="Calibrant used for the calibration supported by pyFAI: https://github.com/silx-kit/pyFAI/tree/main/src/pyFAI/resources/calibration"
    )

    wavelength: float = Field(
        1e-10,
        description="Wavelength of the X-ray beam in meters.",
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
        if exp == "":
            exp: str = values["lute_config"].experiment
        return exp
    
    @validator("run", always=True)
    def validate_run(cls, run: int, values: Dict[str, Any]) -> Union[str, int]:
        if run is None:
            run: Union[str, int] = values["lute_config"].run
        return run
    
    @validator("date", always=True)
    def validate_date(cls, date: str, values: Dict[str, Any]) -> str:
        if date == "":
            date: str = values["lute_config"].date
        return date
    
    @validator("work_dir", always=True)
    def validate_work_dir(cls, work_dir: str, values: Dict[str, Any]) -> str:
        if work_dir == "":
            work_dir: str = values["lute_config"].work_dir
        return work_dir
    
    @validator("in_file", always=True)
    def validate_in_file(cls, in_file: str) -> str:
        if in_file == "":
            exp = cls.exp
            run = cls.run
            det_type = cls.det_type
            cdir = f'/sdf/data/lcls/ds/{exp[:3]}/{exp}/calib'
            src = 'MfxEndstation.0:Epix10ka2M.0'
            type = 'geometry'
            cff = CalibFileFinder(cdir)
            in_file = cff.findCalibFile(src, type, run)
        return in_file

    @validator("powder", always=True)
    def validate_powder(cls, powder: str, values: Dict[str, Any]) -> str:
        if powder == "":
            powder: str = read_latest_db_entry(
                f"{values['lute_config'].work_dir}/powder", "ComputePowder", "out_file"
            )
        return powder
    
    @validator("out_file", always=True)
    def validate_out_file(cls, out_file: str, run: Union[str, int], in_file: str) -> str:
        if out_file == "":
            out_file: str = in_file.replace("0-end.data", f"{run}-end.data")
        return out_file