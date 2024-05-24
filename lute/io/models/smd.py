"""Models for smalldata_tools Tasks.

Classes:
    SubmitSMDParameters(ThirdPartyParameters): Parameters to run smalldata_tools
        to produce a smalldata HDF5 file.

    FindOverlapXSSParameters(TaskParameters): Parameter model for the
        FindOverlapXSS Task. Used to determine spatial/temporal overlap based on
        XSS difference signal.
"""

__all__ = ["SubmitSMDParameters", "FindOverlapXSSParameters"]
__author__ = "Gabriel Dorlhiac"

import os
from typing import Union, List, Optional, Dict, Any

from pydantic import (
    BaseModel,
    HttpUrl,
    PositiveInt,
    NonNegativeInt,
    Field,
    root_validator,
    validator,
)

from .base import TaskParameters, ThirdPartyParameters, TemplateConfig


class SubmitSMDParameters(ThirdPartyParameters):
    """Parameters for running smalldata to produce reduced HDF5 files."""

    class Config(ThirdPartyParameters.Config):
        """Identical to super-class Config but includes a result."""

        set_result: bool = True
        """Whether the Executor should mark a specified parameter as a result."""

        result_from_params: str = ""
        """Defines a result from the parameters. Use a validator to do so."""

    executable: str = Field("mpirun", description="MPI executable.", flag_type="")
    np: PositiveInt = Field(
        max(int(os.environ.get("SLURM_NPROCS", len(os.sched_getaffinity(0)))) - 1, 1),
        description="Number of processes",
        flag_type="-",
    )
    p_arg1: str = Field(
        "python", description="Executable to run with mpi (i.e. python).", flag_type=""
    )
    u: str = Field(
        "", description="Python option for unbuffered output.", flag_type="-"
    )
    m: str = Field(
        "mpi4py.run",
        description="Python option to execute a module's contents as __main__ module.",
        flag_type="-",
    )
    producer: str = Field(
        "", description="Path to the SmallData producer Python script.", flag_type=""
    )
    run: str = Field(
        os.environ.get("RUN_NUM", ""), description="DAQ Run Number.", flag_type="--"
    )
    experiment: str = Field(
        os.environ.get("EXPERIMENT", ""),
        description="LCLS Experiment Number.",
        flag_type="--",
    )
    stn: NonNegativeInt = Field(0, description="Hutch endstation.", flag_type="--")
    nevents: int = Field(
        int(1e9), description="Number of events to process.", flag_type="--"
    )
    directory: Optional[str] = Field(
        None,
        description="Optional output directory. If None, will be in ${EXP_FOLDER}/hdf5/smalldata.",
        flag_type="--",
    )
    ## Need mechanism to set result_from_param=True ...
    gather_interval: PositiveInt = Field(
        25, description="Number of events to collect at a time.", flag_type="--"
    )
    norecorder: bool = Field(
        False, description="Whether to ignore recorder streams.", flag_type="--"
    )
    url: HttpUrl = Field(
        "https://pswww.slac.stanford.edu/ws-auth/lgbk",
        description="Base URL for eLog posting.",
        flag_type="--",
    )
    epicsAll: bool = Field(
        False,
        description="Whether to store all EPICS PVs. Use with care.",
        flag_type="--",
    )
    full: bool = Field(
        False,
        description="Whether to store all data. Use with EXTRA care.",
        flag_type="--",
    )
    fullSum: bool = Field(
        False,
        description="Whether to store sums for all area detector images.",
        flag_type="--",
    )
    default: bool = Field(
        False,
        description="Whether to store only the default minimal set of data.",
        flag_type="--",
    )
    image: bool = Field(
        False,
        description="Whether to save everything as images. Use with care.",
        flag_type="--",
    )
    tiff: bool = Field(
        False,
        description="Whether to save all images as a single TIFF. Use with EXTRA care.",
        flag_type="--",
    )
    centerpix: bool = Field(
        False,
        description="Whether to mask center pixels for Epix10k2M detectors.",
        flag_type="--",
    )
    postRuntable: bool = Field(
        False,
        description="Whether to post run tables. Also used as a trigger for summary jobs.",
        flag_type="--",
    )
    wait: bool = Field(
        False, description="Whether to wait for a file to appear.", flag_type="--"
    )
    xtcav: bool = Field(
        False,
        description="Whether to add XTCAV processing to the HDF5 generation.",
        flag_type="--",
    )
    noarch: bool = Field(
        False, description="Whether to not use archiver data.", flag_type="--"
    )

    lute_template_cfg: TemplateConfig = TemplateConfig(template_name="", output_path="")

    @validator("producer", always=True)
    def validate_producer_path(cls, producer: str) -> str:
        return producer

    @validator("lute_template_cfg", always=True)
    def use_producer(
        cls, lute_template_cfg: TemplateConfig, values: Dict[str, Any]
    ) -> TemplateConfig:
        if not lute_template_cfg.output_path:
            lute_template_cfg.output_path = values["producer"]
        return lute_template_cfg

    @root_validator(pre=False)
    def define_result(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        exp: str = values["lute_config"].experiment
        hutch: str = exp[:3]
        run: int = int(values["lute_config"].run)
        directory: Optional[str] = values["directory"]
        if directory is None:
            directory = f"/sdf/data/lcls/ds/{hutch}/{exp}/hdf5/smalldata"
        fname: str = f"{exp}_Run{run:04d}.h5"

        cls.Config.result_from_params = f"{directory}/{fname}"
        return values

    # detnames: TemplateParameters = TemplateParameters({})
    # epicsPV: TemplateParameters = TemplateParameters({})
    # ttCalib: TemplateParameters = TemplateParameters({})
    # aioParams: TemplateParameters = TemplateParameters({})
    # getROIs: TemplateParameters = TemplateParameters({})
    # getAzIntParams: TemplateParameters = TemplateParameters({})
    # getAzIntPyFAIParams: TemplateParameters = TemplateParameters({})
    # getPhotonsParams: TemplateParameters = TemplateParameters({})
    # getDropletParams: TemplateParameters = TemplateParameters({})
    # getDroplet2Photons: TemplateParameters = TemplateParameters({})
    # getSvdParams: TemplateParameters = TemplateParameters({})
    # getAutocorrParams: TemplateParameters = TemplateParameters({})


class FindOverlapXSSParameters(TaskParameters):
    """TaskParameter model for FindOverlapXSS Task.

    This Task determines spatial or temporal overlap between an optical pulse
    and the FEL pulse based on difference scattering (XSS) signal. This Task
    uses SmallData HDF5 files as a source.
    """

    class ExpConfig(BaseModel):
        det_name: str
        ipm_var: str
        scan_var: Union[str, List[str]]

    class Thresholds(BaseModel):
        min_Iscat: Union[int, float]
        min_ipm: Union[int, float]

    class AnalysisFlags(BaseModel):
        use_pyfai: bool = True
        use_asymls: bool = False

    exp_config: ExpConfig
    thresholds: Thresholds
    analysis_flags: AnalysisFlags
