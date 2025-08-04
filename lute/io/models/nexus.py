"""Models for NeXuS related Tasks.

Classes:
    ConvertSMDToNexusParameters(ThirdPartyParameters): Parameters to convert
        smalldata_tools HDF5 file to CCTBX compatible NeXuS file.
"""

__all__ = ["ConvertSMDToNexusParameters"]
__author__ = "Fred Poitevin and Gabriel Dorlhiac"

from typing import Any, Dict, Union

from pydantic import Field, validator

from lute.io.models.base import ThirdPartyParameters


class ConvertSMDToNexusParameters(ThirdPartyParameters):
    """Parameters for running a conversion of smalldata HDF5 to NEXUS HDF5."""

    class Config(ThirdPartyParameters.Config):
        """Identical to super-class Config but includes a result."""

        set_result: bool = True
        """Whether the Executor should mark a specified parameter as a result."""

        result_from_params: str = ""
        """Defines a result from the parameters. Use a validator to do so."""

    executable: str = Field("python", description="Python executable.", flag_type="")
    convertor_script: str = Field(
        description="smd to nexus conversion script from Derek Mendez.", flag_type=""
    )
    input: str = Field(
        "",
        description="Path to input smd .h5 file.",
        flag_type="--",
        rename_param="psanah5",
    )
    geom: str = Field(description="Detector geometry file (.expt).", flag_type="--")
    output: str = Field(
        "",
        description="Path to output Nexus (.h5) file.",
        flag_type="--",
        rename_param="nexus",
        is_result=True,
    )
    flip: bool = Field(False, description="Flag to flip.", flag_type="--")

    @validator("input", always=True)
    def validate_input_path(cls, input: str, values: Dict[str, Any]) -> str:
        if input == "":
            exp: str = values["lute_config"].experiment
            hutch: str = exp[:3]
            run: Union[str, int] = values["lute_config"].run
            base_path: str = f"/sdf/data/lcls/ds/{hutch}/{exp}/hdf5/cctbx"
            path: str
            path = f"{base_path}/{exp}_Run{int(run):04d}.h5"
            return path
        return input

    @validator("output", always=True)
    def validate_output_path(cls, output: str, values: Dict[str, Any]) -> str:
        if output == "":
            exp: str = values["lute_config"].experiment
            hutch: str = exp[:3]
            run: Union[str, int] = values["lute_config"].run
            base_path: str = f"/sdf/data/lcls/ds/{hutch}/{exp}/hdf5/cctbx"
            path: str
            path = f"{base_path}/{exp}_Run{int(run):04d}_nexus.h5"
            return path
        return output
