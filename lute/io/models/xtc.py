"""Models for the Xtc converter Task.

Classes:
    Xtc1to2(TaskParameters): Parameter model for the Xtc1to2 converter tool,
    which converst lcls1 style xtc files to lcls2 style.
"""

from typing import Any, Dict, Literal, Optional, Tuple, TypedDict, Union

from pydantic import Field, validator

from lute.io.models.base import TaskParameters

Xtc1ObjectTypes = Literal[
    "psana.Detector",
    #    "psana.DataSource.env().epicsStore()"
]


class DataSpec(TypedDict):
    object_name: str
    object_type: Xtc1ObjectTypes
    object_field_name: Union[str, Tuple[str, str]]


class ConvertXtc1to2Parameters(TaskParameters):
    """Parameters for the xtc conversion Task."""

    mode: str = Field(default="idx", description="Psana 1 access mode")
    detector: str = Field(description="Detector")
    node_id: str = Field(default="1", description="Node ID for the detector")
    resolution: str = Field(
        description="Detector channels and resolution",
    )
    geometry: str = Field(
        description="Geometry file",
    )
    eventfile: str = Field(
        default="",
        description="CSV file with event numbers. Otherwise will process all events.",
    )
    verify: str = Field(
        default="False", description="Verify data - for small data only"
    )
    output_file: str = Field(
        description="Where to write the output XTC2 file.", is_result=True
    )
    testfile: Optional[str] = Field(
        None,
        description="Path to test output HDF5 file (only if --verify=1)",
    )
    xtc1_access_pattern: Dict[str, DataSpec] = Field(
        default={
            "calib": {
                "object_name": "$DETNAME",
                "object_type": "psana.Detector",
                "object_field_name": "calib",
            },
            "photon_energy": {
                "object_name": "EBeam",
                "object_type": "psana.Detector",
                "object_field_name": ("get", "ebeamPhotonEnergy"),
            },
        },
        description=(
            "Provides information for how to access the data in XTC1. The keys of "
            "the dictionary will be used for accessing the data on the XTC2 side."
        ),
    )

    @validator("xtc1_access_pattern")
    def fill_in_detname(
        cls, xtc1_access_pattern: Dict[str, DataSpec], values: Dict[str, Any]
    ) -> Dict[str, DataSpec]:
        for key in xtc1_access_pattern:
            if xtc1_access_pattern[key]["object_name"] == "$DETNAME":
                xtc1_access_pattern[key]["object_name"] = values["detector"]
        return xtc1_access_pattern
