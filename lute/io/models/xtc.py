"""Models for the Xtc converter Task.

Classes:
    Xtc1to2(TaskParameters): Parameter model for the Xtc1to2 converter tool,
    which converst lcls1 style xtc files to lcls2 style.
"""

from typing import Any, Dict, List, Literal, Tuple, TypedDict, Union

from pydantic import Field, validator

from lute.io.models.base import TaskParameters

Xtc1ObjectTypes = Literal[
    "psana.Detector",
    #    "psana.DataSource.env().epicsStore()"
]


class DataSpec(TypedDict):
    xtc2_attr_name: str
    object_name: str
    object_type: Xtc1ObjectTypes
    object_field_name: Union[str, Tuple[str, str]]


class ConvertXtc1to2Parameters(TaskParameters):
    """Parameters for the xtc conversion Task."""

    mode: str = Field(default="idx", description="Psana 1 access mode")
    detector: Union[str, List[str]] = Field(
        description="Detector, or comma-separated detectors to save."
    )
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
    output_file: str = Field(
        description="Where to write the output XTC2 file.", is_result=True
    )
    xtc1_access_pattern: Dict[str, List[DataSpec]] = Field(
        default={
            "$DETNAME0": [
                {
                    "xtc2_attr_name": "calib",
                    "object_name": "$DETNAME0",
                    "object_type": "psana.Detector",
                    "object_field_name": "calib",
                }
            ],
            "$DETNAME1": [
                {
                    "xtc2_attr_name": "photon_energy",
                    "object_name": "$DETNAME1",
                    "object_type": "psana.Detector",
                    "object_field_name": ("get", "ebeamPhotonEnergy"),
                }
            ],
        },
        description=(
            "Provides information for how to access the data in XTC1. The keys of "
            "the dictionary will be used for accessing the data on the XTC2 side."
        ),
    )

    @validator("xtc1_access_pattern")
    def fill_in_detname(
        cls, xtc1_access_pattern: Dict[str, List[DataSpec]], values: Dict[str, Any]
    ) -> Dict[str, List[DataSpec]]:
        detnames: List[str] = values["detector"].split(",")
        replaced_access_pattern: Dict[str, List[DataSpec]] = {}
        for key in xtc1_access_pattern:
            if "$DETNAME" in key:
                detname: str = detnames[int(key[-1])]
                replaced_access_pattern[detname] = xtc1_access_pattern[key]
                for spec in replaced_access_pattern[detname]:
                    if spec["object_name"] == key:
                        spec["object_name"] = detname

        return replaced_access_pattern
