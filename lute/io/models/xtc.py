"""Models for the Xtc converter Task.

Classes:
    Xtc1to2(TaskParameters): Parameter model for the Xtc1to2 converter tool,
    which converst lcls1 style xtc files to lcls2 style.
"""

from typing import Optional

from pydantic import Field

from lute.io.models.base import TaskParameters


class ConvertXtc1to2Parameters(TaskParameters):
    """Parameters for the xtc conversion Task."""

    mode: str = Field(default="idx", description="Psana 1 access mode", flag_type="--")
    detector: str = Field(description="Detector", flag_type="--")
    node_id: str = Field(
        default="1", description="Node ID for the detector", flag_type="--"
    )
    resolution: str = Field(
        description="Detector channels and resolution",
        flat_type="--",
    )
    geometry: str = Field(
        description="Geometry file",
        flag_type="--",
    )
    eventfile: str = Field(
        default="",
        description="CSV file with event numbers. Otherwise will process all events.",
        flag_type="--",
    )
    verify: str = Field(
        default="False", description="Verify data - for small data only", flag_type="--"
    )
    output_file: str = Field(description="Where to write the output XTC2 file.")
    testfile: Optional[str] = Field(
        None,
        description="Path to test output HDF5 file (only if --verify=1)",
        flag_type="--",
    )
