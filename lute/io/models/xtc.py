"""Models for the Xtc converter Task.

Classes:
    Xtc1to2(TaskParameters): Parameter model for the Xtc1to2 converter tool,
    which converst lcls1 style xtc files to lcls2 style.
"""

from pydantic import Field
from lute.io.models.base import TaskParameters

class ConvertXtc1to2Parameters(TaskParameters):
    """Parameters for the xtc conversion Task."""

    exp: str = Field("amo06516", description="Experiment name", flag_type="--")
    run: str = Field("90", description="Run number", flag_type="--")
    mode: str = Field("idx", description="Mode", flag_type="--")
    detector: str = Field("pnccdFront", description="Detector", flag_type="--")
    node_id: str = Field("1", description="Node ID for the detector", flag_type="--")
    geometry: str = Field(
            "/reg/d/psdm/amo/amo06516/calib/PNCCD::CalibV1/Camp.0:pnCCD.0/geometry/38-end.data", 
            description="Geometry file", 
            flag_type="--"
            )
    reshape: str = Field("True", description="Reshape 2d to 3d flag", flag_type="--")
    eventfile: str = Field(
            "amo06516_r90_single_hits.csv",
            description="Csv file with event numbers",
            flag_type="--"
            )
