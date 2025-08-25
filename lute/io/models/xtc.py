"""Models for the Xtc converter Task.

Classes:
    Xtc1to2(TaskParameters): Parameter model for the Xtc1to2 converter tool,
    which converst lcls1 style xtc files to lcls2 style.
"""

from pydantic import Field

class XtcParameters(TaskParameters):
    """Parameters for the xtc conversion Task."""

    exp: str = Field("amo06516", description="Experiment name", flag_type="--")
    run: int = Field(90, description="Run number", flag_type="--")
    mode: str = Field("idx", description="Mode", flag_type="--")
    detector: str = Field("pnccdFront", description="Detector", flag_type="--")
    geometry: str = Field(
            "/reg/d/psdm/amo/amo06516/calib/PNCCD::CalibV1/Camp.0:pnCCD.0/geometry/38-end.data", 
            description="Geometry file", 
            flag_type="--")
