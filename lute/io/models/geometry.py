fd
"""Models for smalldata_tools Tasks.

Classes:
    SubmitSMDParameters(ThirdPartyParameters): Parameters to run smalldata_tools
        to produce a smalldata HDF5 file.

    AnalyzeSmallDataXSSParameters(TaskParameters): Parameter model for the
        AnalyzeSmallDataXSS Task. Used to determine spatial/temporal overlap
        based on XSS difference signal and provide basic XSS feedback.

    AnalyzeSmallDataXASParameters(TaskParameters): Parameter model for the
        AnalyzeSmallDataXAS Task. Used to determine spatial/temporal overlap
        based on XAS difference signal and provide basic XAS feedback.

    AnalyzeSmallDataXESParameters(TaskParameters): Parameter model for the
        AnalyzeSmallDataXES Task. Used to determine spatial/temporal overlap
        based on XES difference signal and provide basic XES feedback.
"""

__all__ = ["OptimizeAgBhGeometryExhaustiveParameters"]
__author__ = "Gabriel Dorlhiac"

import os
from typing import Optional, Tuple

from pydantic import Field

from lute.io.models.base import TaskParameters


class OptimizeAgBhGeometryExhaustiveParameters(TaskParameters):
    """TaskParameter model for OptimizeAgBhGeometryExhaustive Task.

    This Task does geometry optimization of detector distance and beam center
    based on a powder image produced from acquiring a run of Ag Behenate.
    """

    detname: str = Field(description="Name of the detector to optimize geometry for.")

    powder: str = Field(
        "", description="Path to the powder image, or file containing it."
    )

    mask: Optional[str] = Field(
        None, description="Path to a detector mask, or file containing it."
    )

    n_peaks: int = Field(4, description="")

    n_iterations: int = Field(
        5, description="Number of optimization iterations. Per MPI rank."
    )

    threshold: float = Field(
        1e6,
        description=(
            "Pixels in the powder image with an intensity above this threshold "
            "are set to 0."
        ),
    )

    dx: Tuple[float] = Field(
        (-6, 6, 5),
        description=(
            "Defines the search radius for beam center x position as offsets from "
            "the image center. Format: (left, right, num_steps). In units of pixels."
        ),
    )

    dy: Tuple[float] = Field(
        (-6, 6, 5),
        description=(
            "Defines the search radius for beam center y position as offsets from "
            "the image center. Format: (up, down, num_steps). In units of pixels."
        ),
    )
