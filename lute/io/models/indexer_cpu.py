"""Models for CPU-based crystal lattice indexing.

Classes:
    IndexerCPUParameters(TaskParameters): Perform crystallographic indexing using
        the differential evolution-based CPU indexer.
"""

__all__ = ["IndexerCPUParameters"]
__author__ = "Claude"

import os
from typing import Any, Dict, List, Optional, Tuple

from pydantic import Field, validator

from lute.io.models.base import TaskParameters


class IndexerCPUParameters(TaskParameters):
    """Parameters for CPU-based crystal lattice indexing.

    This indexer uses quaternion-based rotation representations, lattice basis
    reduction, symmetry operations, and differential evolution optimization to
    find optimal crystal orientation matrices.
    """

    class Config(TaskParameters.Config):
        set_result: bool = True
        """Whether the Executor should mark a specified parameter as a result."""
        result_from_params: Optional[str] = None
        """Defines a result from the parameters. Set by validator."""

    # Input/Output parameters
    q_path: str = Field(
        "",
        description="Path to .npy file containing Q vectors with shape (N,3).",
    )
    unit_cell: Tuple[float, float, float, float, float, float] = Field(
        ...,
        description=(
            "Unit cell parameters: (a, b, c, alpha, beta, gamma). "
            "Lengths in Angstroms, angles in degrees."
        ),
    )
    save_prefix: str = Field(
        "",
        description=(
            "Output prefix. Saves <prefix>_U.npy (rotation matrix), "
            "<prefix>_H.npy (Miller indices), <prefix>_attempts.txt (attempt log)."
        ),
    )

    # Optimization parameters
    obj: List[str] = Field(
        [
            "mse_symm_trimmed_auto",
            "mse_symm_trimmed_auto",
            "mse_symm_trimmed_auto",
            "mse_symm_trimmed_auto",
            "mse_symm",
            "mse_symm",
        ],
        description=(
            "One objective function per try. Options: "
            "'mse_symm_trimmed_auto' (default, most robust), "
            "'mse_symm' (faster, less robust), "
            "'mse_small_n' (for small N with TSN enhancement)."
        ),
    )
    n_tries: int = Field(
        6,
        description="Number of indexing attempts with different parameters.",
        ge=1,
    )
    kappas: List[float] = Field(
        [0.93, 0.90, 0.3, 0.78, 0.54, 0.60],
        description=(
            "Trimming fractions for each try (e.g., 0.40 = use best 40% of residuals). "
            "Must have n_tries elements."
        ),
    )
    strategies: List[str] = Field(
        [
            "best1bin",
            "best1bin",
            "randtobest1bin",
            "best1bin",
            "best1bin",
            "randtobest1bin",
        ],
        description=(
            "Differential evolution strategy for each try. "
            "Options: 'best1bin', 'randtobest1bin', etc. "
            "Must have n_tries elements."
        ),
    )

    # Acceptance criteria
    delta: float = Field(
        0.25,
        description=(
            "Acceptance threshold for CrystFEL-style lattice check. "
            "Fraction of unit cell for maximum deviation from integer Miller indices."
        ),
        gt=0.0,
    )

    # Differential evolution parameters
    tol: float = Field(
        3e-2,
        description="Convergence tolerance for differential evolution.",
        gt=0.0,
    )
    maxiter: int = Field(
        1500,
        description="Maximum iterations for differential evolution.",
        ge=1,
    )
    popsize: int = Field(
        24,
        description="Population size multiplier for differential evolution.",
        ge=1,
    )
    polish: bool = Field(
        False,
        description="Whether to apply final polishing to DE solution.",
    )
    updating: str = Field(
        "deferred",
        description="DE update strategy: 'deferred' or 'immediate'.",
    )
    workers: int = Field(
        max(int(os.environ.get("SLURM_NPROCS", len(os.sched_getaffinity(0)))) - 1, 1),
        description="Number of parallel workers for DE optimization (1 = serial).",
        ge=1,
    )

    # Advanced parameters
    init_mode: str = Field(
        "random",
        description="Initialization mode for DE population.",
    )
    niche_radius_deg: float = Field(
        3.2,
        description="Angular radius for niching (degrees).",
        gt=0.0,
    )
    elite_per_niche: int = Field(
        3,
        description="Number of elite individuals per niche.",
        ge=1,
    )
    immigrants_frac: float = Field(
        0.10,
        description="Fraction of immigrants in population.",
        ge=0.0,
        le=1.0,
    )
    jitter_floor_deg: float = Field(
        1.0,
        description="Minimum jitter angle (degrees).",
        ge=0.0,
    )
    jitter_scale: float = Field(
        0.8,
        description="Jitter scaling factor.",
        ge=0.0,
    )
    callback_every: int = Field(
        250,
        description="Callback frequency for refinement during optimization.",
        ge=1,
    )

    @validator("save_prefix", always=True)
    def set_default_save_prefix(cls, save_prefix: str, values: Dict[str, Any]) -> str:
        """Set default output prefix and result."""
        if save_prefix == "":
            exp: str = values["lute_config"].experiment
            run: int = int(values["lute_config"].run)
            work_dir: str = values["lute_config"].work_dir
            save_prefix = f"{work_dir}/{exp}_r{run:04d}_indexed"

        # Set the result for the Executor
        cls.Config.result_from_params = save_prefix
        return save_prefix

    @validator("kappas")
    def validate_kappas_length(
        cls, kappas: List[float], values: Dict[str, Any]
    ) -> List[float]:
        """Ensure kappas list has n_tries elements."""
        if "n_tries" in values and len(kappas) != values["n_tries"]:
            raise ValueError(
                f"Number of kappas ({len(kappas)}) must equal n_tries ({values['n_tries']})"
            )
        return kappas

    @validator("strategies")
    def validate_strategies_length(
        cls, strategies: List[str], values: Dict[str, Any]
    ) -> List[str]:
        """Ensure strategies list has n_tries elements."""
        if "n_tries" in values and len(strategies) != values["n_tries"]:
            raise ValueError(
                f"Number of strategies ({len(strategies)}) must equal n_tries ({values['n_tries']})"
            )
        return strategies

    @validator("obj")
    def validate_obj_length(cls, obj: List[str], values: Dict[str, Any]) -> List[str]:
        """Ensure obj list has n_tries elements."""
        if "n_tries" in values and len(obj) != values["n_tries"]:
            raise ValueError(
                f"Number of obj entries ({len(obj)}) must equal n_tries ({values['n_tries']})"
            )
        return obj
