"""Models for serial femtosecond crystallography indexing.

Classes:
    IndexCrystFELParameters(BaseBinaryParameters): Perform indexing of hits/peaks using
        CrystFEL's `indexamajig`.
"""

__all__ = ["IndexCrystFELParameters"]
__author__ = "Gabriel Dorlhiac"

import os
from typing import Union, List, Optional, Dict, Any

from pydantic import (
    AnyUrl,
    PositiveInt,
    PositiveFloat,
    NonNegativeInt,
    Field,
    conint,
    validator,
)

from .base import BaseBinaryParameters
from ..db import read_latest_db_entry


class IndexCrystFELParameters(BaseBinaryParameters):
    """Parameters for CrystFEL's `indexamajig`.

    There are many parameters, and many combinations. For more information on
    usage, please refer to the CrystFEL documentation, here:
    https://www.desy.de/~twhite/crystfel/manual-indexamajig.html
    """

    class Config(BaseBinaryParameters.Config):
        long_flags_use_eq: bool = True
        """Whether long command-line arguments are passed like `--long=arg`."""

    executable: str = Field(
        "/sdf/group/lcls/ds/tools/crystfel/0.10.2/bin/indexamajig",
        description="CrystFEL's indexing binary.",
        flag_type="",
    )
    # Basic options
    in_file: Optional[str] = Field(
        "", description="Path to input file.", flag_type="-", rename_param="i"
    )
    out_file: str = Field(
        "", description="Path to output file.", flag_type="-", rename_param="o"
    )
    geometry: str = Field(
        "", description="Path to geometry file.", flag_type="-", rename_param="g"
    )
    zmq_input: Optional[str] = Field(
        description="ZMQ address to receive data over. `input` and `zmq-input` are mutually exclusive",
        flag_type="--",
        rename_param="zmq-input",
    )
    zmq_subscribe: Optional[str] = Field(  # Can be used multiple times...
        description="Subscribe to ZMQ message of type `tag`",
        flag_type="--",
        rename_param="zmq-subscribe",
    )
    zmq_request: Optional[AnyUrl] = Field(
        description="Request new data over ZMQ by sending this value",
        flag_type="--",
        rename_param="zmq-request",
    )
    asapo_endpoint: Optional[str] = Field(
        description="ASAP::O endpoint. zmq-input and this are mutually exclusive.",
        flag_type="--",
        rename_param="asapo-endpoint",
    )
    asapo_token: Optional[str] = Field(
        description="ASAP::O authentication token.",
        flag_type="--",
        rename_param="asapo-token",
    )
    asapo_beamtime: Optional[str] = Field(
        description="ASAP::O beatime.",
        flag_type="--",
        rename_param="asapo-beamtime",
    )
    asapo_source: Optional[str] = Field(
        description="ASAP::O data source.",
        flag_type="--",
        rename_param="asapo-source",
    )
    asapo_group: Optional[str] = Field(
        description="ASAP::O consumer group.",
        flag_type="--",
        rename_param="asapo-group",
    )
    asapo_stream: Optional[str] = Field(
        description="ASAP::O stream.",
        flag_type="--",
        rename_param="asapo-stream",
    )
    asapo_wait_for_stream: Optional[str] = Field(
        description="If ASAP::O stream does not exist, wait for it to appear.",
        flag_type="--",
        rename_param="asapo-wait-for-stream",
    )
    data_format: Optional[str] = Field(
        description="Specify format for ZMQ or ASAP::O. `msgpack`, `hdf5` or `seedee`.",
        flag_type="--",
        rename_param="data-format",
    )
    basename: bool = Field(
        False,
        description="Remove directory parts of filenames. Acts before prefix if prefix also given.",
        flag_type="--",
    )
    prefix: Optional[str] = Field(
        description="Add a prefix to the filenames from the infile argument.",
        flag_type="--",
        rename_param="asapo-stream",
    )
    nthreads: PositiveInt = Field(
        max(int(os.environ.get("SLURM_NPROCS", len(os.sched_getaffinity(0)))) - 1, 1),
        description="Number of threads to use. See also `max_indexer_threads`.",
        flag_type="-",
        rename_param="j",
    )
    no_check_prefix: bool = Field(
        False,
        description="Don't attempt to correct the prefix if it seems incorrect.",
        flag_type="--",
        rename_param="no-check-prefix",
    )
    highres: Optional[float] = Field(
        description="Mark all pixels greater than `x` has bad.", flag_type="--"
    )
    profile: bool = Field(
        False, description="Display timing data to monitor performance.", flag_type="--"
    )
    temp_dir: Optional[str] = Field(
        description="Specify a path for the temp files folder.",
        flag_type="--",
        rename_param="temp-dir",
    )
    wait_for_file: conint(gt=-2) = Field(
        0,
        description="Wait at most `x` seconds for a file to be created. A value of -1 means wait forever.",
        flag_type="--",
        rename_param="wait-for-file",
    )
    no_image_data: bool = Field(
        False,
        description="Load only the metadata, no iamges. Can check indexability without high data requirements.",
        flag_type="--",
        rename_param="no-image-data",
    )
    # Peak-finding options
    # ....
    # Indexing options
    indexing: Optional[str] = Field(
        description="Comma-separated list of supported indexing algorithms to use. Default is to automatically detect.",
        flag_type="--",
    )
    cell_file: Optional[str] = Field(
        description="Path to a file containing unit cell information (PDB or CrystFEL format).",
        flag_type="-",
        rename_param="p",
    )
    tolerance: str = Field(
        "5,5,5,1.5",
        description=(
            "Tolerances (in percent) for unit cell comparison. "
            "Comma-separated list a,b,c,angle. Default=5,5,5,1.5"
        ),
        flag_type="--",
    )
    no_check_cell: bool = Field(
        False,
        description="Do not check cell parameters against unit cell. Replaces '-raw' method.",
        flag_type="--",
        rename_param="no-check-cell",
    )
    no_check_peaks: bool = Field(
        False,
        description="Do not verify peaks are accounted for by solution.",
        flag_type="--",
        rename_param="no-check-peaks",
    )
    multi: bool = Field(
        False, description="Enable multi-lattice indexing.", flag_type="--"
    )
    wavelength_estimate: Optional[float] = Field(
        description="Estimate for X-ray wavelength. Required for some methods.",
        flag_type="--",
        rename_param="wavelength-estimate",
    )
    camera_length_estimate: Optional[float] = Field(
        description="Estimate for camera distance. Required for some methods.",
        flag_type="--",
        rename_param="camera-length-estimate",
    )
    max_indexer_threads: Optional[PositiveInt] = Field(
        # 1,
        description="Some indexing algos can use multiple threads. In addition to image-based.",
        flag_type="--",
        rename_param="max-indexer-threads",
    )
    no_retry: bool = Field(
        False,
        description="Do not remove weak peaks and try again.",
        flag_type="--",
        rename_param="no-retry",
    )
    no_refine: bool = Field(
        False,
        description="Skip refinement step.",
        flag_type="--",
        rename_param="no-refine",
    )
    no_revalidate: bool = Field(
        False,
        description="Skip revalidation step.",
        flag_type="--",
        rename_param="no-revalidate",
    )
    # TakeTwo specific parameters
    taketwo_member_threshold: Optional[PositiveInt] = Field(
        # 20,
        description="Minimum number of vectors to consider.",
        flag_type="--",
        rename_param="taketwo-member-threshold",
    )
    taketwo_len_tolerance: Optional[PositiveFloat] = Field(
        # 0.001,
        description="TakeTwo length tolerance in Angstroms.",
        flag_type="--",
        rename_param="taketwo-len-tolerance",
    )
    taketwo_angle_tolerance: Optional[PositiveFloat] = Field(
        # 0.6,
        description="TakeTwo angle tolerance in degrees.",
        flag_type="--",
        rename_param="taketwo-angle-tolerance",
    )
    taketwo_trace_tolerance: Optional[PositiveFloat] = Field(
        # 3,
        description="Matrix trace tolerance in degrees.",
        flag_type="--",
        rename_param="taketwo-trace-tolerance",
    )
    # Felix-specific parameters
    # felix_domega
    # felix-fraction-max-visits
    # felix-max-internal-angle
    # felix-max-uniqueness
    # felix-min-completeness
    # felix-min-visits
    # felix-num-voxels
    # felix-sigma
    # felix-tthrange-max
    # felix-tthrange-min
    # XGANDALF-specific parameters
    xgandalf_sampling_pitch: Optional[NonNegativeInt] = Field(
        # 6,
        description="Density of reciprocal space sampling.",
        flag_type="--",
        rename_param="xgandalf-sampling-pitch",
    )
    xgandalf_grad_desc_iterations: Optional[NonNegativeInt] = Field(
        # 4,
        description="Number of gradient descent iterations.",
        flag_type="--",
        rename_param="xgandalf-grad-desc-iterations",
    )
    xgandalf_tolerance: Optional[PositiveFloat] = Field(
        # 0.02,
        description="Relative tolerance of lattice vectors",
        flag_type="--",
        rename_param="xgandalf-tolerance",
    )
    xgandalf_no_deviation_from_provided_cell: Optional[bool] = Field(
        description="Found unit cell must match provided.",
        flag_type="--",
        rename_param="xgandalf-no-deviation-from-provided-cell",
    )
    xgandalf_min_lattice_vector_length: Optional[PositiveFloat] = Field(
        # 30,
        description="Minimum possible lattice length.",
        flag_type="--",
        rename_param="xgandalf-min-lattice-vector-length",
    )
    xgandalf_max_lattice_vector_length: Optional[PositiveFloat] = Field(
        # 250,
        description="Minimum possible lattice length.",
        flag_type="--",
        rename_param="xgandalf-max-lattice-vector-length",
    )
    xgandalf_max_peaks: Optional[PositiveInt] = Field(
        # 250,
        description="Maximum number of peaks to use for indexing.",
        flag_type="--",
        rename_param="xgandalf-max-peaks",
    )
    xgandalf_fast_execution: bool = Field(
        False,
        description="Shortcut to set sampling-pitch=2, and grad-desc-iterations=3.",
        flag_type="--",
        rename_param="xgandalf-fast-execution",
    )
    # pinkIndexer parameters
    # ...
    # asdf_fast: bool = Field(False, description="Enable fast mode for asdf. 3x faster for 7% loss in accuracy.", flag_type="--", rename_param="asdf-fast")
    # Integration parameters
    integration: str = Field(
        "rings-nocen", description="Method for integrating reflections.", flag_type="--"
    )
    fix_profile_radius: Optional[float] = Field(
        description="Fix the profile radius (m^{-1})",
        flag_type="--",
        rename_param="fix-profile-radius",
    )
    fix_divergence: Optional[float] = Field(
        0,
        description="Fix the divergence (rad, full angle).",
        flag_type="--",
        rename_param="fix-divergence",
    )
    int_radius: str = Field(
        "4,5,7",
        description="Inner, middle, and outer radii for 3-ring integration.",
        flag_type="--",
        rename_param="int-radius",
    )
    int_diag: str = Field(
        "none",
        description="Show detailed information on integration when condition is met.",
        flag_type="--",
        rename_param="int-diag",
    )
    push_res: str = Field(
        "infinity",
        description="Integrate `x` higher than apparent resolution limit (nm-1).",
        flag_type="--",
        rename_param="push-res",
    )
    overpredict: bool = Field(
        False,
        description="Over-predict reflections. Maybe useful with post-refinement.",
        flag_type="--",
    )
    cell_parameters_only: bool = Field(
        False, description="Do not predict refletions at all", flag_type="--"
    )
    # Output parameters
    no_non_hits_in_stream: bool = Field(
        False,
        description="Exclude non-hits from the stream file.",
        flag_type="--",
        rename_param="no-non-hits-in-stream",
    )
    copy_hheader: Optional[str] = Field(
        description="Copy information from header in the image to output stream.",
        flag_type="--",
        rename_param="copy-hheader",
    )
    no_peaks_in_stream: bool = Field(
        False,
        description="Do not record peaks in stream file.",
        flag_type="--",
        rename_param="no-peaks-in-stream",
    )
    no_refls_in_stream: bool = Field(
        False,
        description="Do not record reflections in stream.",
        flag_type="--",
        rename_param="no-refls-in-stream",
    )
    serial_offset: Optional[PositiveInt] = Field(
        description="Start numbering at `x` instead of 1.",
        flag_type="--",
        rename_param="serial-offset",
    )
    harvest_file: Optional[str] = Field(
        description="Write parameters to file in JSON format.",
        flag_type="--",
        rename_param="harvest-file",
    )

    @validator("in_file", always=True)
    def validate_in_file(cls, in_file: str, values: Dict[str, Any]) -> str:
        if in_file == "":
            filename: Optional[str] = read_latest_db_entry(
                f"{values['lute_config'].work_dir}", "FindPeaksPyAlgos", "out_file"
            )
            if filename is not None:
                return filename
        return in_file

    @validator("out_file", always=True)
    def validate_in_file(cls, out_file: str, values: Dict[str, Any]) -> str:
        if out_file == "":
            expmt: str = values["lute_config"].experiment
            run: int = values["lute_config"].run
            work_dir: str = values["lute_config"].work_dir
            fname: str = f"{expmt}_r{run:04d}.stream"
            return f"{work_dir}/{fname}"
        return out_file
