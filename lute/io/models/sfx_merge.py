"""Models for merging reflections in serial femtosecond crystallography.

Classes:
    MergePartialatorParameters(ThirdPartyParameters): Perform merging using
        CrystFEL's `partialator`.

    CompareHKLParameters(ThirdPartyParameters): Calculate figures of merit using
        CrystFEL's `compare_hkl`.

    ManipulateHKLParameters(ThirdPartyParameters): Perform transformations on
        lists of reflections using CrystFEL's `get_hkl`.
"""

__all__ = [
    "MergePartialatorParameters",
    "MergeCCTBXXFELParameters",
    "CompareHKLParameters",
    "ManipulateHKLParameters",
]
__author__ = "Gabriel Dorlhiac"

import os
from typing import Union, Optional, Dict, Any

from pydantic import Field, validator, BaseModel
from pydantic.schema import model_schema, default_ref_template  # Update `schema` method

from lute.io.db import read_latest_db_entry
from lute.io.models.base import ThirdPartyParameters, TemplateConfig
from lute.io.models.validators import template_parameter_validator


class MergePartialatorParameters(ThirdPartyParameters):
    """Parameters for CrystFEL's `partialator`.

    There are many parameters, and many combinations. For more information on
    usage, please refer to the CrystFEL documentation, here:
    https://www.desy.de/~twhite/crystfel/manual-partialator.html
    """

    class Config(ThirdPartyParameters.Config):
        long_flags_use_eq: bool = True
        """Whether long command-line arguments are passed like `--long=arg`."""

        set_result: bool = True
        """Whether the Executor should mark a specified parameter as a result."""

    executable: str = Field(
        "/sdf/group/lcls/ds/tools/crystfel/0.10.2/bin/partialator",
        description="CrystFEL's Partialator binary.",
        flag_type="",
    )
    in_file: Optional[str] = Field(
        "", description="Path to input stream.", flag_type="-", rename_param="i"
    )
    out_file: str = Field(
        "",
        description="Path to output file.",
        flag_type="-",
        rename_param="o",
        is_result=True,
    )
    symmetry: str = Field(description="Point group symmetry.", flag_type="--")
    niter: Optional[int] = Field(
        description="Number of cycles of scaling and post-refinement.",
        flag_type="-",
        rename_param="n",
    )
    no_scale: Optional[bool] = Field(
        description="Disable scaling.", flag_type="--", rename_param="no-scale"
    )
    no_Bscale: Optional[bool] = Field(
        description="Disable Debye-Waller part of scaling.",
        flag_type="--",
        rename_param="no-Bscale",
    )
    no_pr: Optional[bool] = Field(
        description="Disable orientation model.", flag_type="--", rename_param="no-pr"
    )
    no_deltacchalf: Optional[bool] = Field(
        description="Disable rejection based on deltaCC1/2.",
        flag_type="--",
        rename_param="no-deltacchalf",
    )
    model: str = Field(
        "unity",
        description="Partiality model. Options: xsphere, unity, offset, ggpm.",
        flag_type="--",
    )
    nthreads: int = Field(
        max(int(os.environ.get("SLURM_NPROCS", len(os.sched_getaffinity(0)))) - 1, 1),
        description="Number of parallel analyses.",
        flag_type="-",
        rename_param="j",
    )
    polarisation: Optional[str] = Field(
        description="Specification of incident polarisation. Refer to CrystFEL docs for more info.",
        flag_type="--",
    )
    no_polarisation: Optional[bool] = Field(
        description="Synonym for --polarisation=none",
        flag_type="--",
        rename_param="no-polarisation",
    )
    max_adu: Optional[float] = Field(
        description="Maximum intensity of reflection to include.",
        flag_type="--",
        rename_param="max-adu",
    )
    min_res: Optional[float] = Field(
        description="Only include crystals diffracting to a minimum resolution.",
        flag_type="--",
        rename_param="min-res",
    )
    min_measurements: int = Field(
        2,
        description="Include a reflection only if it appears a minimum number of times.",
        flag_type="--",
        rename_param="min-measurements",
    )
    push_res: Optional[float] = Field(
        description="Merge reflections up to higher than the apparent resolution limit.",
        flag_type="--",
        rename_param="push-res",
    )
    start_after: int = Field(
        0,
        description="Ignore the first n crystals.",
        flag_type="--",
        rename_param="start-after",
    )
    stop_after: int = Field(
        0,
        description="Stop after processing n crystals. 0 means process all.",
        flag_type="--",
        rename_param="stop-after",
    )
    no_free: Optional[bool] = Field(
        description="Disable cross-validation. Testing ONLY.",
        flag_type="--",
        rename_param="no-free",
    )
    custom_split: Optional[str] = Field(
        description="Read a set of filenames, event and dataset IDs from a filename.",
        flag_type="--",
        rename_param="custom-split",
    )
    max_rel_B: float = Field(
        100,
        description="Reject crystals if |relB| > n sq Angstroms.",
        flag_type="--",
        rename_param="max-rel-B",
    )
    output_every_cycle: bool = Field(
        False,
        description="Write per-crystal params after every refinement cycle.",
        flag_type="--",
        rename_param="output-every-cycle",
    )
    no_logs: bool = Field(
        False,
        description="Do not write logs needed for plots, maps and graphs.",
        flag_type="--",
        rename_param="no-logs",
    )
    set_symmetry: Optional[str] = Field(
        description="Set the apparent symmetry of the crystals to a point group.",
        flag_type="-",
        rename_param="w",
    )
    operator: Optional[str] = Field(
        description="Specify an ambiguity operator. E.g. k,h,-l.", flag_type="--"
    )
    force_bandwidth: Optional[float] = Field(
        description="Set X-ray bandwidth. As percent, e.g. 0.0013 (0.13%).",
        flag_type="--",
        rename_param="force-bandwidth",
    )
    force_radius: Optional[float] = Field(
        description="Set the initial profile radius (nm-1).",
        flag_type="--",
        rename_param="force-radius",
    )
    force_lambda: Optional[float] = Field(
        description="Set the wavelength. In Angstroms.",
        flag_type="--",
        rename_param="force-lambda",
    )
    harvest_file: Optional[str] = Field(
        description="Write parameters to file in JSON format.",
        flag_type="--",
        rename_param="harvest-file",
    )

    @validator("in_file", always=True)
    def validate_in_file(cls, in_file: str, values: Dict[str, Any]) -> str:
        if in_file == "":
            stream_file: Optional[str] = read_latest_db_entry(
                f"{values['lute_config'].work_dir}",
                "ConcatenateStreamFiles",
                "out_file",
            )
            if stream_file:
                return stream_file
        return in_file

    @validator("out_file", always=True)
    def validate_out_file(cls, out_file: str, values: Dict[str, Any]) -> str:
        if out_file == "":
            in_file: str = values["in_file"]
            if in_file:
                tag: str = in_file.split(".")[0]
                return f"{tag}.hkl"
            else:
                return "partialator.hkl"
        return out_file


class MergeCCTBXXFELParameters(ThirdPartyParameters):
    """Parameters for merging with cctbx.xfel."""

    class Config(ThirdPartyParameters.Config):
        set_result: bool = False
        """Whether the Executor should mark a specified parameter as a result."""

    class PhilParameters(BaseModel):
        """Template parameters for CCTBX phil file."""

        class Config(BaseModel.Config):
            extra: str = "allow"

        # Generic input settings: input_
        input_path: str = Field(
            "",
            description="Input file(s).",
        )
        input_experiments_suffix: str = Field(
            "_integrated.expt", description="Suffix appened to experiments."
        )
        input_reflections_suffix: str = Field(
            "_integrated.refl", description="Suffix appened to experiments."
        )
        input_parallel_file_load_method: str = Field(
            "uniform",  # *uniform node_memory
            description="Parallel file loading method.",
        )

        # Filtering settings: filter_
        filter_algorithm: str = Field(
            "unit_cell",  # n_obs reindex resolution unit_cell report
            description="",
        )
        filter_unit_cell_algorithm: str = Field(
            "cluster", description=""  # range *value cluster
        )
        filter_unit_cell_cluster_covariance_file: str = Field(
            "",  # $MODULES/$COV?
            description="",
        )
        filter_unit_cell_cluster_covariance_component: int = Field(
            0,
            description="",
        )
        filter_unit_cell_cluster_covariance_mahalanobis: float = Field(
            5.0,
            description="",
        )
        filter_outlier_min_corr: float = Field(
            -1.0,
            description="",
        )

        # Selection settings: select_
        select_algorithm: str = Field(
            "significance_filter",
            description="",
        )
        select_significance_filter_sigma: float = Field(
            0.1,
            description="",
        )

        # Scaling settings: scaling_
        scaling_model: str = Field(
            "",  # $MODULES/$COV?
            description="",
        )
        scaling_resolution_scalar: float = Field(
            0.993420862158964,
            description="",
        )

        # Post-refinement: postrefinement_
        postrefinement_enable: bool = Field(
            True, description="Enable post-refinement processing?"
        )
        postrefinement_algorithm: str = Field("rs", description="")

        # Merging: merging_
        merging_d_min: int = Field(
            3,  # What's a good default?
            description="",
        )
        merging_merge_anomalous: bool = Field(False, description="")
        merging_set_average_unit_cell: bool = Field(True, description="")
        merging_error_model: str = Field(
            "ev11", description=""  # ha14 *ev11 mm24 errors_from_sample_residuals
        )

        # Statistics: statistics_
        statistics_n_bins: int = Field(20, description="")
        statistics_report_ML: bool = Field(True, description="")
        statistics_cciso_mtz_file: str = Field(
            "",  # $H5_SIM_PATH/ground_truth.mtz
            description="",
        )
        statistics_cciso_mtz_column_F: str = Field("F", description="")

        # Output settings: output_
        output_prefix: str = Field("", description="")
        output_output_dir: str = Field(
            "",
            description="",
        )
        output_tmp_dir: str = Field(
            "",
            description="",
        )
        output_do_timing: bool = Field(True, description="")
        output_log_level: int = Field(0, description="")
        output_save_experiments_and_reflections: bool = Field(True, description="")

        # Parallel processing settings: parallel_
        parallel_a2a: int = Field(1, description="")

    _set_phil_template_parameters = template_parameter_validator("phil_parameters")

    executable: str = Field(
        "/sdf/group/lcls/ds/tools/cctbx/conda_base/bin/mpirun",
        description="MPI executable.",
        flag_type="",
    )
    cctbx_executable: str = Field(
        "/sdf/group/lcls/ds/tools/cctbx/build/bin/cctbx.xfel.merge",
        description="CCTBX merge program.",
        flag_type="",
    )
    phil_file: str = Field(
        "",
        description="Location of the input settings ('phil') file.",
        flag_type="",
    )
    phil_parameters: Optional[PhilParameters] = Field(
        None,
        description="Optional template parameters to fill in a CCTBX phil file.",
        flag_type="",  # Does nothing since always None by time it's seen by Task
    )
    lute_template_cfg: TemplateConfig = Field(
        TemplateConfig(
            template_name="cctbx_merge.phil",
            output_path="",
        ),
        description="Template information for the cctbx_merge file.",
    )

    @validator("phil_file", always=True)
    def set_default_phil_path(cls, phil_file: str, values: Dict[str, Any]) -> str:
        if phil_file == "":
            return f"{values['lute_config'].work_dir}/cctbx_merge.phil"
        return phil_file

    @validator("lute_template_cfg", always=True)
    def set_phil_template_path(
        cls, lute_template_cfg: TemplateConfig, values: Dict[str, Any]
    ) -> TemplateConfig:
        if lute_template_cfg.output_path == "":
            lute_template_cfg.output_path = values["phil_file"]
        return lute_template_cfg


class CompareHKLParameters(ThirdPartyParameters):
    """Parameters for CrystFEL's `compare_hkl` for calculating figures of merit.

    There are many parameters, and many combinations. For more information on
    usage, please refer to the CrystFEL documentation, here:
    https://www.desy.de/~twhite/crystfel/manual-partialator.html
    """

    class Config(ThirdPartyParameters.Config):
        long_flags_use_eq: bool = True
        """Whether long command-line arguments are passed like `--long=arg`."""

        set_result: bool = True
        """Whether the Executor should mark a specified parameter as a result."""

    executable: str = Field(
        "/sdf/group/lcls/ds/tools/crystfel/0.10.2/bin/compare_hkl",
        description="CrystFEL's reflection comparison binary.",
        flag_type="",
    )
    in_files: Optional[str] = Field(
        "",
        description="Path to input HKLs. Space-separated list of 2. Use output of partialator e.g.",
        flag_type="",
    )
    ## Need mechanism to set is_result=True ...
    symmetry: str = Field("", description="Point group symmetry.", flag_type="--")
    cell_file: str = Field(
        "",
        description="Path to a file containing unit cell information (PDB or CrystFEL format).",
        flag_type="-",
        rename_param="p",
    )
    fom: str = Field(
        "Rsplit", description="Specify figure of merit to calculate.", flag_type="--"
    )
    nshells: int = Field(10, description="Use n resolution shells.", flag_type="--")
    # NEED A NEW CASE FOR THIS -> Boolean flag, no arg, one hyphen...
    # fix_unity: bool = Field(
    #    False,
    #    description="Fix scale factors to unity.",
    #    flag_type="-",
    #    rename_param="u",
    # )
    shell_file: str = Field(
        "",
        description="Write the statistics in resolution shells to a file.",
        flag_type="--",
        rename_param="shell-file",
        is_result=True,
    )
    ignore_negs: bool = Field(
        False,
        description="Ignore reflections with negative reflections.",
        flag_type="--",
        rename_param="ignore-negs",
    )
    zero_negs: bool = Field(
        False,
        description="Set negative intensities to 0.",
        flag_type="--",
        rename_param="zero-negs",
    )
    sigma_cutoff: Optional[Union[float, int, str]] = Field(
        # "-infinity",
        description="Discard reflections with I/sigma(I) < n. -infinity means no cutoff.",
        flag_type="--",
        rename_param="sigma-cutoff",
    )
    rmin: Optional[float] = Field(
        description="Low resolution cutoff of 1/d (m-1). Use this or --lowres NOT both.",
        flag_type="--",
    )
    lowres: Optional[float] = Field(
        descirption="Low resolution cutoff in Angstroms. Use this or --rmin NOT both.",
        flag_type="--",
    )
    rmax: Optional[float] = Field(
        description="High resolution cutoff in 1/d (m-1). Use this or --highres NOT both.",
        flag_type="--",
    )
    highres: Optional[float] = Field(
        description="High resolution cutoff in Angstroms. Use this or --rmax NOT both.",
        flag_type="--",
    )

    @validator("in_files", always=True)
    def validate_in_files(cls, in_files: str, values: Dict[str, Any]) -> str:
        if in_files == "":
            partialator_file: Optional[str] = read_latest_db_entry(
                f"{values['lute_config'].work_dir}", "MergePartialator", "out_file"
            )
            if partialator_file:
                hkls: str = f"{partialator_file}1 {partialator_file}2"
                return hkls
        return in_files

    @validator("cell_file", always=True)
    def validate_cell_file(cls, cell_file: str, values: Dict[str, Any]) -> str:
        if cell_file == "":
            idx_cell_file: Optional[str] = read_latest_db_entry(
                f"{values['lute_config'].work_dir}",
                "IndexCrystFEL",
                "cell_file",
                valid_only=False,
            )
            if idx_cell_file:
                return idx_cell_file
        return cell_file

    @validator("symmetry", always=True)
    def validate_symmetry(cls, symmetry: str, values: Dict[str, Any]) -> str:
        if symmetry == "":
            partialator_sym: Optional[str] = read_latest_db_entry(
                f"{values['lute_config'].work_dir}", "MergePartialator", "symmetry"
            )
            if partialator_sym:
                return partialator_sym
        return symmetry

    @validator("shell_file", always=True)
    def validate_shell_file(cls, shell_file: str, values: Dict[str, Any]) -> str:
        if shell_file == "":
            partialator_file: Optional[str] = read_latest_db_entry(
                f"{values['lute_config'].work_dir}", "MergePartialator", "out_file"
            )
            if partialator_file:
                shells_out: str = partialator_file.split(".")[0]
                shells_out = f"{shells_out}_{values['fom']}_n{values['nshells']}.dat"
                return shells_out
        return shell_file


class ManipulateHKLParameters(ThirdPartyParameters):
    """Parameters for CrystFEL's `get_hkl` for manipulating lists of reflections.

    This Task is predominantly used internally to convert `hkl` to `mtz` files.
    Note that performing multiple manipulations is undefined behaviour. Run
    the Task with multiple configurations in explicit separate steps. For more
    information on usage, please refer to the CrystFEL documentation, here:
    https://www.desy.de/~twhite/crystfel/manual-partialator.html
    """

    class Config(ThirdPartyParameters.Config):
        long_flags_use_eq: bool = True
        """Whether long command-line arguments are passed like `--long=arg`."""

        set_result: bool = True
        """Whether the Executor should mark a specified parameter as a result."""

    executable: str = Field(
        "/sdf/group/lcls/ds/tools/crystfel/0.10.2/bin/get_hkl",
        description="CrystFEL's reflection manipulation binary.",
        flag_type="",
    )
    in_file: str = Field(
        "",
        description="Path to input HKL file.",
        flag_type="-",
        rename_param="i",
    )
    out_file: str = Field(
        "",
        description="Path to output file.",
        flag_type="-",
        rename_param="o",
        is_result=True,
    )
    cell_file: str = Field(
        "",
        description="Path to a file containing unit cell information (PDB or CrystFEL format).",
        flag_type="-",
        rename_param="p",
    )
    output_format: str = Field(
        "mtz",
        description="Output format. One of mtz, mtz-bij, or xds. Otherwise CrystFEL format.",
        flag_type="--",
        rename_param="output-format",
    )
    expand: Optional[str] = Field(
        description="Reflections will be expanded to fill asymmetric unit of specified point group.",
        flag_type="--",
    )
    # Reducing reflections to higher symmetry
    twin: Optional[str] = Field(
        description="Reflections equivalent to specified point group will have intensities summed.",
        flag_type="--",
    )
    no_need_all_parts: Optional[bool] = Field(
        description="Use with --twin to allow reflections missing a 'twin mate' to be written out.",
        flag_type="--",
        rename_param="no-need-all-parts",
    )
    # Noise - Add to data
    noise: Optional[bool] = Field(
        description="Generate 10% uniform noise.", flag_type="--"
    )
    poisson: Optional[bool] = Field(
        description="Generate Poisson noise. Intensities assumed to be A.U.",
        flag_type="--",
    )
    adu_per_photon: Optional[int] = Field(
        description="Use with --poisson to convert A.U. to photons.",
        flag_type="--",
        rename_param="adu-per-photon",
    )
    # Remove duplicate reflections
    trim_centrics: Optional[bool] = Field(
        description="Duplicated reflections (according to symmetry) are removed.",
        flag_type="--",
    )
    # Restrict to template file
    template: Optional[str] = Field(
        description="Only reflections which also appear in specified file are written out.",
        flag_type="--",
    )
    # Multiplicity
    multiplicity: Optional[bool] = Field(
        description="Reflections are multiplied by their symmetric multiplicites.",
        flag_type="--",
    )
    # Resolution cutoffs
    cutoff_angstroms: Optional[Union[str, int, float]] = Field(
        description=(
            "Either n, or n1,n2,n3. For n, reflections < n are removed. "
            "For n1,n2,n3 anisotropic trunction performed at separate resolution "
            "limits for a*, b*, c*."
        ),
        flag_type="--",
        rename_param="cutoff-angstroms",
    )
    lowres: Optional[float] = Field(
        description="Remove reflections with d > n", flag_type="--"
    )
    highres: Optional[float] = Field(
        description="Synonym for first form of --cutoff-angstroms"
    )
    reindex: Optional[str] = Field(
        description="Reindex according to specified operator. E.g. k,h,-l.",
        flag_type="--",
    )
    # Override input symmetry
    symmetry: Optional[str] = Field(
        description="Point group symmetry to use to override. Almost always OMIT this option.",
        flag_type="--",
    )

    @validator("in_file", always=True)
    def validate_in_file(cls, in_file: str, values: Dict[str, Any]) -> str:
        if in_file == "":
            partialator_file: Optional[str] = read_latest_db_entry(
                f"{values['lute_config'].work_dir}", "MergePartialator", "out_file"
            )
            if partialator_file:
                return partialator_file
        return in_file

    @validator("out_file", always=True)
    def validate_out_file(cls, out_file: str, values: Dict[str, Any]) -> str:
        if out_file == "":
            partialator_file: Optional[str] = read_latest_db_entry(
                f"{values['lute_config'].work_dir}", "MergePartialator", "out_file"
            )
            if partialator_file:
                mtz_out: str = partialator_file.split(".")[0]
                mtz_out = f"{mtz_out}.mtz"
                return mtz_out
        return out_file

    @validator("cell_file", always=True)
    def validate_cell_file(cls, cell_file: str, values: Dict[str, Any]) -> str:
        if cell_file == "":
            idx_cell_file: Optional[str] = read_latest_db_entry(
                f"{values['lute_config'].work_dir}",
                "IndexCrystFEL",
                "cell_file",
                valid_only=False,
            )
            if idx_cell_file:
                return idx_cell_file
        return cell_file
