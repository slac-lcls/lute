"""Models for merging reflections in serial femtosecond crystallography.

Classes:
    IndexCrystFEL(BaseBinaryParameters): Perform indexing of hits/peaks using
        CrystFEL's `indexamajig`.
"""

__all__ = ["MergePartialatorParameters", "CompareHKLParameters", "ManipulateHKLParameters"]
__author__ = "Gabriel Dorlhiac"

from typing import Union, List, Optional, Dict, Any

from pydantic import Field

from .base import BaseBinaryParameters
from ..db import read_latest_db_entry


class MergePartialatorParameters(BaseBinaryParameters):
    """Parameters for CrystFEL's `partialator`.

    There are many parameters, and many combinations. For more information on
    usage, please refer to the CrystFEL documentation, here:
    https://www.desy.de/~twhite/crystfel/manual-partialator.html
    """

    class Config(BaseBinaryParameters.Config):
        long_flags_use_eq: bool = True
        """Whether long command-line arguments are passed like `--long=arg`."""

    executable: str = Field(
        "/sdf/group/lcls/ds/tools/crystfel/0.10.2/bin/partialator",
        description="CrystFEL's Partialator binary.",
        flag_type="",
    )
    in_file: Optional[str] = Field(
        "", description="Path to input stream.", flag_type="-", rename_param="i"
    )
    out_file: str = Field(
        "partialator.hkl",
        description="Path to output file.",
        flag_type="-",
        rename_param="o",
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
        1, description="Number of parallel analyses.", flag_type="-", rename_param="j"
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


class CompareHKLParameters(BaseBinaryParameters):
    """Parameters for CrystFEL's `compare_hkl` for calculating figures of merit.

    There are many parameters, and many combinations. For more information on
    usage, please refer to the CrystFEL documentation, here:
    https://www.desy.de/~twhite/crystfel/manual-partialator.html
    """

    class Config(BaseBinaryParameters.Config):
        long_flags_use_eq: bool = True
        """Whether long command-line arguments are passed like `--long=arg`."""

    executable: str = Field(
        "/sdf/group/lcls/ds/tools/crystfel/0.10.2/bin/compare_hkl",
        description="CrystFEL's reflection comparison binary.",
        flag_type="",
    )
    in_files: Optional[str] = Field(
        "out.hkl1 out.hkl2", description="Path to input stream.", flag_type=""
    )
    symmetry: str = Field(description="Point group symmetry.", flag_type="--")
    cell_file: Optional[str] = Field(
        description="Path to a file containing unit cell information (PDB or CrystFEL format).",
        flag_type="-",
        rename_param="p",
    )
    fom: str = Field(
        "Rsplit", description="Specify figure of merit to calculate.", flag_type="--"
    )
    nshells: int = Field(10, description="Use n resolution shells.", flag_type="--")
    fix_unity: bool = Field(
        False,
        description="Fix scale factors to unity.",
        flag_type="-",
        rename_param="u",
    )
    shell_file: str = Field(
        description="Write the statistics in resolution shells to a file.",
        flag_type="--",
        rename_param="shell-file",
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
        "-infinity",
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


class ManipulateHKLParameters(BaseBinaryParameters):
    """Parameters for CrystFEL's `get_hkl` for manipulating lists of reflections.

    This Task is predominantly used internally to convert `hkl` to `mtz` files.
    Note that performing multiple manipulations is undefined behaviour. Run
    the Task with multiple configurations in explicit separate steps. For more
    information on usage, please refer to the CrystFEL documentation, here:
    https://www.desy.de/~twhite/crystfel/manual-partialator.html
    """

    class Config(BaseBinaryParameters.Config):
        long_flags_use_eq: bool = True
        """Whether long command-line arguments are passed like `--long=arg`."""

    executable: str = Field(
        "/sdf/group/lcls/ds/tools/crystfel/0.10.2/bin/get_hkl",
        description="CrystFEL's reflection manipulation binary.",
        flag_type="",
    )
    in_file: Optional[str] = Field(
        "reflections.hkl",
        description="Path to input stream.",
        flag_type="-",
        rename_param="i",
    )
    out_file: str = Field(
        "output.hkl",
        description="Path to output file.",
        flag_type="-",
        rename_param="o",
    )
    cell_file: Optional[str] = Field(
        description="Path to a file containing unit cell information (PDB or CrystFEL format).",
        flag_type="-",
        rename_param="p",
    )
    output_format: Optional[str] = Field(
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
    cutoff_angstroms: Union[str, int, float] = Field(
        description="Either n, or n1,n2,n3. For n, reflections < n are removed. For n1,n2,n3 anisotropic trunction performed at separate resolution limits for a*, b*, c*.",
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