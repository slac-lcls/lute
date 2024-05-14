"""Models for structure solution in serial femtosecond crystallography.

Classes:
    DimpleSolveParameters(ThirdPartyParameters): Perform structure solution
        using CCP4's dimple (molecular replacement).
"""

__all__ = [
    "DimpleSolveParameters",
    "RunSHELXCParameters",
    "RunSHELXDParameters",
    "EditSHELXDInstructionsParameters",
    "RunPhenixEMMAParameters",
    "EditPDBFileParameters",
]
__author__ = "Gabriel Dorlhiac"

import os
from typing import Union, List, Optional, Dict, Any, Tuple

from pydantic import Field, validator, PositiveFloat, PositiveInt, root_validator

from .base import ThirdPartyParameters, TaskParameters
from ..db import read_latest_db_entry


class DimpleSolveParameters(ThirdPartyParameters):
    """Parameters for CCP4's dimple program.

    There are many parameters. For more information on
    usage, please refer to the CCP4 documentation, here:
    https://ccp4.github.io/dimple/
    """

    executable: str = Field(
        "/sdf/group/lcls/ds/tools/ccp4-8.0/bin/dimple",
        description="CCP4 Dimple for solving structures with MR.",
        flag_type="",
    )
    # Positional requirements - all required.
    in_file: str = Field(
        "",
        description="Path to input mtz.",
        flag_type="",
    )
    pdb: str = Field("", description="Path to a PDB.", flag_type="")
    out_dir: str = Field("", description="Output DIRECTORY.", flag_type="")
    # Most used options
    mr_thresh: PositiveFloat = Field(
        0.4,
        description="Threshold for molecular replacement.",
        flag_type="--",
        rename_param="mr-when-r",
    )
    slow: Optional[bool] = Field(
        False, description="Perform more refinement.", flag_type="--"
    )
    # Other options (IO)
    hklout: str = Field(
        "final.mtz", description="Output mtz file name.", flag_type="--"
    )
    xyzout: str = Field(
        "final.pdb", description="Output PDB file name.", flag_type="--"
    )
    icolumn: Optional[str] = Field(
        # "IMEAN",
        description="Name for the I column.",
        flag_type="--",
    )
    sigicolumn: Optional[str] = Field(
        # "SIG<ICOL>",
        description="Name for the Sig<I> column.",
        flag_type="--",
    )
    fcolumn: Optional[str] = Field(
        # "F",
        description="Name for the F column.",
        flag_type="--",
    )
    sigfcolumn: Optional[str] = Field(
        # "F",
        description="Name for the Sig<F> column.",
        flag_type="--",
    )
    libin: Optional[str] = Field(
        description="Ligand descriptions for refmac (LIBIN).", flag_type="--"
    )
    refmac_key: Optional[str] = Field(
        description="Extra Refmac keywords to use in refinement.",
        flag_type="--",
        rename_param="refmac-key",
    )
    free_r_flags: Optional[str] = Field(
        description="Path to a mtz file with freeR flags.",
        flag_type="--",
        rename_param="free-r-flags",
    )
    freecolumn: Optional[Union[int, float]] = Field(
        # 0,
        description="Refree column with an optional value.",
        flag_type="--",
    )
    img_format: Optional[str] = Field(
        description="Format of generated images. (png, jpeg, none).",
        flag_type="-",
        rename_param="f",
    )
    white_bg: bool = Field(
        False,
        description="Use a white background in Coot and in images.",
        flag_type="--",
        rename_param="white-bg",
    )
    no_cleanup: bool = Field(
        False,
        description="Retain intermediate files.",
        flag_type="--",
        rename_param="no-cleanup",
    )
    # Calculations
    no_blob_search: bool = Field(
        False,
        description="Do not search for unmodelled blobs.",
        flag_type="--",
        rename_param="no-blob-search",
    )
    anode: bool = Field(
        False, description="Use SHELX/AnoDe to find peaks in the anomalous map."
    )
    # Run customization
    no_hetatm: bool = Field(
        False,
        description="Remove heteroatoms from the given model.",
        flag_type="--",
        rename_param="no-hetatm",
    )
    rigid_cycles: Optional[PositiveInt] = Field(
        # 10,
        description="Number of cycles of rigid-body refinement to perform.",
        flag_type="--",
        rename_param="rigid-cycles",
    )
    jelly: Optional[PositiveInt] = Field(
        # 4,
        description="Number of cycles of jelly-body refinement to perform.",
        flag_type="--",
    )
    restr_cycles: Optional[PositiveInt] = Field(
        # 8,
        description="Number of cycles of refmac final refinement to perform.",
        flag_type="--",
        rename_param="restr-cycles",
    )
    lim_resolution: Optional[PositiveFloat] = Field(
        description="Limit the final resolution.", flag_type="--", rename_param="reso"
    )
    weight: Optional[str] = Field(
        # "auto-weight",
        description="The refmac matrix weight.",
        flag_type="--",
    )
    mr_prog: Optional[str] = Field(
        # "phaser",
        description="Molecular replacement program. phaser or molrep.",
        flag_type="--",
        rename_param="mr-prog",
    )
    mr_num: Optional[Union[str, int]] = Field(
        # "auto",
        description="Number of molecules to use for molecular replacement.",
        flag_type="--",
        rename_param="mr-num",
    )
    mr_reso: Optional[PositiveFloat] = Field(
        # 3.25,
        description="High resolution for molecular replacement. If >10 interpreted as eLLG.",
        flag_type="--",
        rename_param="mr-reso",
    )
    itof_prog: Optional[str] = Field(
        description="Program to calculate amplitudes. truncate, or ctruncate.",
        flag_type="--",
        rename_param="ItoF-prog",
    )

    @validator("in_file", always=True)
    def validate_in_file(cls, in_file: str, values: Dict[str, Any]) -> str:
        if in_file == "":
            get_hkl_file: Optional[str] = read_latest_db_entry(
                f"{values['lute_config'].work_dir}", "ManipulateHKL", "out_file"
            )
            if get_hkl_file:
                return get_hkl_file
        return in_file

    @validator("out_dir", always=True)
    def validate_out_dir(cls, out_dir: str, values: Dict[str, Any]) -> str:
        if out_dir == "":
            get_hkl_file: Optional[str] = read_latest_db_entry(
                f"{values['lute_config'].work_dir}", "ManipulateHKL", "out_file"
            )
            if get_hkl_file:
                return os.path.dirname(get_hkl_file)
        return out_dir


class RunSHELXCParameters(ThirdPartyParameters):
    """Parameters for CCP4's SHELXC program.

    SHELXC prepares files for SHELXD and SHELXE.

    For more information please refer to the official documentation:
    https://www.ccp4.ac.uk/html/crank.html
    """

    executable: str = Field(
        "/bin/bash", description="Shell is required for redirect.", flag_type=""
    )

    shelxc_executable: Optional[str] = Field(
        "/sdf/group/lcls/ds/tools/ccp4-8.0/bin/shelxc",
        description="CCP4 SHELXC. Generates input files for SHELXD/SHELXE.",
        flag_type="",
    )
    outfiles_prefix: Optional[str] = Field(
        "",
        description="Prefix for generated output files, including path. (No extension)",
        flag_type="",
    )
    redirect: Optional[str] = Field("<", description="Redirect input", flag_type="")
    instructions_file: Optional[str] = Field(
        "",
        description="Input file for SHELXC with reflections AND proper records.",
        flag_type="",
    )

    bash_c_flag: str = Field(
        "",
        description="Command to run (SHELXC full string).",
        flag_type="-",
        rename_param="c",
    )

    ## NEED TO REWRITE THIS...
    # USAGE: `shelxc $OUT_FILES_PREFIX <INSTRUCTIONS`
    # INSTRUCTIONS will be a file like
    #
    # CELL 50.84 98.52 53.43 90.00 112.38 90.00
    # SPAG P21
    # SAD <output_>
    # FIND 3
    #
    # Will want a separate Task which writes this file by taking
    # the outputs from ManipulateHKL (location of XDS-formatted file)
    # and also a cell input file from somewhere (IndexCrystFEL?) and spacgroup?
    #

    @validator("outfiles_prefix")
    def validate_outfiles_prefix(cls, out_prefix: str, values: Dict[str, Any]) -> str:
        if out_prefix == "":
            # get_hkl needed to be run to produce an XDS format file...
            xds_format_file: Optional[str] = read_latest_db_entry(
                f"{values['lute_config'].work_dir}", "ManipulateHKL", "out_file"
            )
            out_directory: str
            if xds_format_file:
                out_directory = os.path.dirname(xds_format_file)
            else:
                out_directory = values["lute_config"].work_dir

            return f"{out_directory}/shelx"
        return out_prefix

    @validator("instructions_file", always=True)
    def validate_instructions_file(
        cls, instructions_file: str, values: Dict[str, Any]
    ) -> str:
        if instructions_file != "":
            return f"{instructions_file}"
        return instructions_file

    @validator("bash_c_flag")
    def validate_bash_command(cls, bash_cmd: str, values: Dict[str, Any]) -> str:
        """This validator also sets the run directory.

        It's easier here tahn with a separate root_validator because this
        validator sets all the values to None.
        """
        shelxc: str = values["shelxc_executable"]
        values["shelxc_executable"] = None

        outfiles_prefix: str = values["outfiles_prefix"]
        # SET run directory here!
        directory: str = os.path.dirname(outfiles_prefix)
        cls.Config.run_directory = f"{directory}"
        values["outfiles_prefix"] = None

        redirect: str = values["redirect"]
        values["redirect"] = None

        instructions_file: str = values["instructions_file"]
        values["instructions_file"] = None

        # return f"\"{shelxc} {outfiles_prefix} {redirect}{instructions_file}\""
        return f"'{shelxc} {outfiles_prefix} {redirect}{instructions_file}'"


# NEED A TASK WHICH replaces some lines
class EditSHELXDInstructionsParameters(TaskParameters):
    """Parameters for editing the SHELXD instructions generated by SHELXC."""

    in_file: str = Field("", description="Instruction file output from SHELXC.")
    SHEL: Tuple[float, float] = Field(
        (999, 3.0), description="Resolution cutoffs (min, max)."
    )
    MIND: Tuple[float, float] = Field(
        (-3.5, 2.2), description="Minimum distance between atoms."
    )
    ESEL: float = Field(1.5, description="Who knows.")
    TEST: Tuple[float, float] = Field((0, 99), description="Who knows.")
    out_file: str = Field("", description="Edited instructions output.")

    @validator("out_file")
    def validate_out_file(cls, out_file: str, values: Dict[str, Any]) -> str:
        if out_file == "":
            in_file: str = values["in_file"]
            return in_file
        return out_file


class RunSHELXDParameters(ThirdPartyParameters):
    """Parameters for CCP4's SHELXD program.

    SHELXD performs a heavy atom search. Input files can be created by SHELXC.

    For more information please refer to the official documentation:
    https://www.ccp4.ac.uk/html/crank.html

    Command is:
    /sdf/group/lcls/ds/tools/ccp4-8.0/bin/shelxd /sdf/data/lcls/ds/prj/prjlute22/results/lute_output/shelx_strep_fa
    """

    executable: str = Field(
        "/sdf/group/lcls/ds/tools/ccp4-8.0/bin/shelxd",
        description="CCP4 SHELXD. Searches for heavy atoms.",
        flag_type="",
    )
    instruction_file: str = Field(
        "", description="Search instructions. Generated by SHELXC.", flag_type=""
    )

    @validator("instruction_file")
    def validate_instructions_file(cls, in_file: str, values: Dict[str, Any]) -> str:
        if in_file == "":
            shelxc_cmd: Optional[str] = read_latest_db_entry(
                f"{values['lute_config'].work_dir}", "RunSHELXC", "bash_c_flag"
            )
            if shelxc_cmd is not None:
                prefix: str = shelxc_cmd.split()[1]
                return f"{prefix}_fa"
        return in_file


class RunPhenixEMMAParameters(ThirdPartyParameters):
    """Phenix's Euclidian Model Matching.

    Superimposes two structure solutions to derive a consensus.

    More information is availble at:
    https://phenix-online.org/documentation/reference/emma.html
    """

    executable: str = Field(
        "/sdf/group/lcls/ds/tools/phenix/phenix-1.16-3549/build/bin/phenix.emma",
        description="Phenix EMMA program for model comparison.",
        flag_type="",
    )
    unit_cell: Optional[str] = Field(
        description=(
            "Unit cell parameters listed as a comma-separated list.\n"
            "Listed as: a,b,c,al,be,ga"
        ),
        flag_type="--",
    )
    space_group: Optional[str] = Field(
        description="Space group symbol, e.g. P212121.",
        flag_type="--",
    )
    symmetry_file: Optional[str] = Field(
        description="External file with symmetry information.",
        rename_param="symmetry",
        flag_type="--",
    )
    tolerance: Optional[float] = Field(
        description="Match tolerance for comparison.", flag_type="--"
    )
    diffraction_index_equivalent: Optional[bool] = Field(
        description="Use only if models are diffraction-index equivalent.",
        flag_type="--",
    )
    reference_coords: str = Field(
        "", description="Reference coordinates to compare with.", flag_type=""
    )
    other_coords: str = Field(
        "", description="Other coordinates to compare with.", flag_type=""
    )


class EditPDBFileParameters(TaskParameters):
    """Perform modifications of a PDB file."""

    in_file: str = Field("", description="PDB to modify.")
    delete_hetatom_number: Optional[Union[int, List[int]]] = Field(
        description=(
            "Delete an atom (or list of atoms), specified by indices.\n"
            "Atoms are numbered beginning from 1!"
        )
    )
    substitute_element: Optional[Tuple[str, str]] = Field(
        description="Substitute instances of one element for another."
    )
    out_file: str = Field("", description="Edited output PDB.")

    @validator("out_file")
    def validate_out_file(cls, out_file: str, values: Dict[str, Any]) -> str:
        if out_file == "":
            in_file: str = values["in_file"]
            return in_file
        return out_file


class RunSolomonParameters(ThirdPartyParameters):
    """Density modification program by solvent flipping.

    More information available here:
    https://www.ccp4.ac.uk/html/solomon.html

    Usage:
    solomon MAPIN foo_in.map [ MAPOUT foo_out.map ] [ RMSMAP foo_out2.map ]
    """

    executable: str = Field(
        "/sdf/group/lcls/ds/tools/ccp4-8.0/bin/solomon",
        description="CCP4 Solomon density modification program.",
        flag_type="",
    )


class RunMulticombParameters(ThirdPartyParameters):
    """Phase combination program.

    More information available here:

    """

    executable: str = Field(
        "/sdf/group/lcls/ds/tools/ccp4-8.0/bin/multicomb",
        description="CCP4 phase combination program.",
        flag_type="",
    )


class RunParrotParameters(ThirdPartyParameters):
    """Phase combination program.

    More information available here:
    https://www.ccp4.ac.uk/html/parrot.html

    Usage:
        -mtzin-ref <filename>
        -pdbin-ref <filename>
        -mtzin <filename>		COMPULSORY
        -seqin <filename>
        -pdbin <filename>
        -pdbin-ha <filename>
        -pdbin-mr <filename>
        -colin-ref-fo <colpath>
        -colin-ref-hl <colpath>
        -colin-fo <colpath>		COMPULSORY
        -colin-hl <colpath> or -colin-phifom <colpath>	COMPULSORY
        -colin-fc <colpath>
        -colin-free <colpath>
        -mtzout <filename>
        -colout <colpath>
        -colout-hl <colpath>
        -colout-fc <colpath>
        -mapout-ncs <filename prefix>
        -solvent-flatten
        -histogram-match
        -ncs-average
        -rice-probability
        -anisotropy-correction
        -cycles <cycles>
        -resolution <resolution/A>
        -solvent-content <fraction>
        -solvent-mask-filter-radius <radius>
        -ncs-mask-filter-radius <radius>
        -ncs-asu-fraction <fraction>
        -ncs-operator <alpha>,<beta>,<gamma>,<x>,<y>,<z>,<x>,<y>,<z>
        -xmlout <filename>
    An input mtz is specified, F's and HL coefficients are required.
    """

    executable: str = Field(
        "/sdf/group/lcls/ds/tools/ccp4-8.0/bin/multicomb",
        description="CCP4 phase combination program.",
        flag_type="",
    )


class RunRefmac5Parameters(ThirdPartyParameters):
    """Macromolecular refinement program.

    More information available here:
    https://www.ccp4.ac.uk/html/refmac5/description.html

    Usage:
    refmac5 XYZIN foo_cycle_i.brk HKLIN foo.mtz TLSIN tlsin.txt TLSOUT tlsout.txt XYZOUT foo_cycle_j.brk HKLOUT foo_cycle_j.mtz
    """

    executable: str = Field(
        "/sdf/group/lcls/ds/tools/ccp4-8.0/bin/refmac5",
        description="CCP4 refinement program.",
        flag_type="",
    )
