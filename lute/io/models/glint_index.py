"""LUTE task-parameters model for the GLINT GPU indexer -- a drop-in alternative to CrystFELIndexer
in the SFX DAG:  PeakFinderSFX -> [IndexGLINT (peaks|images)] -> ConcatenateStreamFiles -> MergePartialator.

Why GLINT in LUTE: none of LUTE's bundled CrystFEL builds (0.10.2 default ... 0.12.0) are compiled
with FFBIDX, so GPU fast-feedback-style indexing is simply unavailable via indexamajig. GLINT fills
that gap -- GPU blind indexing + cross-frame consensus -- and emits the same CrystFEL `.stream` that
ConcatenateStreamFiles / partialator already consume. For the best MERGE, set `fromfile` (GLINT hands
CrystFEL the refined-merge solution file).

With `images` + `peakfinder=stored` GLINT reuses the .cxi's own peakfinder8 peaks (/entry_1/result_1);
`integrate=true` runs GLINT's NATIVE predict+integrate over data[event] in the stacked .cxi and writes
REAL I/sigma into the stream (int_dmin/int_tol tune the resolution limit and Ewald excitation-error gate),
so no CrystFEL integration step is needed on the merge path.

INSTALL (see lute/README.md): copy to `lute/io/models/glint_index.py`, add
`from .glint_index import *` to `lute/io/models/__init__.py`, and add
`GLINTIndexer: Executor = Executor("IndexGLINT")` to `lute/managed_tasks.py`.
"""

from typing import Optional

from pydantic import Field, PositiveInt

from lute.io.models.base import ThirdPartyParameters

__all__ = ["IndexGLINTParameters"]
__author__ = "GLINT (S. Marchesini)"


class IndexGLINTParameters(ThirdPartyParameters):
    """Parameters for the GLINT GPU blind SFX indexer (peaks + geometry -> CrystFEL .stream)."""

    class Config(ThirdPartyParameters.Config):
        set_result: bool = True
        result_from_params: str = ""

    executable: str = Field(
        "/sdf/home/s/smarches/git/glint/lute/glint_launch.sh",
        description="Launcher that activates the GLINT GPU (torch) env and runs glint.glint_cli.",
        flag_type="",
    )
    peaks: str = Field(
        "",
        description="CrystFEL peak-search stream from FindPeaksSFX (peakfinder8).",
        flag_type="--",
        rename_param="peaks",
    )
    images: Optional[str] = Field(
        None,
        description="Raw detector .cxi (jf16m) from FindPeaksSFX; GLINT peak-finds it itself with "
        "peakfinder_v4 -- self-contained GPU front end (no CrystFEL peak-search stream). "
        "Use EITHER images OR peaks.",
        flag_type="--",
        rename_param="images",
    )
    geom: str = Field(
        "",
        description="CrystFEL .geom file.",
        flag_type="--",
        rename_param="geom",
    )
    out: str = Field(
        "",
        description="Output .stream (mergeable).",
        flag_type="--",
        rename_param="out",
        is_result=True,
    )
    cell: Optional[str] = Field(
        None,
        description='Known unit cell "a b c al be ga" (else fully-blind cross-frame consensus).',
        flag_type="--",
        rename_param="cell",
    )
    mode: str = Field(
        "auto",
        description="Front end: auto | sparse (SFX stills) | dense (rotation clouds).",
        flag_type="--",
        rename_param="mode",
    )
    nbest: PositiveInt = Field(
        3,
        description="N-best consensus hypotheses kept per frame.",
        flag_type="--",
        rename_param="nbest",
    )
    min_peaks: PositiveInt = Field(
        6,
        description="Skip frames with fewer peaks.",
        flag_type="--",
        rename_param="min-peaks",
    )
    n: Optional[int] = Field(
        None,
        description="Limit to first N frames (0/None = all).",
        flag_type="-",
        rename_param="N",
    )
    device: str = Field(
        "auto",
        description="auto (GPU if present) | cpu.",
        flag_type="--",
        rename_param="device",
    )
    fromfile: Optional[str] = Field(
        None,
        description="Also emit a CrystFEL --indexing=file solution file (the refined-MERGE handoff): "
        "run `indexamajig --indexing=file --fromfile-input-file=<f> --tolerance=10,10,10,3`.",
        flag_type="--",
        rename_param="fromfile",
    )
    lattice: str = Field(
        "aP",
        description="Bravais lattice code for --fromfile (e.g. tPc tetragonal).",
        flag_type="--",
        rename_param="lattice",
    )
    cascade: Optional[str] = Field(
        None,
        description="Optional external cell-given indexer binary (ffbidx driver) as a fallback.",
        flag_type="--",
        rename_param="cascade",
    )
    peakfinder: str = Field(
        "stored",
        description="Peak source for --images: v4 | pf9 | pf8 | stored. 'stored' REUSES the .cxi's own "
        "peakfinder8 peaks (/entry_1/result_1) -- no re-finding, avoids v4 water-ring over-find.",
        flag_type="--",
        rename_param="peakfinder",
    )
    top_peaks: Optional[int] = Field(
        None,
        description="Keep only the N strongest peaks per frame (guards v4 over-finding on water rings).",
        flag_type="--",
        rename_param="top-peaks",
    )
    integrate: bool = Field(
        False,
        description="NATIVE predict+integrate reading data[event] from the stacked .cxi (emits REAL I/sigma). "
        "Bare on/off flag: True -> --integrate, False -> omitted.",
        flag_type="--",
        rename_param="integrate",
    )
    int_dmin: Optional[float] = Field(
        None,
        description="Integrate resolution limit in Angstrom (GLINT default 2.0; validated 2.1 for lyso).",
        flag_type="--",
        rename_param="int-dmin",
    )
    int_tol: Optional[float] = Field(
        None,
        description="Integrate Ewald excitation-error gate in 1/Angstrom (GLINT default 0.006).",
        flag_type="--",
        rename_param="int-tol",
    )
