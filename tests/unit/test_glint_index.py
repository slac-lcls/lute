"""Config-time validation of the IndexGLINT task parameters.

These validators decide whether a GLINT run does what its YAML says. Each one guards a failure that
would otherwise be silent: a job that "succeeds" and hands StreamFileConcatenator nothing usable.
"""

import tempfile
from typing import Any, Dict

import pytest

from lute.io.models.base import AnalysisHeader
from lute.io.models.glint_index import IndexGLINTParameters

HEADER = AnalysisHeader(experiment="test_exp", run=1, work_dir=tempfile.gettempdir())
XTC: Dict[str, Any] = dict(exp="mfxx49820", run=16, zdist=0.1027, out="o.stream")
LAUNCHER = "/path/to/glint/lute/glint_launch.sh"


def P(**kw: Any) -> IndexGLINTParameters:
    kw.setdefault("executable", LAUNCHER)
    return IndexGLINTParameters(lute_config=HEADER, **kw)


def bad(**kw: Any) -> str:
    """Assert the config is rejected and return the message, so a test can check which rule fired."""
    with pytest.raises(Exception) as e:
        P(**kw)
    return str(e.value)


# GLINT is not bundled with LUTE: no default launcher
def test_executable_is_required() -> None:
    with pytest.raises(Exception) as e:
        IndexGLINTParameters(lute_config=HEADER, peaks="p.stream", out="o.stream")
    assert "`executable` is required" in str(e.value)
    assert "`executable` is required" in bad(
        peaks="p.stream", out="o.stream", executable=""
    )
    assert P(peaks="p.stream", out="o.stream").executable == LAUNCHER


# Exactly one frame source
def test_no_source_rejected() -> None:
    assert "frame source is required" in bad(out="o.stream")


@pytest.mark.parametrize(
    "a,b",
    [
        ({"peaks": "p.stream"}, {"images": "i.cxi"}),
        ({"peaks": "p.stream"}, {"exp": "mfxx49820", "run": 16, "zdist": 0.1}),
        ({"images": "i.cxi"}, {"exp": "mfxx49820", "run": 16, "zdist": 0.1}),
    ],
)
def test_two_sources_rejected(a: Dict[str, Any], b: Dict[str, Any]) -> None:
    assert "exactly ONE frame source" in bad(out="o.stream", **a, **b)


@pytest.mark.parametrize(
    "src",
    [
        {"peaks": "p.stream"},
        {"images": "i.cxi"},
        {"exp": "mfxx49820", "run": 16, "zdist": 0.1027},
    ],
)
def test_one_source_accepted(src: Dict[str, Any]) -> None:
    assert P(out="o.stream", **src) is not None


# Raw xtc needs a run and a refined distance
@pytest.mark.parametrize("missing", ["run", "zdist"])
def test_xtc_requires_run_and_zdist(missing: str) -> None:
    kw = dict(XTC)
    kw.pop(missing)
    assert f"`{missing}` is required with `exp`" in bad(**kw)


# Peak-finder validity is per source
def test_peakfinder_defaults_per_source() -> None:
    assert P(**XTC).peakfinder == "v4"
    assert P(images="i.cxi", out="o.stream").peakfinder == "stored"


@pytest.mark.parametrize("pf", ["v4", "pf8", "pf8-panel"])
def test_peakfinder_valid_on_xtc(pf: str) -> None:
    assert P(peakfinder=pf, **XTC).peakfinder == pf


@pytest.mark.parametrize("pf", ["stored", "pf9"])
def test_peakfinder_rejected_on_xtc(pf: str) -> None:
    assert "not available on the `exp`" in bad(peakfinder=pf, **XTC)


@pytest.mark.parametrize("pf", ["pf8", "pf8-panel"])
def test_pf8_rejected_on_images(pf: str) -> None:
    assert "only wired on the `exp`" in bad(
        peakfinder=pf, images="i.cxi", out="o.stream"
    )


# Knobs that only the raw-xtc source reads
@pytest.mark.parametrize(
    "field,value",
    [
        ("det", "MfxEndstation.0:Epix10ka2M.0"),
        ("psana", "1"),
        ("calib_dir", "/tmp/calib"),
    ],
)
def test_xtc_only_knobs_rejected_elsewhere(field: str, value: str) -> None:
    assert "applies only to the `exp`" in bad(
        peaks="p.stream", out="o.stream", **{field: value}
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("det", "MfxEndstation.0:Epix10ka2M.0"),
        ("psana", "1"),
        ("calib_dir", "/tmp/calib"),
    ],
)
def test_xtc_only_knobs_accepted_on_xtc(field: str, value: str) -> None:
    assert P(**XTC, **{field: value}) is not None


def test_wavelength_is_accepted_on_every_source() -> None:
    assert P(peaks="p.stream", out="o.stream", wavelength=1.29) is not None


def test_top_peaks_only_with_images() -> None:
    assert "applies only to `images`" in bad(
        peaks="p.stream", out="o.stream", top_peaks=100
    )
    assert P(images="i.cxi", out="o.stream", top_peaks=100).top_peaks == 100


# Outputs the downstream Tasks depend on
def test_out_is_required() -> None:
    assert "`out` is required" in bad(peaks="p.stream", out="")


def test_integrate_with_peaks_needs_image_dir() -> None:
    assert "requires `image_dir`" in bad(
        peaks="p.stream", out="o.stream", integrate=True
    )
    assert (
        P(peaks="p.stream", out="o.stream", integrate=True, image_dir="/data")
        is not None
    )


def test_legacy_fromfile_is_renamed_not_dropped() -> None:
    assert P(peaks="p.stream", out="o.stream", fromfile="sol.txt").tofile == "sol.txt"
    assert "not both" in bad(
        peaks="p.stream", out="o.stream", fromfile="a.txt", tofile="b.txt"
    )


@pytest.mark.parametrize(
    "field,flag,values",
    [
        ("event_axis", "event-axis", ("auto", "event", "panel")),
        ("bg_mode", "bg-mode", ("clipmean", "median", "mean")),
        ("gate", "gate", ("none", "strict")),
    ],
)
def test_enum_fields_render_as_cli_flags(field: str, flag: str, values: tuple) -> None:
    f = IndexGLINTParameters.__fields__[field]
    assert f.field_info.extra["rename_param"] == flag
    assert f.field_info.extra["flag_type"] == "--"
    assert getattr(P(**XTC), field) is None  # unset defers to GLINT's own default
    for v in values:
        assert getattr(P(**dict(XTC, **{field: v})), field) == v
    with pytest.raises(Exception):  # a typo must not silently mean "default"
        P(**dict(XTC, **{field: "typo"}))
