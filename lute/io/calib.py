"""Tools for mapping sources and groups to a detector type.

The current implementation tries to mimic the PSCalib source code
from CalibFileFinder.py (https://github.com/lcls-psana/PSCalib/blob/master/src/CalibFileFinder.py).

Functions:
    group_from_det_type(det_type: str) -> str: Retrieve the group string

    source_from_det_info(det_type: str, hutch: str) -> str: Retrieve the source string

Exceptions:
"""

import os

__all__ = ["group_from_det_type", "source_from_det_info"]
__author__ = "Louis Conreux"

calib_groups = (
    "UNDEFINED",
    "CsPad::CalibV1",
    "CsPad2x2::CalibV1",
    "Princeton::CalibV1",
    "PNCCD::CalibV1",
    "Camera::CalibV1",
    "Camera::CalibV1",
    "Camera::CalibV1",
    "Camera::CalibV1",
    "Camera::CalibV1",
    "Camera::CalibV1",
    "Epix::CalibV1",
    "Epix10ka::CalibV1",
    "Epix100a::CalibV1",
    "Camera::CalibV1",
    "Andor::CalibV1",
    "Acqiris::CalibV1",
    "Imp::CalibV1",
    "Camera::CalibV1",
    "Camera::CalibV1",
    "EvrData::CalibV1",
    "Camera::CalibV1",
    "Timepix::CalibV1",
    "Fli::CalibV1",
    "Pimax::CalibV1",
    "Andor3d::CalibV1",
    "Jungfrau::CalibV1",
    "Camera::CalibV1",
    "Camera::CalibV1",
    "Epix10ka::CalibV1",
    "Uxi::CalibV1",
    "Pixis::CalibV1",
    "Epix10ka2M::CalibV1",
    "Epix10kaQuad::CalibV1",
    "Camera::CalibV1",
    "Camera::CalibV1",
    "Camera::CalibV1",
    "Camera::CalibV1",
)
det_names = (
    "UNDEFINED",
    "Cspad",
    "Cspad2x2",
    "Princeton",
    "pnCCD",
    "Tm6740",
    "Opal1000",
    "Opal2000",
    "Opal4000",
    "Opal8000",
    "OrcaFl40",
    "Epix",
    "Epix10k",
    "Epix100a",
    "Fccd960",
    "Andor",
    "Acqiris",
    "Imp",
    "Quartz4A150",
    "Rayonix",
    "Evr",
    "Fccd",
    "Timepix",
    "Fli",
    "Pimax",
    "Andor3d",
    "Jungfrau",
    "Zyla",
    "ControlsCamera",
    "Epix10ka",
    "Uxi",
    "Pixis",
    "Epix10ka2M",
    "Epix10kaQuad",
    "Streak",
    "Archon",
    "iStar",
    "Alvium",
)

hutches = (
    "UNDEFINED",
    "XPP",
    "XCS",
    "CXI",
    "MEC",
    "MFX",
)

stations = (
    "UNDEFINED",
    "XppEndstation.0",
    "XcsEndstation.0",
    "CxiDs1.0",
    "MecTargetChamber.0",
    "MfxEndstation.0",
)


det_names_lower = tuple(name.lower() for name in det_names)
det_lower_to_name = dict(zip(det_names_lower, det_names))
det_to_group = dict(zip(det_names, calib_groups))
det_to_group_lower = dict(zip(det_names_lower, calib_groups))

hutches_lower = tuple(hutch.lower() for hutch in hutches)
hutch_to_station = dict(zip(hutches, stations))
hutch_to_station_lower = dict(zip(hutches_lower, stations))

def group_from_det_type(det_type: str) -> str:
    """Retrieve the group string from the detector type."""
    det_type_lower = det_type.lower()
    group = det_to_group_lower.get(det_type_lower, "NOT IMPLEMENTED")
    if group == "NOT IMPLEMENTED":
        raise ValueError(f"Unknown detector type: {det_type}")
    return group

def source_from_det_info(det_type: str, hutch: str) -> str:
    """Retrieve the source string from the detector type and hutch."""
    hutch_upper = hutch.upper()
    source_begin = hutch_to_station_lower.get(hutch_upper, "NOT IMPLEMENTED")
    if source_begin == "NOT IMPLEMENTED":
        raise ValueError(f"Unknown hutch: {hutch}")
    if "." in det_type and det_type.split(".")[-1].isdigit():
        det_type, det_id = det_type.rsplit(".", 1)
        det_type_lower = det_type.lower()
        det = det_lower_to_name.get(det_type_lower, "NOT IMPLEMENTED")
        if det == "NOT IMPLEMENTED":
            raise ValueError(f"Unknown detector type: {det_type}")
        return f"{source_begin}:{det}.{det_id}"
    else:
        det_type_lower = det_type.lower()
        det = det_lower_to_name.get(det_type_lower, "NOT IMPLEMENTED")
        if det == "NOT IMPLEMENTED":
            raise ValueError(f"Unknown detector type: {det_type}")
        return f"{source_begin}:{det}.0"

def select_calib_file(calib_dir: str, run: int) -> str:
    """Select the calibration file from the calibration directory and run number."""
    fnames = os.listdir(calib_dir)
    files = [os.path.join(calib_dir, fname) for fname in fnames]

    run_max = 9999
    run_files = []
    for file in files:
        f = os.path.basename(file)
        if f == "HISTORY":
            continue
        if os.path.splitext(f)[1] != ".data":
            continue
        basename = os.path.splitext(f)[0]
        fields = basename.split('-')
        begin, end = fields

        if begin.isdigit():
            begin_int = int(begin)
            if begin_int >= run_max:
                raise ValueError(f"Begin run number {run} is too high for calibration directory {calib_dir}")
        
        if end.isdigit():
            end_int = int(end)
            if end_int >= run_max:
                raise ValueError(f"End run number {run} is too high for calibration directory {calib_dir}")
        elif end == "end":
            end_int = run_max
        
        run_files.append((begin, end, file))
    sorted_list = sorted(list, key=lambda x: int(x[0]))

    for run_file in sorted_list[::-1]:
        if run_file[0] <= run <= run_file[1]:
            return run_file[2]
    
    return ""
    