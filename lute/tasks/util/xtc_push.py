"""Starts Zmq process to send xtc1 file.

Based on Mona's converter from https://github.com/monarin/xtc1to2
"""


# Specify the dataset and detector...
exp, run, mode, detector_name = "amo06516", "90", "idx", "pnccdFront"
geom = (
    "/reg/d/psdm/amo/amo06516/calib/PNCCD::CalibV1/Camp.0:pnCCD.0/geometry/38-end.data"
)

