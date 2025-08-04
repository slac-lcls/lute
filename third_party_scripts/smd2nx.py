from argparse import ArgumentParser

ap = ArgumentParser()
ap.add_argument("--nexus", type=str, required=True)
ap.add_argument("--geom", type=str, required=True)
ap.add_argument("--psanah5", type=str, required=True)
ap.add_argument("--flip", action="store_true")
args = ap.parse_args()


import dxtbx
import h5py
from dxtbx.model import ExperimentList
from dxtbx.format.nxmx_writer import NXmxWriter
from dxtbx.format.nxmx_writer import phil_scope
from libtbx.phil import parse
from dxtbx.model import Beam
from copy import deepcopy


params = phil_scope.extract()
params.nexus_details.source_name = "SLAC LCLS"
params.nexus_details.instrument_name = "SLAC LCLS BEAMLINE MFX"
params.nexus_details.instrument_short_name = "MFX"
params.nexus_details.source_short_name = "LCLS"
params.output_file = args.nexus
writer = NXmxWriter(params)
writer.construct_entry()

E = ExperimentList.from_file(args.geom, False)[0]
D = E.detector
if args.flip:
    import numpy as np
    from dxtbx.model import Detector, Panel

    new_D = Detector()
    for i_p, p in enumerate(D):
        pd = p.to_dict()
        ox, oy, oz = p.get_origin()
        fast = np.array(p.get_slow_axis()) * -1
        slow = np.array(p.get_fast_axis()) * 1
        orig = -oy, ox, oz
        pd["origin"] = orig
        pd["slow_axis"] = tuple(slow)
        pd["fast_axis"] = tuple(fast)
        new_p = Panel.from_dict(pd)
        new_D.add_panel(new_p)
    D = new_D

writer.detector = D
writer.construct_detector(detector=D)
h = h5py.File(args.psanah5, "r")
print("loading beams")
from scipy import constants

en_convert = 1e10 * constants.c * constants.h / constants.electron_volt
waves = en_convert / h["ebeamh/ebeamPhotonEnergy"][()]
B = E.beam

if args.flip:
    B.set_unit_s0((0, 0, -1))

beams = []
for i, w in enumerate(waves):
    b = deepcopy(B)
    b.set_wavelength(w)
    beams.append(b)
    print(i)

writer.beams = beams
writer.add_beams(beams)

entry = writer.handle["entry"]
data_group = entry.create_group("data")
data_group.attrs["NX_class"] = "NXdata"
imgs_path = "jungfrau/full_area"
vlay = h5py.VirtualLayout(shape=h[imgs_path].shape, dtype=h[imgs_path].dtype)
vs = h5py.VirtualSource(h.filename, imgs_path, h[imgs_path].shape)
vlay[:] = vs
data_group.create_virtual_dataset("data", vlay)

h = writer.handle
del h["entry/instrument/detector/sensor_material"]
dt = h5py.special_dtype(vlen=str)
h.create_dataset("entry/instrument/detector/sensor_material", data="Si", dtype=dt)

del h["entry/instrument/detector/sensor_thickness"]
thick_dset = h.create_dataset(
    "entry/instrument/detector/sensor_thickness", data=0.00032
)
thick_dset.attrs["units"] = "m"

h.close()

# write semaphore file for Wilko
with open(f"{args.nexus}.todo", "w") as f:
    f.write("")
