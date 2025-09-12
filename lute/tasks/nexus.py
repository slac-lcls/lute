"""
Class for handling NeXuS HDF5 files.

Classes:
    ConvertSMDToNexus: Convert a smalldata HDF5 file to a NeXuS HDF5 file.
"""

from __future__ import annotations

__all__ = ["ConvertSMDToNexus"]
__author__ = "Fred Poitevin and Gabriel Dorlhiac"

import copy
from typing import Any, Dict, List, Tuple, TYPE_CHECKING

import h5py  # type: ignore
import numpy as np
import numpy.typing as npt
from dxtbx.model import Detector, ExperimentList, Panel  # type: ignore
from dxtbx.format.nxmx_writer import NXmxWriter, phil_scope  # type: ignore
from scipy import constants  # type: ignore

from lute.execution.ipc import Message
from lute.tasks.task import Task

if TYPE_CHECKING:
    from lute.io.models.nexus import ConvertSMDToNexusParameters
else:
    from lute.io.parameters import TaskParameters

    ConvertSMDToNexusParameters = TaskParameters


class ConvertSMDToNexus(Task):
    """Task that converts a smalldata HDF5 file to a NeXus HDF5 file.

    The current implementation is NOT generic, and is targeted specifically at
    handling the Jungfrau16M in MFX. This can in principle be extended fairly
    easily, but there has been no reason to do this yet.
    """

    def __init__(self, *, params: ConvertSMDToNexusParameters, row_ids=None) -> None:
        self._task_parameters: ConvertSMDToNexusParameters
        super().__init__(params=params, row_ids=row_ids)

    def _pre_run(self) -> None:
        super()._pre_run()
        params: libtbx.phil.scope_extract = phil_scope.extract()  # type: ignore # noqa: F821
        params.nexus_details.source_name = "SLAC LCLS"
        params.nexus_details.instrument_name = "SLAC LCLS BEAMLINE MFX"
        params.nexus_details.instrument_short_name = "MFX"
        params.nexus_details.source_short_name = "LCLS"
        params.output_file = self._task_parameters.output
        self._nexus_writer: NXmxWriter = NXmxWriter(params)
        self._nexus_writer.construct_entry()

    def _run(self) -> None:
        E: "dxtbx_model_ext.Experiment" = ExperimentList.from_file(  # type: ignore # noqa: F821
            self._task_parameters.geom, False
        )[
            0
        ]
        D: "dxtbx_model_ext.Detector" = E.detector  # type: ignore # noqa: F821

        if self._task_parameters.flip:
            new_D: "dxtbx_model_ext.Detector" = Detector()  # type: ignore # noqa: F821
            p: "dxtbx_model_ext.DetectorNode"  # type: ignore # noqa: F821
            for _, p in enumerate(D):
                pd: Dict[str, Any] = p.to_dict()
                ox: float
                oy: float
                oz: float
                ox, oy, oz = p.get_origin()
                fast: npt.NDArray[np.int64] = np.array(p.get_slow_axis()) * -1
                slow: npt.NDArray[np.int64] = np.array(p.get_fast_axis()) * 1
                orig: Tuple[float, float, float] = -oy, ox, oz
                pd["origin"] = orig
                pd["slow_axis"] = tuple(slow)
                pd["fast_axis"] = tuple(fast)
                new_p: "dxtbx_model_ext.Panel" = Panel.from_dict(pd)  # type: ignore # noqa: F821
                new_D.add_panel(new_p)
            D = new_D

        self._nexus_writer.detector = D
        self._nexus_writer.construct_detector(detector=D)
        h5: h5py.File = h5py.File(self._task_parameters.input, "r")
        msg: Message = Message(contents="Loading beams")
        self._report_to_executor(msg)

        en_convert: float = 1e10 * constants.c * constants.h / constants.electron_volt
        waves: npt.NDArray[np.float64] = en_convert / h5["ebeamh/ebeamPhotonEnergy"][()]
        B: "dxtbx_model_ext.Beam" = E.beam  # type: ignore # noqa: F821

        if self._task_parameters.flip:
            B.set_unit_s0((0, 0, -1))

        beams: List["dxtbx_model_ext.Beam"] = []  # type: ignore # noqa: F821
        for i, w in enumerate(waves):
            b = copy.deepcopy(B)
            b.set_wavelength(w)
            beams.append(b)
            msg = Message(contents=f"Wave: {i}")
            self._report_to_executor(msg)

        self._nexus_writer.beams = beams
        self._nexus_writer.add_beams(beams)

        entry: h5py._hl.group.Group = self._nexus_writer.handle["entry"]
        data_group: h5py._hl.group.Group = entry.create_group("data")
        data_group.attrs["NX_class"] = "NXdata"
        imgs_path: str = "jungfrau/full_area"
        vlay: h5py._hl.vds.VirtualLayout = h5py.VirtualLayout(
            shape=h5[imgs_path].shape, dtype=h5[imgs_path].dtype
        )
        vs: h5py._hl.vds.VirtualSource = h5py.VirtualSource(
            h5.filename, imgs_path, h5[imgs_path].shape
        )
        vlay[:] = vs
        data_group.create_virtual_dataset("data", vlay)

        h5 = self._nexus_writer.handle
        del h5["entry/instrument/detector/sensor_material"]
        dt: np.dtypes.ObjectDType = h5py.special_dtype(vlen=str)
        h5.create_dataset(
            "entry/instrument/detector/sensor_material", data="Si", dtype=dt
        )

        del h5["entry/instrument/detector/sensor_thickness"]
        thick_dset = h5.create_dataset(
            "entry/instrument/detector/sensor_thickness", data=0.00032
        )
        thick_dset.attrs["units"] = "m"

        h5.close()
        msg = Message(contents="Finished HDF5 construction. Closing file.")
        self._report_to_executor(msg)

        # Wilko has configured data mover to look for a semaphore file `{output}.todo`
        # to initiate data transfer of the NeXuS file to NERSC
        msg = Message(contents="Writing semaphore file for data transfer to NERSC.")
        self._report_to_executor(msg)
        with open(f"{self._task_parameters.output}.todo", "w") as f:
            f.write("")

        msg = Message(contents="Finished conversion.")
        self._report_to_executor(msg)
