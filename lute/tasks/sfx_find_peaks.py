import sys
from pathlib import Path
from typing import Any, Dict, List, Literal, TextIO, Tuple

import h5py
import numpy
from libpressio import PressioCompressor
from mpi4py.MPI import COMM_WORLD, SUM
from numpy.typing import NDArray
from psalgos.pypsalgos import PyAlgos
from psana import Detector, EventId, MPIDataSource

from lute.execution.ipc import Message
from lute.io.models.base import *
from lute.tasks.task import *


class CxiWriter:

    def __init__(
        self,
        outdir: str,
        rank: int,
        exp: str,
        run: int,
        n_events: int,
        det_shape: Tuple[int, ...],
        min_peaks: int,
        max_peaks: int,
        i_x: Any,  # Not typed becomes it comes from psana
        i_y: Any,  # Not typed becomes it comes from psana
        ipx: Any,  # Not typed becomes it comes from psana
        ipy: Any,  # Not typed becomes it comes from psana
        tag: str,
    ):
        """
        Set up the CXI files to which peak finding results will be saved.

        Parameters:

            outdir (str): Output directory for cxi file.

            rank (int): MPI rank of the caller.

            exp (str): Experiment string.

            run (int): Experimental run.

            n_events (int): Number of events to process.

            det_shape (Tuple[int, int]): Shape of the numpy array storing the detector
                data. This must be aCheetah-stile 2D array.

            min_peaks (int): Minimum number of peaks per image.

            max_peaks (int): Maximum number of peaks per image.

            i_x (Any): Array of pixel indexes along x

            i_y (Any): Array of pixel indexes along y

            ipx (Any): Pixel indexes with respect to detector origin (x component)

            ipy (Any): Pixel indexes with respect to detector origin (y component)

            tag (str): Tag to append to cxi file names.
        """
        self._det_shape: Tuple[int, ...] = det_shape
        self._i_x: Any = i_x
        self._i_y: Any = i_y
        self._ipx: Any = ipx
        self._ipy: Any = ipy
        self._index: int = 0

        # Create and open the HDF5 file
        fname: str = f"{exp}_r{run:0>4}_{rank}{tag}.cxi"
        Path(outdir).mkdir(exist_ok=True)
        self._outh5: Any = h5py.File(Path(outdir) / fname, "w")

        # Entry_1 entry for processing with CrystFEL
        entry_1: Any = self._outh5.create_group("entry_1")
        keys: List[str] = [
            "nPeaks",
            "peakXPosRaw",
            "peakYPosRaw",
            "rcent",
            "ccent",
            "rmin",
            "rmax",
            "cmin",
            "cmax",
            "peakTotalIntensity",
            "peakMaxIntensity",
            "peakRadius",
        ]
        ds_expId: Any = entry_1.create_dataset(
            "experimental_identifier", (n_events,), maxshape=(None,), dtype=int
        )
        ds_expId.attrs["axes"] = "experiment_identifier"
        data_1: Any = entry_1.create_dataset(
            "/entry_1/data_1/data",
            (n_events, det_shape[0], det_shape[1]),
            chunks=(1, det_shape[0], det_shape[1]),
            maxshape=(None, det_shape[0], det_shape[1]),
            dtype=numpy.float32,
        )
        data_1.attrs["axes"] = "experiment_identifier"
        key: str
        for key in ["powderHits", "powderMisses", "mask"]:
            entry_1.create_dataset(
                f"/entry_1/data_1/{key}",
                (det_shape[0], det_shape[1]),
                chunks=(det_shape[0], det_shape[1]),
                maxshape=(det_shape[0], det_shape[1]),
                dtype=float,
            )

        # Peak-related entries
        for key in keys:
            if key == "nPeaks":
                ds_x: Any = self._outh5.create_dataset(
                    f"/entry_1/result_1/{key}",
                    (n_events,),
                    maxshape=(None,),
                    dtype=int,
                )
                ds_x.attrs["minPeaks"] = min_peaks
                ds_x.attrs["maxPeaks"] = max_peaks
            else:
                ds_x: Any = self._outh5.create_dataset(
                    f"/entry_1/result_1/{key}",
                    (n_events, max_peaks),
                    maxshape=(None, max_peaks),
                    chunks=(1, max_peaks),
                    dtype=float,
                )
            ds_x.attrs["axes"] = "experiment_identifier:peaks"

        # Timestamp entries
        lcls_1: Any = self._outh5.create_group("LCLS")
        keys: List[str] = [
            "eventNumber",
            "machineTime",
            "machineTimeNanoSeconds",
            "fiducial",
            "photon_energy_eV",
        ]
        key: str
        for key in keys:
            if key == "photon_energy_eV":
                ds_x: Any = lcls_1.create_dataset(
                    f"{key}", (n_events,), maxshape=(None,), dtype=float
                )
            else:
                ds_x = lcls_1.create_dataset(
                    f"{key}", (n_events,), maxshape=(None,), dtype=int
                )
            ds_x.attrs["axes"] = "experiment_identifier"

        ds_x = self._outh5.create_dataset(
            "/LCLS/detector_1/EncoderValue", (n_events,), maxshape=(None,), dtype=float
        )
        ds_x.attrs["axes"] = "experiment_identifier"

    def write_event(
        self,
        img: NDArray[numpy.float_],
        peaks: Any,  # Not typed becomes it comes from psana
        timestamp_seconds: int,
        timestamp_nanoseconds: int,
        timestamp_fiducials: int,
        photon_energy: float,
    ):
        """
        Write peak finding results for an event into the HDF5 file.

        Parameters:

            img (NDArray[numpy.float_]): Detector data for the event

            peaks: (Any): Peak information for the event, as recovered from the PyAlgos
                algorithm

            timestamp_seconds (int): Second part of the event's timestamp information

            timestamp_nanoseconds (int): Nanosecond part of the event's timestamp
                information

            timestamp_fiducials (int): Fiducials part of the event's timestamp
                information

            photon_energy (float): Photon energy for the event
        """
        ch_rows: NDArray[numpy.float_] = peaks[:, 0] * self._det_shape[1] + peaks[:, 1]
        ch_cols: NDArray[numpy.float_] = peaks[:, 2]

        # Entry_1 entry for processing with CrystFEL
        self._outh5["/entry_1/data_1/data"][self._index, :, :] = img.reshape(
            -1, img.shape[-1]
        )
        self._outh5["/entry_1/result_1/nPeaks"][self._index] = peaks.shape[0]
        self._outh5["/entry_1/result_1/peakXPosRaw"][self._index, : peaks.shape[0]] = (
            ch_cols.astype("int")
        )
        self._outh5["/entry_1/result_1/peakYPosRaw"][self._index, : peaks.shape[0]] = (
            ch_rows.astype("int")
        )
        self._outh5["/entry_1/result_1/rcent"][self._index, : peaks.shape[0]] = peaks[
            :, 6
        ]
        self._outh5["/entry_1/result_1/ccent"][self._index, : peaks.shape[0]] = peaks[
            :, 7
        ]
        self._outh5["/entry_1/result_1/rmin"][self._index, : peaks.shape[0]] = peaks[
            :, 10
        ]
        self._outh5["/entry_1/result_1/rmax"][self._index, : peaks.shape[0]] = peaks[
            :, 11
        ]
        self._outh5["/entry_1/result_1/cmin"][self._index, : peaks.shape[0]] = peaks[
            :, 12
        ]
        self._outh5["/entry_1/result_1/cmax"][self._index, : peaks.shape[0]] = peaks[
            :, 13
        ]
        self._outh5["/entry_1/result_1/peakTotalIntensity"][
            self._index, : peaks.shape[0]
        ] = peaks[:, 5]
        self._outh5["/entry_1/result_1/peakMaxIntensity"][
            self._index, : peaks.shape[0]
        ] = peaks[:, 4]

        # Calculate and write pixel radius
        peaks_cenx: NDArray[numpy.float_] = (
            self._i_x[
                numpy.array(peaks[:, 0], dtype=numpy.int64),
                numpy.array(peaks[:, 1], dtype=numpy.int64),
                numpy.array(peaks[:, 2], dtype=numpy.int64),
            ]
            + 0.5
            - self._ipx
        )
        peaks_ceny: NDArray[numpy.float_] = (
            self._i_y[
                numpy.array(peaks[:, 0], dtype=numpy.int64),
                numpy.array(peaks[:, 1], dtype=numpy.int64),
                numpy.array(peaks[:, 2], dtype=numpy.int64),
            ]
            + 0.5
            - self._ipy
        )
        peak_radius: NDArray[numpy.float_] = numpy.sqrt(
            (peaks_cenx**2) + (peaks_ceny**2)
        )
        self._outh5["/entry_1/result_1/peakRadius"][
            self._index, : peaks.shape[0]
        ] = peak_radius

        # LCLS entry dataset
        self._outh5["/LCLS/machineTime"][self._index] = timestamp_seconds
        self._outh5["/LCLS/machineTimeNanoSeconds"][self._index] = timestamp_nanoseconds
        self._outh5["/LCLS/fiducial"][self._index] = timestamp_fiducials
        self._outh5["/LCLS/photon_energy_eV"][self._index] = photon_energy

        self._index += 1

    def write_non_event_data(
        self,
        powder_hits: NDArray[numpy.float_],
        powder_misses: NDArray[numpy.float_],
        mask: NDArray[numpy.uint16],
        clen: float,
    ):
        """
        Write to the file data that is not related to a specific event (masks, powders)

        Parameters:

            powder_hits (NDArray[numpy.float_]): Virtual powder pattern from hits

            powder_misses (NDArray[numpy.float_]): Virtual powder pattern from hits

            mask: (NDArray[numpy.uint16]): Pixel ask to write into the file

        """
        # Add powders and mask to files, reshaping them to match the crystfel
        # convention
        self._outh5["/entry_1/data_1/powderHits"][:] = powder_hits.reshape(
            -1, powder_hits.shape[-1]
        )
        self._outh5["/entry_1/data_1/powderMisses"][:] = powder_misses.reshape(
            -1, powder_misses.shape[-1]
        )
        self._outh5["/entry_1/data_1/mask"][:] = (1 - mask).reshape(
            -1, mask.shape[-1]
        )  # Crystfel expects inverted values

        # Add clen distance
        self._outh5["/LCLS/detector_1/EncoderValue"][:] = clen

    def optimize_and_close_file(
        self,
        num_hits: int,
        max_peaks: int,
    ):
        """
        Resize data blocks and write additional information to the file

        Parameters:

            num_hits (int): Number of hits for which information has been saved to the
                file

            max_peaks (int): Maximum number of peaks (per event) for which information
                can be written into the file
        """

        # Resize the entry_1 entry
        data_shape: Tuple[int, ...] = self._outh5["/entry_1/data_1/data"].shape
        self._outh5["/entry_1/data_1/data"].resize(
            (num_hits, data_shape[1], data_shape[2])
        )
        self._outh5[f"/entry_1/result_1/nPeaks"].resize((num_hits,))
        key: str
        for key in [
            "peakXPosRaw",
            "peakYPosRaw",
            "rcent",
            "ccent",
            "rmin",
            "rmax",
            "cmin",
            "cmax",
            "peakTotalIntensity",
            "peakMaxIntensity",
            "peakRadius",
        ]:
            self._outh5[f"/entry_1/result_1/{key}"].resize((num_hits, max_peaks))

        # Resize LCLS entry
        for key in [
            "eventNumber",
            "machineTime",
            "machineTimeNanoSeconds",
            "fiducial",
            "detector_1/EncoderValue",
            "photon_energy_eV",
        ]:
            self._outh5[f"/LCLS/{key}"].resize((num_hits,))
        self._outh5.close()


def write_master_file(
    mpi_size: int,
    outdir: str,
    exp: str,
    run: int,
    tag: str,
    n_hits_per_rank: List[int],
    n_hits_total: int,
) -> Path:
    """
    Generate a virtual dataset to map all individual files for this run.

    Parameters:

        mpi_size (int): Number of ranks in the MPI pool.

        outdir (str): Output directory for cxi file.

        exp (str): Experiment string.

        run (int): Experimental run.

        tag (str): Tag to append to cxi file names.

        n_hits_per_rank (List[int]): Array containing the number of hits found on each
            node processing data.

        n_hits_total (int): Total number of hits found across all nodes.

    Returns:

        The path to the the written master file
    """
    # Retrieve paths to the files containing data
    fnames: List[Path] = []
    fi: int
    for fi in range(mpi_size):
        if n_hits_per_rank[fi] > 0:
            fnames.append(Path(outdir) / f"{exp}_r{run:0>4}_{fi}{tag}.cxi")
    if len(fnames) == 0:
        sys.exit("No hits found")

    # Retrieve list of entries to populate in the virtual hdf5 file
    dname_list, key_list, shape_list, dtype_list = [], [], [], []
    datasets = ["/entry_1/result_1", "/LCLS/detector_1", "/LCLS", "/entry_1/data_1"]
    f = h5py.File(fnames[0], "r")
    for dname in datasets:
        dset = f[dname]
        for key in dset.keys():
            if f"{dname}/{key}" not in datasets:
                dname_list.append(dname)
                key_list.append(key)
                shape_list.append(dset[key].shape)
                dtype_list.append(dset[key].dtype)
    f.close()

    # Compute cumulative powder hits and misses for all files
    powder_hits, powder_misses = None, None
    for fn in fnames:
        f = h5py.File(fn, "r")
        if powder_hits is None:
            powder_hits = f["entry_1/data_1/powderHits"][:].copy()
            powder_misses = f["entry_1/data_1/powderMisses"][:].copy()
        else:
            powder_hits = numpy.maximum(
                powder_hits, f["entry_1/data_1/powderHits"][:].copy()
            )
            powder_misses = numpy.maximum(
                powder_misses, f["entry_1/data_1/powderMisses"][:].copy()
            )
        f.close()

    vfname: Path = Path(outdir) / f"{exp}_r{run:0>4}{tag}.cxi"
    with h5py.File(vfname, "w") as vdf:

        # Write the virtual hdf5 file
        for dnum in range(len(dname_list)):
            dname = f"{dname_list[dnum]}/{key_list[dnum]}"
            if key_list[dnum] not in ["mask", "powderHits", "powderMisses"]:
                layout = h5py.VirtualLayout(
                    shape=(n_hits_total,) + shape_list[dnum][1:], dtype=dtype_list[dnum]
                )
                cursor = 0
                for i, fn in enumerate(fnames):
                    vsrc = h5py.VirtualSource(
                        fn, dname, shape=(n_hits_per_rank[i],) + shape_list[dnum][1:]
                    )
                    if len(shape_list[dnum]) == 1:
                        layout[cursor : cursor + n_hits_per_rank[i]] = vsrc
                    else:
                        layout[cursor : cursor + n_hits_per_rank[i], :] = vsrc
                    cursor += n_hits_per_rank[i]
                vdf.create_virtual_dataset(dname, layout, fillvalue=-1)

        vdf["entry_1/data_1/powderHits"] = powder_hits
        vdf["entry_1/data_1/powderMisses"] = powder_misses

    return vfname


def generate_libpressio_configuration(
    compressor: Literal["sz3", "qoz"],
    roi_window_size: int,
    bin_size: int,
    abs_error: float,
    libpressio_mask,
) -> Dict[str, Any]:
    """
    Create the configuration JSON for the libpressio library

    Parameters:

        compressor (Literal["sz3", "qoz"]): Compression algorithm to use
            ("qoz" or "sz3").

        abs_error (float): Bound value for the absolute error.

        bin_size (int): Bining Size.

        roi_window_size (int): Default size of the ROI window.

        libpressio_mask (NDArray): mask to be applied to the data.

    Returns:

        lp_json (Dict[str, Any]): Dictionary storing the JSON configuration structure
        for the libpressio library
    """

    if compressor == "qoz":
        pressio_opts: Dict[str, Any] = {
            "pressio:abs": abs_error,
            "qoz": {"qoz:stride": 8},
        }
    elif compressor == "sz3":
        pressio_opts = {"pressio:abs": abs_error}

    lp_json = {
        "compressor_id": "pressio",
        "early_config": {
            "pressio": {
                "pressio:compressor": "roibin",
                "roibin": {
                    "roibin:metric": "composite",
                    "roibin:background": "mask_binning",
                    "roibin:roi": "fpzip",
                    "background": {
                        "binning:compressor": "pressio",
                        "mask_binning:compressor": "pressio",
                        "pressio": {"pressio:compressor": compressor},
                    },
                    "composite": {
                        "composite:plugins": [
                            "size",
                            "time",
                            "input_stats",
                            "error_stat",
                        ]
                    },
                },
            }
        },
        "compressor_config": {
            "pressio": {
                "roibin": {
                    "roibin:roi_size": [roi_window_size, roi_window_size, 0],
                    "roibin:centers": None,  # "roibin:roi_strategy": "coordinates",
                    "roibin:nthreads": 4,
                    "roi": {"fpzip:prec": 0},
                    "background": {
                        "mask_binning:mask": None,
                        "mask_binning:shape": [bin_size, bin_size, 1],
                        "mask_binning:nthreads": 4,
                        "pressio": pressio_opts,
                    },
                }
            }
        },
        "name": "pressio",
    }

    lp_json["compressor_config"]["pressio"]["roibin"]["background"][
        "mask_binning:mask"
    ] = (1 - libpressio_mask)

    return lp_json


def add_peaks_to_libpressio_configuration(lp_json, peaks) -> Dict[str, Any]:
    """
    Add peak infromation to libpressio configuration

    Parameters:

        lp_json: Dictionary storing the configuration JSON structure for the libpressio
            library.

        peaks (Any): Peak information as returned by psana.

    Returns:

        lp_json: Updated configuration JSON structure for the libpressio library.
    """
    lp_json["compressor_config"]["pressio"]["roibin"]["roibin:centers"] = (
        numpy.ascontiguousarray(numpy.uint64(peaks[:, [2, 1, 0]]))
    )
    return lp_json


class FindPeaksPyAlgos(Task):
    """
    Task that performs peak finding using the PyAlgos peak finding algorithms and
    writes the peak information to CXI files.
    """

    def __init__(self, *, params: TaskParameters) -> None:
        super().__init__(params=params)

    def _run(self) -> None:
        ds: Any = MPIDataSource(
            f"exp={self._task_parameters.lute_config.experiment}:"
            f"run={self._task_parameters.lute_config.run}:smd"
        )
        if self._task_parameters.n_events != 0:
            ds.break_after(self._task_parameters.n_events)

        det: Any = Detector(self._task_parameters.det_name)
        det.do_reshape_2d_to_3d(flag=True)

        evr: Any = Detector(self._task_parameters.event_receiver)

        i_x: Any = det.indexes_x(self._task_parameters.lute_config.run).astype(
            numpy.int64
        )
        i_y: Any = det.indexes_y(self._task_parameters.lute_config.run).astype(
            numpy.int64
        )
        ipx: Any
        ipy: Any
        ipx, ipy = det.point_indexes(
            self._task_parameters.lute_config.run, pxy_um=(0, 0)
        )

        alg: Any = None
        num_hits: int = 0
        num_events: int = 0
        num_empty_images: int = 0
        tag: str = self._task_parameters.tag
        if (tag != "") and (tag[0] != "_"):
            tag = "_" + tag

        evt: Any
        for evt in ds.events():

            evt_id: Any = evt.get(EventId)
            timestamp_seconds: int = evt_id.time()[0]
            timestamp_nanoseconds: int = evt_id.time()[1]
            timestamp_fiducials: int = evt_id.fiducials()
            event_codes: Any = evr.eventCodes(evt)

            if isinstance(self._task_parameters.pv_camera_length, float):
                clen: float = self._task_parameters.pv_camera_length
            else:
                clen = (
                    ds.env().epicsStore().value(self._task_parameters.pv_camera_length)
                )

            if self._task_parameters.event_logic:
                if not self._task_parameters.event_code in event_codes:
                    continue

            img: Any = det.calib(evt)

            if img is None:
                num_empty_images += 1
                continue

            if alg is None:
                det_shape: Tuple[int, ...] = img.shape
                if len(det_shape) == 3:
                    det_shape = (det_shape[0] * det_shape[1], det_shape[2])
                else:
                    det_shape = img.shape

                mask: NDArray[numpy.uint16] = numpy.ones(det_shape).astype(numpy.uint16)

                if self._task_parameters.psana_mask:
                    mask = det.mask(
                        self.task_parameters.run,
                        calib=False,
                        status=True,
                        edges=False,
                        centra=False,
                        unbond=False,
                        unbondnbrs=False,
                    ).astype(numpy.uint16)

                hdffh: Any
                if self._task_parameters.mask_file is not None:
                    with h5py.File(self._task_parameters.mask_file, "r") as hdffh:
                        loaded_mask: NDArray[numpy.int] = hdffh["entry_1/data_1/mask"][
                            :
                        ]
                        mask *= loaded_mask.astype(numpy.uint16)

                file_writer: CxiWriter = CxiWriter(
                    outdir=self._task_parameters.outdir,
                    rank=ds.rank,
                    exp=self._task_parameters.lute_config.experiment,
                    run=self._task_parameters.lute_config.run,
                    n_events=self._task_parameters.n_events,
                    det_shape=det_shape,
                    i_x=i_x,
                    i_y=i_y,
                    ipx=ipx,
                    ipy=ipy,
                    min_peaks=self._task_parameters.min_peaks,
                    max_peaks=self._task_parameters.max_peaks,
                    tag=tag,
                )
                alg: Any = PyAlgos(mask=mask, pbits=0)  # pbits controls verbosity
                alg.set_peak_selection_pars(
                    npix_min=self._task_parameters.npix_min,
                    npix_max=self._task_parameters.npix_max,
                    amax_thr=self._task_parameters.amax_thr,
                    atot_thr=self._task_parameters.atot_thr,
                    son_min=self._task_parameters.son_min,
                )

                if self._task_parameters.compression is not None:

                    libpressio_config = generate_libpressio_configuration(
                        compressor=self._task_parameters.compression.compressor,
                        roi_window_size=self._task_parameters.compression.roi_window_size,
                        bin_size=self._task_parameters.compression.bin_size,
                        abs_error=self._task_parameters.compression.abs_error,
                        libpressio_mask=mask,
                    )

                powder_hits: NDArray[numpy.float_] = numpy.zeros(det_shape)
                powder_misses: NDArray[numpy.float_] = numpy.zeros(det_shape)

            peaks: Any = alg.peak_finder_v3r3(
                img,
                rank=self._task_parameters.peak_rank,
                r0=self._task_parameters.r0,
                dr=self._task_parameters.dr,
                #      nsigm=self._task_parameters.nsigm,
            )

            num_events += 1

            if (peaks.shape[0] >= self._task_parameters.min_peaks) and (
                peaks.shape[0] <= self._task_parameters.max_peaks
            ):

                if self._task_parameters.compression is not None:

                    libpressio_config_with_peaks = (
                        add_peaks_to_libpressio_configuration(libpressio_config, peaks)
                    )
                    compressor = PressioCompressor.from_config(
                        libpressio_config_with_peaks
                    )
                    compressed_img = compressor.encode(img)
                    decompressed_img = numpy.zeros_like(img)
                    decompressed = compressor.decode(compressed_img, decompressed_img)
                    img = decompressed_img

                try:
                    photon_energy: float = (
                        Detector("EBeam").get(evt).ebeamPhotonEnergy()
                    )
                except AttributeError:
                    photon_energy = (
                        1.23984197386209e-06
                        / ds.env().epicsStore().value("SIOC:SYS0:ML00:AO192")
                        / 1.0e9
                    )

                file_writer.write_event(
                    img=img,
                    peaks=peaks,
                    timestamp_seconds=timestamp_seconds,
                    timestamp_nanoseconds=timestamp_nanoseconds,
                    timestamp_fiducials=timestamp_fiducials,
                    photon_energy=photon_energy,
                )
                num_hits += 1

            # TODO: Fix bug here
            # generate / update powders
            if peaks.shape[0] >= self._task_parameters.min_peaks:
                powder_hits = numpy.maximum(powder_hits, img)
            else:
                powder_misses = numpy.maximum(powder_misses, img)

        if num_empty_images != 0:
            msg: Message = Message(
                contents=f"Rank {ds.rank} encountered {num_empty_images} empty images."
            )
            self._report_to_executor(msg)

        file_writer.write_non_event_data(
            powder_hits=powder_hits,
            powder_misses=powder_misses,
            mask=mask,
            clen=clen,
        )

        file_writer.optimize_and_close_file(
            num_hits=num_hits, max_peaks=self._task_parameters.max_peaks
        )

        COMM_WORLD.Barrier()

        num_hits_per_rank: List[int] = COMM_WORLD.gather(num_hits, root=0)
        num_hits_total: int = COMM_WORLD.reduce(num_hits, SUM)
        num_events_per_rank: List[int] = COMM_WORLD.gather(num_events, root=0)

        if ds.rank == 0:
            master_fname: Path = write_master_file(
                mpi_size=ds.size,
                outdir=self._task_parameters.outdir,
                exp=self._task_parameters.lute_config.experiment,
                run=self._task_parameters.lute_config.run,
                tag=tag,
                n_hits_per_rank=num_hits_per_rank,
                n_hits_total=num_hits_total,
            )

            # Write final summary file
            f: TextIO
            with open(
                Path(self._task_parameters.outdir) / f"peakfinding{tag}.summary", "w"
            ) as f:
                print(f"Number of events processed: {num_events_per_rank[-1]}", file=f)
                print(f"Number of hits found: {num_hits_total}", file=f)
                print(
                    "Fractional hit rate: "
                    f"{(num_hits_total/num_events_per_rank[-1]):.2f}",
                    file=f,
                )
                print(f"No. hits per rank: {num_hits_per_rank}", file=f)

            with open(Path(self._task_parameters.out_file), "w") as f:
                print(f"{master_fname}", file=f)

            # Write out_file

    def _post_run(self) -> None:
        super()._post_run()
        self._result.task_status = TaskStatus.COMPLETED
