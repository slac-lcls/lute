import argparse
import h5py # type: ignore
import numpy as np
import pickle
from typing import Any, BinaryIO
import zmq
import zlib

from psana import DataSource # type: ignore
from psana.dgramedit import DgramEdit, AlgDef, DetectorDef # type: ignore
from psana.psexp import TransitionId # type: ignore


class ZmqReceiver:

    def __init__(self, socket: str) -> None:
        """
        A helper for receiving messages using pyzmq.
        Bind to socket (e.g. tcp://127.0.0.1:5557) with PULL
        """
        context: zmq.Context = zmq.Context()
        self.zmq_socket: zmq.Socket = context.socket(zmq.PULL)
        self.zmq_socket.connect(socket)

    def recv_zipped_pickle(self, flags: int = 0, protocol: int = -1) -> Any:
        """Receive zipped pickle"""
        z: bytes = self.zmq_socket.recv(flags)
        p: bytes = zlib.decompress(z)
        return pickle.loads(p)

    def recv_array(
        self, md: Any, flags: int = 0, copy: bool = True, track: bool = False
    ) -> None:
        """Receive a numpy array"""
        msg: Any = self.zmq_socket.recv(flags=flags, copy=copy, track=track)
        buf: Any = memoryview(msg)
        temp: Any = np.frombuffer(buf, dtype=md["dtype"])
        return temp.reshape(md["shape"])

    def close(self):
        """Close zmq socket"""
        self.zmq_socket.close()


def test_output(num_events: int, resolution: str) -> None:
    """
    Psana1 reader saves the content as a hdf5, we compare this file to the data that we received.
    """
    print("[XTC2 Writer]: Testing")
    try:
        f: h5py._hl.files.File = h5py.File(
            "/sdf/scratch/users/k/kmecseki/out.hdf5", "r"
        )
        pixel_position: h5py._hl.dataset.Dataset = f["pixel_position"]
        pixel_index_map: h5py._hl.dataset.Dataset = f["pixel_index_map"]
        data: h5py._hl.dataset.Dataset = f["data"]
        photon_energy: h5py._hl.dataset.Dataset = f["photon_energy"]

        ds: DataSource = DataSource(files="/sdf/scratch/users/k/kmecseki/out.xtc2")
        run: Any = next(ds.runs())
        det: Any = run.Detector("xpppnccd")

        pp_det: Any = run.Detector("pixel_position")
        pim_det: Any = run.Detector("pixel_index_map")

        channels: int
        res_x: int
        res_y: int
        channels, res_x, res_y = map(int, resolution.split("x"))
        data_array: np.ndarray = np.zeros(
            [num_events, channels, res_x, res_y], dtype=np.float32
        )
        photon_array: np.ndarray = np.zeros(num_events, dtype=np.float64)

        for i, evt in enumerate(run.events()):
            data_array[i, :, :, :] = det.raw.calib(evt)
            photon_array[i] = det.raw.photon_energy(evt)
            pixel_position_array = pp_det(evt)
            pixel_index_map_array = pim_det(evt)
        assert np.array_equal(data, data_array)
        assert np.array_equal(photon_energy, photon_array)
        assert np.array_equal(pixel_position, pixel_position_array)
        assert np.array_equal(pixel_index_map, pixel_index_map_array)
        print("[XTC2 writer]: All test passed successfully")

    except (OSError, IOError) as e:
        print(f"Error opening hdf5 file: {e}")


def save_dgramedit(dg_edit: DgramEdit, outbuf: bytearray, outfile: BinaryIO) -> None:
    """Save dgram edit to output buffer and write to file"""
    dg_edit.save(outbuf)
    outfile.write(outbuf[: dg_edit.size])


if __name__ == "__main__":

    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        prog="Xtc2 writer", description="Write received file as Xtc2 using psana2"
    )
    parser.add_argument(
        "-n",
        "--node-id",
        type=str,
        help="Node ID for the detector",
        default="1",
    )
    parser.add_argument(
        "-l",
        "--resolution",
        type=str,
        help="Detector channels and resolution in the format: CxRxR",
        default="4x512x512",
    )
    parser.add_argument(
        "-v",
        "--verify",
        type=str,
        help="Verify data at the end - only for small datasets that fit in memory",
        default="1",
    )
    args: argparse.Namespace = parser.parse_args()

    # TEMP # Do we want to parametrize this?
    namesId = {"xpppnccd": 0, "runinfo": 1, "scan": 2}

    # Setup socket for zmq connection
    socket: str = "tcp://127.0.0.1:5557"
    zmq_recv: ZmqReceiver = ZmqReceiver(socket)

    # Allocating memory for DgramEdit output buffer
    MEMSIZE: int = 64000000
    outbuf: bytearray = bytearray(MEMSIZE)

    # Open output file for writing
    ofname: str = "/sdf/scratch/users/k/kmecseki/out.xtc2"
    xtc2file: BinaryIO = open(ofname, "wb")

    # Create config, algorithm, and detector
    config: DgramEdit = DgramEdit(transition_id=TransitionId.Configure)

    alg: AlgDef = AlgDef("raw", 1, 2, 3)
    det: DetectorDef = DetectorDef("xpppnccd", "pnccd", "detnum1234")

    runinfo_alg: AlgDef = AlgDef("runinfo", 0, 0, 1)
    runinfo_det: DetectorDef = DetectorDef("runinfo", "runinfo", "")

    scan_alg: AlgDef = AlgDef("raw", 2, 0, 0)
    scan_det: DetectorDef = DetectorDef("scan", "scan", "detnum1234")

    # Define data formats
    datadef = {
        "calib": (np.float32, 3),
        "photon_energy": (np.float64, 0),
    }

    runinfodef = {
        "expt": (str, 1),
        "runnum": (np.uint32, 0),
    }

    scandef = {
        "pixel_position": (np.float32, 4),
        "pixel_index_map": (np.int16, 4),
    }

    # Create detetors
    pnccd = config.Detector(
        det, alg, datadef, nodeId=int(args.node_id), namesId=namesId["xpppnccd"]
    )
    runinfo = config.Detector(
        runinfo_det,
        runinfo_alg,
        runinfodef,
        nodeId=int(args.node_id),
        namesId=namesId["runinfo"],
    )
    scan = config.Detector(
        scan_det,
        scan_alg,
        scandef,
        nodeId=int(args.node_id),
        namesId=namesId["scan"],
    )

    num_events = int(zmq_recv.zmq_socket.recv_string())
    # Start saving data
    print("[XTC2 Writer]: Starting receiving")
    while True:
        obj = zmq_recv.recv_zipped_pickle()
        # Begin timestamp is needed (we calculate this from the first L1Accept)
        # to set the correct timestamp for all transitions prior to the first L1.
        if "start" in obj:
            config_timestamp = obj["config_timestamp"]
            config.updatetimestamp(config_timestamp)
            save_dgramedit(config, outbuf, xtc2file)

            beginrun = DgramEdit(
                transition_id=TransitionId.BeginRun,
                config_dgramedit=config,
                ts=config_timestamp + 1,
            )
            runinfo.runinfo.expt = "xpptut15"
            runinfo.runinfo.runnum = 291
            beginrun.adddata(runinfo.runinfo)
            scan.raw.pixel_position = obj["pixel_position"]
            scan.raw.pixel_index_map = obj["pixel_index_map"]
            beginrun.adddata(scan.raw)
            save_dgramedit(beginrun, outbuf, xtc2file)

            beginstep = DgramEdit(
                transition_id=TransitionId.BeginStep,
                config_dgramedit=config,
                ts=config_timestamp + 2,
            )
            save_dgramedit(beginstep, outbuf, xtc2file)

            enable = DgramEdit(
                transition_id=TransitionId.Enable,
                config_dgramedit=config,
                ts=config_timestamp + 3,
            )
            save_dgramedit(enable, outbuf, xtc2file)
            current_timestamp = config_timestamp + 3

        elif "end" in obj:
            disable = DgramEdit(
                transition_id=TransitionId.Disable,
                config_dgramedit=config,
                ts=current_timestamp + 1,
            )
            save_dgramedit(disable, outbuf, xtc2file)
            current_timestamp = config_timestamp + 3
            endstep = DgramEdit(
                transition_id=TransitionId.EndStep,
                config_dgramedit=config,
                ts=current_timestamp + 2,
            )
            save_dgramedit(endstep, outbuf, xtc2file)
            endrun = DgramEdit(
                transition_id=TransitionId.EndRun,
                config_dgramedit=config,
                ts=current_timestamp + 3,
            )
            save_dgramedit(endrun, outbuf, xtc2file)
            break

        else:
            # Create L1Accept
            d0 = DgramEdit(
                transition_id=TransitionId.L1Accept,
                config_dgramedit=config,
                ts=obj["timestamp"],
            )
            pnccd.raw.calib = obj["calib"]
            pnccd.raw.photon_energy = obj["photon_energy"]
            d0.adddata(pnccd.raw)
            save_dgramedit(d0, outbuf, xtc2file)
            current_timestamp = obj["timestamp"]
    print("[XTC2 Writer]: Complete")
    xtc2file.close()
    zmq_recv.close()
    verify: bool = bool(int(args.verify))
    if verify:
        test_output(num_events, args.resolution)
