import argparse
import pickle
import zlib
from typing import Any, BinaryIO, Dict, List, Optional, Tuple, Type

import numpy as np
import zmq
from psana.dgramedit import DgramEdit, AlgDef, DetectorDef  # type: ignore
from psana.psexp import TransitionId  # type: ignore


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


def save_dgramedit(dg_edit: DgramEdit, outbuf: bytearray, outfile: BinaryIO) -> None:
    """Save dgram edit to output buffer and write to file"""
    dg_edit.save(outbuf)
    outfile.write(outbuf[: dg_edit.size])


if __name__ == "__main__":

    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        prog="Xtc2 writer", description="Write received file as Xtc2 using psana2"
    )
    parser.add_argument(
        "-d", "--detector", type=str, help="Input detector (in the XTC1 file.)"
    )
    parser.add_argument("-e", "--experiment", type=str, help="The experiment name.")
    parser.add_argument("-f", "--filename", type=str, help="Output XTC2 filename.")
    parser.add_argument(
        "-n",
        "--node-id",
        type=str,
        help="Node ID for the detector",
        default="1",
    )
    parser.add_argument("-r", "--run", type=int, help="The experiment run number.")
    args: argparse.Namespace = parser.parse_args()

    detnames: List[str] = args.detector.split(",")
    all_detnames: List[str] = detnames + ["runinfo", "epicsinfo", "scan"]

    namesId: Dict[str, int] = {}
    for idx, detname in enumerate(all_detnames):
        namesId[detname] = idx

    # Setup socket for zmq connection
    socket: str = "tcp://127.0.0.1:5557"
    zmq_recv: ZmqReceiver = ZmqReceiver(socket)

    # Allocating memory for DgramEdit output buffer
    MEMSIZE: int = 640000000
    outbuf: bytearray = bytearray(MEMSIZE)

    # Open output file for writing
    xtc2file: BinaryIO = open(args.filename, "wb")

    # Create config, algorithm, and detector
    config: DgramEdit = DgramEdit(transition_id=TransitionId.Configure)

    generic_det_alg: AlgDef = AlgDef("xtc1dump", 0, 1, 0)

    det_defs: Dict[str, DetectorDef] = {}
    serial_num: str = "detnum1"
    counter: int = 2
    for detname in detnames:
        det_def: DetectorDef = DetectorDef(detname, "generic_container", serial_num)
        serial_num += str(counter)
        counter += 1
        det_defs[detname] = det_def

    runinfo_alg: AlgDef = AlgDef("runinfo", 0, 0, 1)
    runinfo_det: DetectorDef = DetectorDef("runinfo", "runinfo", "")

    scan_alg: AlgDef = AlgDef("raw", 2, 0, 0)
    scan_det: DetectorDef = DetectorDef("scan", "scan", "detnum1234")

    # Hold calibration information
    calib_serial_num: str = "detnum"
    calib_detectors: Dict[str, DetectorDef] = {}

    # Define data formats

    runinfodef: Dict[str, Tuple[Type, int]] = {
        "expt": (str, 1),
        "runnum": (np.uint32, 0),
    }

    scandef: Dict[str, Tuple[Type, int]] = {}
    num_events = int(zmq_recv.zmq_socket.recv_string())

    epics_alg: AlgDef = AlgDef("raw", 2, 0, 0)
    epics_det: DetectorDef = DetectorDef("epics", "epics", "detnum1234")
    epics_def: Dict[str, Tuple[Type, int]] = {}

    epicsinfo_det: DetectorDef = DetectorDef("epicsinfo", "epicsinfo", "detnum1234")
    epicsinfo_alg: AlgDef = AlgDef("epicsinfo", 1, 0, 0)
    epicsinfo_def: Dict[str, Tuple[Type, int]] = {"keys": (str, 1)}
    epicsinfo: Optional[config.Detector] = None
    # This will be sent before anything else - contains rank and type
    # of all the information to be stored for the detector
    # detname: {field: (type, rank)}
    datadef: Optional[Dict[str, Dict[str, Tuple[Type, int]]]] = None
    # Start saving data
    print("[XTC2 Writer]: Starting receiving")
    detector: config.Detector
    detectors: Dict[str, config.Detector] = {}
    namesId["epics"] = len(namesId)
    while True:
        obj = zmq_recv.recv_zipped_pickle()
        # Begin timestamp is needed (we calculate this from the first L1Accept)
        # to set the correct timestamp for all transitions prior to the first L1.
        if "DATA_TYPE_INFO" in obj:
            datadef = obj["DATA_TYPE_INFO"]
            # detname: {
            #    "calib": (np.float32, 3),
            #    "photon_energy": (np.float64, 0),
            # }
            # Create detetors
            assert datadef is not None
            for detname in datadef:
                if detname in detnames:
                    detector = config.Detector(
                        det_defs[detname],
                        generic_det_alg,
                        datadef[detname],
                        nodeId=int(args.node_id),
                        namesId=namesId[detname],
                    )
                    detectors[detname] = detector
                elif "_calib" in detname:
                    # Constants are too big... We will split them up and attach them to
                    # indiviudal SlowUpdate datagrams as "epics" detectors
                    calib_type_info: Dict[str, Tuple[Type, int]] = datadef[detname]
                    for const_name, const_type in calib_type_info.items():
                        prefixed_name: str = (
                            f"{detname.replace('_calib','')}_{const_name}"
                        )
                        calib_serial_num += str(len(calib_detectors))
                        calib_epics_det: DetectorDef = DetectorDef(
                            "epics", "epics", calib_serial_num
                        )
                        namesId[prefixed_name] = len(namesId)
                        detector = config.Detector(
                            calib_epics_det,
                            epics_alg,
                            {prefixed_name: const_type},
                            nodeId=int(args.node_id),
                            namesId=namesId[prefixed_name],
                        )
                        epicsinfo_def.update({prefixed_name: (str, 1)})
                        calib_detectors[prefixed_name] = detector
            runinfo = config.Detector(
                runinfo_det,
                runinfo_alg,
                runinfodef,
                nodeId=int(args.node_id),
                namesId=namesId["runinfo"],
            )
            epicsinfo = config.Detector(
                epicsinfo_det,
                epicsinfo_alg,
                epicsinfo_def,
                nodeId=int(args.node_id),
                namesId=namesId["epicsinfo"],
            )
        elif "start" in obj:
            config_timestamp = obj["config_timestamp"]
            config.updatetimestamp(config_timestamp)

            if epicsinfo is not None:
                for calib_name in calib_detectors.keys():
                    setattr(epicsinfo.epicsinfo, calib_name, calib_name)
                epicsinfo.epicsinfo.keys = "epicsname"
                config.adddata(epicsinfo.epicsinfo)

            save_dgramedit(config, outbuf, xtc2file)

            beginrun = DgramEdit(
                transition_id=TransitionId.BeginRun,
                config_dgramedit=config,
                ts=config_timestamp + 1,
            )
            runinfo.runinfo.expt = args.experiment
            runinfo.runinfo.runnum = args.run
            beginrun.adddata(runinfo.runinfo)
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
            if "calib_const" in obj:
                for detname, det_consts in obj["calib_const"].items():
                    if detname == "timestamp":
                        continue
                    keys: List[str] = list(det_consts.keys())
                    for const_name, constants in det_consts.items():
                        prefixed_name = f"{detname}_{const_name}"
                        detector = calib_detectors[prefixed_name]
                        slow_update: DgramEdit = DgramEdit(
                            transition_id=TransitionId.SlowUpdate,
                            config_dgramedit=config,
                            ts=current_timestamp + 1,
                        )
                        setattr(detector.raw, prefixed_name, constants)
                        slow_update.adddata(detector.raw)
                        save_dgramedit(slow_update, outbuf, xtc2file)
                        current_timestamp += 1

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
            for detname in obj:
                if detname == "timestamp":
                    continue
                detector = detectors[detname]
                for attr in obj[detname]:
                    setattr(detector.xtc1dump, attr, obj[detname][attr])
                d0.adddata(detector.xtc1dump)
            save_dgramedit(d0, outbuf, xtc2file)
            current_timestamp = obj["timestamp"]
    print("[XTC2 Writer]: Complete")
    xtc2file.close()
    zmq_recv.close()
