"""Starts Zmq process to send xtc1 file.

Based on Mona's converter from https://github.com/monarin/xtc1to2
"""

import argparse
import csv
import numpy as np
import psana
import pickle
from typing import Any, Dict, List, Tuple, Union
import zlib
import zmq

from PSCalib.GeometryAccess import GeometryAccess

class PsanaImg:

    def __init__(
        self,
        exp: str,
        run: str,
        mode: str,
        detector_name: str,
        reshape: str,
    ) -> None:
        """
        It serves as an image accessing layer based on the data management system
        psana in LCLS.

        Args:
            exp (str): Experiment's name
            run (str): Run number
            mode (str): Mode
            detector_name (str): Name of the detector
            reshape (str): True/False whether to reshape 2d arrays to 3d - Chuck
        """

        # Set up data source
        self.datasource_id: str = f"exp={exp}:run={run}:{mode}"
        self.datasource: psana._Datasource = psana.DataSource(self.datasource_id)
        self.run_current: psana.Run = next(self.datasource.runs())
        self.timestamps: Tuple[psana.EventTime, ...] = self.run_current.times()

        # Set up detector
        self.detector: psana.Detector.AreaDetector.AreaDetector = psana.Detector(
            detector_name
        )

        # set flag (for Chuck)
        if bool(reshape):
            self.detector.do_reshape_2d_to_3d(flag=True)


    def get(self, event_num: Union[int, str], calib: bool=False) -> np.ndarray:
        # Fetch the timestamp according to event number
        timestamp: psana.EventTime = self.timestamps[int(event_num)]

        # Access each event based on timestamp
        event: psana.Event = self.run_current.event(timestamp)

        # Fetch image data based on timestamp from detector
        if calib:
            img: np.ndarray = self.detector.calib(event)
        else:
            img: np.ndarray = self.detector.image(event)
        return img

    def timestamp(self, event_num: Union[str, int]) -> psana.EventTime:
        ts: psana.EventTime = self.timestamps[int(event_num)]
        return ts


class PsanaPhotonEnergy:

    def __init__(self, exp: str, run: str, mode: str) -> None:
        """
        Uses psana1 ebeam and epicsStore to retrieve photon energy.
        """
        # Set up data source
        self.datasource_id: str = f"exp={exp}:run={run}:{mode}"
        self.datasource: psana.DataSource = psana.DataSource(self.datasource_id)
        self.run_current: psana.Run = next(self.datasource.runs())
        self.timestamps: tuple = self.run_current.times()

        # Set up detector and epicsStore
        self.ebeam_det: psana.Detector = psana.Detector("EBeam")
        self.es: psana.EpicsStore = self.datasource.env().epicsStore()

    def get(self, event_num: Union[int, str]) -> float:

        # Access each event based on timestamp
        timestamp: psana.EventTime = self.timestamps[int(event_num)]
        event: psana.Event = self.run_current.event(timestamp)

        # Try to get photon energy from ebeam
        ebeam: psana.Bld.BldDataEBeamV4 = self.ebeam_det.get(event)
        try:
            photon_energy: float = ebeam.ebeamPhotonEnergy()
        except AttributeError:
            photon_energy: float = 0.0

        return photon_energy


class PsanaGeometry:

    pixel_position = None
    pixel_index_map = None

    def __init__(self, geom: str) -> None:
        """A getter that reads in lcls1-style geometry file.
        Use this access info from  geometry file (*-end.data).
        Available info is set as class attributes.
        """
        cframe: int = 0  # fixed to psana style (1 is for lab conventions)
        geometry: PSCalib.GeometryAccess.GeometryAccess = GeometryAccess(
            geom,
            cframe=cframe
        )

        # Stores a tuple of x,y, and z coordinate arrays
        pixel_coords: tuple = geometry.get_pixel_coords(cframe=cframe)
        pixel_coord_indexes: tuple = geometry.get_pixel_coord_indexes(cframe=cframe)

        # Converts from microns to meters
        temp: List[np.ndarray[Any, np.dtype[np.float64]]] = [
            np.asarray(t) * 1e-6 for t in pixel_coords
        ]
        temp_index: List[np.ndarray[Any, np.dtype[np.float64]]] = [
            np.asarray(t) for t in pixel_coord_indexes
        ]

        # The shape of each axis is represented by five numbers (for this det)
        # e.g. (1,2,2,512,512). We calculate no. of panels by multiplying
        # all numbers except the last two (#pixel_x, #pixel_y).
        panel_num: int = np.prod(temp[0].shape[:-2])

        shape: tuple = (panel_num, temp[0].shape[-2], temp[0].shape[-1])
        pixel_position: np.ndarray = np.zeros(shape + (3,))     # x,y,z
        pixel_index_map: np.ndarray = np.zeros(shape + (2,))    # x,y

        for n in range(3):
            pixel_position[..., n]: np.ndarray = temp[n].reshape(shape)

        for n in range(2):
            pixel_index_map[..., n]: np.ndarray = temp_index[n].reshape(shape)

        pixel_index_map: np.ndarray = pixel_index_map.astype(np.int16)
        pixel_position: np.ndarray = pixel_position.astype(np.float32)

        self.pixel_position: np.ndarray = pixel_position
        self.pixel_index_map: np.ndarray = pixel_index_map


class ZmqSender:

    zmq_socket = None

    def __init__(self, socket: zmq.Context.socket) -> None:
        """
        A helper for sending messages using pyzmq.
        """
        context: zmq.Context = zmq.Context()
        self.zmq_socket: zmq.Context.socket = context.socket(zmq.PUSH)
        self.zmq_socket.bind(socket)

    def send_zipped_pickle(
        self,
        obj: dict,
        flags: int = 0,
        protocol: int = -1
    ) -> bool:
        """Pickle an object, and zip the pickle before sending it"""
        try:
            p: bytes = pickle.dumps(obj, protocol)
            z: bytes = zlib.compress(p)
            return self.zmq_socket.send(z, flags=flags)
        except (pickle.PickleError, TypeError, zlib.error, zmq.ZMQError) as e:
            print(f"[XTC1 Sender]: Error during sending pickled object: {e}")

    def send_array(
        self,
        data: np.ndarray,
        flags: int = 0,
        copy: bool = True,
        track: bool = False
    ) -> bool:
        """
        Send a NumPy array with metadata (dtype and shape) over a ZeroMQ socket.
    
        Parameters:
            A (np.ndarray): Array to send.
            flags (int): ZMQ flags (e.g., zmq.SNDMORE).
            copy (bool): Whether to copy the message.
            track (bool): Whether to track the message.

        Returns:
            bool: True if the message was successfully sent, False otherwise.
        """
        try:
            md: dict[str, Any] = {
                "dtype": str(data.dtype),
                "shape": data.shape,
            }

            # Send metadata then the actual array
            self.zmq_socket.send_json(md, flags | zmq.SNDMORE)
            return self.zmq_socket.send(data, flags, copy=copy, track=track)

        except (AttributeError, zmq.ZMQError, TypeError, ValueError) as e:
            print(f"[XTC1 Sender]: Error, failed to send array: {e}")


    def close(self) -> None:
        """Closes the zmq socket"""
        self.zmq_socket.close()

if __name__ == "__main__":


    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        prog="Xtc1 reader",
        description="Read in Xtc1 files using psana1"
    )
    parser.add_argument(
        "-e", "--exp",
        type=str,
        help="Experiment's name",
        default="xpptut15",
    )
    parser.add_argument(
        "-r",
        "--run",
        type=str,
        help="Run number",
        default="291",
    )
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        help="Mode",
        default="idx",
    )
    parser.add_argument(
        "-d",
        "--detector",
        type=str,
        help="Detector name",
        default="Camp.0:pnCCD.0",
    )
    parser.add_argument(
        "-g",
        "--geometry",
        type=str,
        help="Geometry file",
        default=("/sdf/data/lcls/ds/xpp/xpptut15/calib/PNCCD::CalibV1/"
            "Camp.0:pnCCD.0/geometry/290-292.data"
        ),
    )
    parser.add_argument(
        "-f",
        "--eventfile",
        type=str,
        help="File with the event numbers",
        default="",
    )
    parser.add_argument(
        "-s",
        "--reshape",
        type=str,
        help="Reshape 2d detector images to 3d - Chuck",
        default="",
    )
    parser.add_argument(
        "-v",
        "--verify",
        type=str,
        help="Verify data at the end - only for small datasets that fit in memory",
        default="True",
    )

    args: argparse.Namespace = parser.parse_args()

    img_reader: PsanaImg = PsanaImg(args.exp, args.run, args.mode, args.detector, args.reshape)
    phe_reader: PsanaPhotonEnergy = PsanaPhotonEnergy(args.exp, args.run, args.mode)
    gmt_reader: PsanaGeometry = PsanaGeometry(args.geometry)

    socket: str = "tcp://127.0.0.1:5557"
    zmq_send: ZmqSender = ZmqSender(socket)

    # If the eventfile is presented select those, otherwise use all events
    event_num_list: List[int]

    if args.eventfile == "":
        # All events
        tss: List = img_reader.timestamps
        event_num_list = list(range(len(tss)))
    else:
        event_num_list = []
        with open(args.eventfile, newline='') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            for row in csvreader:
                event_num_list += list(map(int, row))

    # Test override:
    event_num_list = [290, 291]

    total_events: int = len(event_num_list)

    zmq_send.zmq_socket.send_string(str(total_events))

    verify: bool = bool(args.verify)
    if verify:
        data_array: np.ndarray = np.zeros([total_events,4,512,512], dtype=np.float32)
        photon_array: np.ndarray = np.zeros(total_events, dtype=np.float64)
    else:
        data_array: np.ndarray = np.zeros([4,512,512], dtype=np.float32)

    for i, event_num in enumerate(event_num_list):
        img: np.ndarray = img_reader.get(event_num, calib=True)
        photon_energy: float = phe_reader.get(event_num)
        ts: psana.EventTime = img_reader.timestamp(event_num)

        if verify:
            data_array[i,:,:,:] = img
            photon_array[i] = photon_energy

        if i == 0:
            # Send beginning timestamp - this will create config, beginrun,
            # beginstep, and enable on the client.
            start_dict: Dict[str, Any] = {
                "start": True, 
                "config_timestamp": ts.time() - 10,
                "pixel_position": gmt_reader.pixel_position,
                "pixel_index_map": gmt_reader.pixel_index_map,
            }
            print("[XTC1 Sender]: Starting sending..")
            zmq_send.send_zipped_pickle(start_dict)

        data: Dict[str, Any] = {
            "calib": img,
            "photon_energy": photon_energy,
            "timestamp": ts.time(),
        }

        print(
            f"[XTC1 Sender]: event_num={event_num} ts={ts.time()} img={img.shape} "
            f"dtype={img.dtype} photon energy:{photon_energy:.3f}"
        )

        # Send the dataset
        zmq_send.send_zipped_pickle(data)

    # Send end message
    done_dict: Dict[str, bool] = {"end": True}
    zmq_send.send_zipped_pickle(done_dict)
    print("[XTC1 Sender]: Sending complete!")

    zmq_send.close()

    if verify:
        import h5py

        with h5py.File("/sdf/scratch/users/k/kmecseki/out.hdf5", "w") as f:
            f.create_dataset("pixel_position", data=gmt_reader.pixel_position)
            f.create_dataset("pixel_index_map", data=gmt_reader.pixel_index_map)
            f.create_dataset("data", data=data_array)
            f.create_dataset("photon_energy", data=photon_array)
