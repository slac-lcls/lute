"""
Task for converting xtc1 files to xtc2 format using zmq-based communication
between psana1 and psana2 environments.

This module includes the classes needed to read XTC1 files and process them.

Classes:
    - ReadXtc1(Task): Read XTC1 files and transmit them to an WriteXtc2 Task
        for conversion.

Based on Mona's converter:
    https://github.com/monarin/xtc1to2
"""

import csv
import logging
import pickle
import time
import zlib
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    TypedDict,
    Union,
    cast,
)


import numpy as np
import numpy.typing as npt
import psana  # type: ignore
import zmq
from mpi4py import MPI
from PSCalib.GeometryAccess import GeometryAccess  # type: ignore

from lute.execution.logging import get_logger
from lute.io.models.xtc import ReadXtc1Parameters
from lute.tasks.dataclasses import TaskStatus
from lute.tasks.task import Task

logger: logging.Logger = get_logger(__name__)


class ZmqSender:
    def __init__(self, addr: str) -> None:
        """
        A helper for sending messages using pyzmq.

        Args:
            addr (str): Socket string e.g. "tcp://127.0.0.1:5557"
        """
        self._zmq_context: zmq.Context = zmq.Context()
        self.zmq_socket: zmq.sugar.socket.Socket = self._zmq_context.socket(zmq.PUSH)

        self.zmq_socket.connect(addr)

    def send_zipped_pickle(self, obj: dict, flags: int = 0, protocol: int = -1) -> None:
        """Pickle an object, and zip the pickle before sending it"""

        try:
            p: bytes = pickle.dumps(obj, protocol)
            z: bytes = zlib.compress(p)
            self.zmq_socket.send(z, flags=flags)
        except (pickle.PickleError, TypeError, zlib.error, zmq.ZMQError) as e:
            logger.error(f"[XTC1 Sender]: Error during sending pickled object: {e}")

    def send_array(
        self, data: np.ndarray, flags: int = 0, copy: bool = True, track: bool = False
    ) -> None:
        """
        Send a NumPy array with metadata (dtype and shape) over a ZeroMQ socket.

        Args:
            data (np.ndarray): Array to send.

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
            self.zmq_socket.send(data, flags, copy=copy, track=track)

        except (AttributeError, zmq.ZMQError, TypeError, ValueError) as e:
            logger.error(f"[XTC1 Sender]: Error, failed to send array: {e}")

    def close(self) -> None:
        """Closes the zmq socket"""
        self.zmq_socket.close()


class DataSpec(TypedDict):
    xtc2_attr_name: str
    object_name: str
    object_type: str
    object_field_name: Union[str, Tuple[str, str]]


def get_geometry(
    geometry: GeometryAccess,
) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.uint16]]:
    """Return the pixel coordinates and index map.


    Args:
        geometry (GeometryAccess): A GeometryAccess object. This can be created
            for a detector (that has geometry) by calling geometry = det.geometry(evt).

    Returns:
        pixel_position (npt.NDArray[np.float32]): Pixel position array.

        pixel_index_map (npt.NDArray[np.uint16]): Pixel index array.
    """
    cframe: int = 0
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
    panel_num: np.integer = np.prod(temp[0].shape[:-2])

    shape: tuple = (panel_num, temp[0].shape[-2], temp[0].shape[-1])
    pixel_position: npt.NDArray[np.float32] = np.zeros(
        shape + (3,), dtype=np.float32
    )  # x,y,z
    pixel_index_map: npt.NDArray[np.uint16] = np.zeros(
        shape + (2,), dtype=np.uint16
    )  # x,y

    for n in range(3):
        pixel_position[..., n] = temp[n].reshape(shape).astype(np.float32)

    for n in range(2):
        pixel_index_map[..., n] = temp_index[n].reshape(shape).astype(np.uint16)

    return pixel_position, pixel_index_map


def get_data(data_specs: List[DataSpec], evt: psana.Event) -> Any:
    data: Dict[str, Any] = {}
    for data_spec in data_specs:
        obj_type: Type = eval(data_spec["object_type"])
        obj: object = obj_type(data_spec["object_name"])
        # Because of round trip through JSON, this will be a list
        # hence the cast. True type is tuple
        field_name: Union[str, List[str]] = cast(
            Union[str, List[str]], data_spec["object_field_name"]
        )
        obj_field: Callable[[psana.Event], Any]
        attr_name: str = data_spec["xtc2_attr_name"]
        if isinstance(field_name, list):
            obj_field = getattr(obj, field_name[0])
            data[attr_name] = getattr(obj_field(evt), field_name[1])()
        else:
            obj_field = getattr(obj, field_name)
            data[attr_name] = obj_field(evt)
    return data


def get_calib_constants(data_specs: List[DataSpec], evt: psana.Event) -> Any:
    data: Dict[str, Any] = {}
    for data_spec in data_specs:
        # We assume currently that it is the same object in all specs
        # so only need to loop once
        obj_type: Type = eval(data_spec["object_type"])
        obj: object = obj_type(data_spec["object_name"])
        if hasattr(obj, "pedestals"):
            data["pedestals"] = obj.pedestals(evt)
        if hasattr(obj, "status"):
            data["pixel_status"] = obj.status(evt)
        if hasattr(obj, "status_as_mask"):
            data["mask"] = obj.status_as_mask(evt)
        if hasattr(obj, "gain"):
            data["gain"] = obj.gain(evt)
        if hasattr(obj, "geometry"):
            pixel_position, pixel_index_map = get_geometry(obj.geometry(evt))
            data["pixel_position"] = pixel_position
            data["pixel_index_map"] = pixel_index_map
        break
    return data if data else None


class ReadXtc1(Task):
    """
    A task that reads XTC1 files and sends them to a Task running in parallel to write
    them out as XTC2.
    (old psana and new psana) to convert (old) xtc1 files to (new) xtc2 files.

    This uses a ZMQ to communicate between the two Tasks run in parallel. Any startup
    information needed to get this communication up and running is transmitted via
    maestro RPC and multi-Task communication APIs.

    Args:
        params (ConvertXtc1to2Parameters): Configuration for the conversion task.
    """

    def __init__(
        self, *, params: ReadXtc1Parameters, use_mpi: bool = True, row_ids=None
    ) -> None:
        self._task_parameters: ReadXtc1Parameters
        super().__init__(params=params, use_mpi=use_mpi, row_ids=row_ids)

        self._mpi_rank: int = MPI.COMM_WORLD.Get_rank()
        self._mpi_size: int = MPI.COMM_WORLD.Get_size()

    def _run(self) -> None:
        par: ReadXtc1Parameters = self._task_parameters
        exp: str = par.lute_config.experiment
        run: Union[int, str] = par.lute_config.run
        logger.debug("Starting [XTC1 Sender] in psana 1")
        data_spec: Dict[str, Any] = {}
        detnames: List[str] = []
        for detname, specs in par.xtc1_access_pattern.items():
            det_specs: List[Any] = []
            for spec in specs:
                spec_d: Dict[str, Any]
                if isinstance(spec, dict):
                    # Case when first-party different env bootstrap
                    spec_d = spec
                elif hasattr(spec, "dict"):
                    # Case when first-party same env, no bootstrap
                    spec_d = spec.dict()
                else:
                    logger.error(
                        "Unable to interpret spec data! Will try to continue without it! Received: ",
                        spec,
                    )
                    continue
                det_specs.append(spec_d)
            data_spec[detname] = det_specs
            detnames.append(detname)

        # Retrieve Xtc2Writer information from maestro for the port number
        ##################################################################
        writer_port: Optional[int] = None
        writer_host: Optional[str] = None
        if self._mpi_rank == 0:
            logger.info(
                f"Querying Maestro for WriteXtc2 (exp={exp}, run={run}) port for rank {self._mpi_rank}"
            )
            while True:
                msg = self.get_running_tasks()
                if msg and isinstance(msg.contents, dict):
                    running_tasks = msg.contents.get("managed_tasks", [])
                    for task_info in running_tasks:
                        if (
                            task_info.get("task") == "WriteXtc2"
                            and task_info.get("xtc1_exp") == exp
                            and task_info.get("xtc1_run") == run
                        ):
                            port_key: str = "xtc1_zmq_port"
                            if port_key in task_info:
                                writer_port = task_info[port_key]
                                if (
                                    "task_hostnames" in task_info
                                    and task_info["task_hostnames"]
                                ):
                                    hostnames: List[str] = task_info["task_hostnames"]
                                    writer_host = hostnames[0]
                                break
                if writer_port is not None:
                    if writer_port == -1:
                        logger.info(
                            f"Writer rank {self._mpi_rank} exiting early as Reader exited."
                        )
                        self._result.task_status = TaskStatus.COMPLETED
                        break
                    if writer_host is not None:
                        break

                time.sleep(1)

        writer_port = MPI.COMM_WORLD.bcast(writer_port, root=0)
        writer_host = MPI.COMM_WORLD.bcast(writer_host, root=0)
        if writer_port == -1 or writer_port is None:
            self._result.task_status = TaskStatus.COMPLETED
            logger.info("Exiting without having sent data.")
            return

        # Sender will bind a random port and expose it
        addr: str = f"tcp://{writer_host}:{writer_port}"
        zmq_send: ZmqSender = ZmqSender(addr=addr)

        mode: str = "idx"
        datasource_id: str = f"exp={exp}:run={run}:{mode}"
        datasource: psana.DataSource = psana.DataSource(datasource_id)
        run_current: psana.Run = next(datasource.runs())
        timestamps: tuple = run_current.times()
        # If the eventfile is presented select those, otherwise use all events
        event_num_list: List[int]

        if not par.eventfile:
            # All events
            if not par.nevents:
                event_num_list = list(range(len(timestamps)))
            else:
                event_num_list = list(range(par.nevents))
        else:
            event_num_list = []
            try:
                with open(par.eventfile, newline="") as csvfile:
                    csvreader = csv.reader(csvfile, delimiter=",")
                    for row in csvreader:
                        event_num_list += list(map(int, row))
            except FileNotFoundError:
                logger.error(
                    f"Error: File not found: {par.eventfile}, using test numbers."
                )
                event_num_list = [290, 291]

        total_events: int = len(event_num_list)
        if self._mpi_rank >= total_events:
            self.publish_metadata(
                {
                    f"zmq_port_rank_{self._mpi_rank}": -1,
                    "exp": par.lute_config.experiment,
                    "run": par.lute_config.run,
                }
            )
            self._result.task_status = TaskStatus.COMPLETED
            logger.info(f"Reader rank {self._mpi_rank} exiting early. Too few events")
            return
        events_per_rank: int = total_events // self._mpi_size
        start_idx: int
        end_idx: int
        if events_per_rank == 0:
            start_idx = self._mpi_rank
            end_idx = self._mpi_rank + 1
        else:
            start_idx = self._mpi_rank * events_per_rank
            if self._mpi_rank == (self._mpi_size - 1):
                # Make sure to have the final rank pick up any remaining events
                # not evenly distributed by world size
                end_idx = total_events
            else:
                end_idx = (self._mpi_rank + 1) * events_per_rank

        event_num_list = event_num_list[start_idx:end_idx]

        data_type_info: Dict[str, Dict[str, Tuple[Type, int]]] = {}
        send_type_info: bool = True

        calib_dict: Dict[str, Any] = {}
        for i, event_num in enumerate(event_num_list):
            timestamp: psana.EventTime = timestamps[int(event_num)]
            event: psana.Event = run_current.event(timestamp)

            data: Dict[str, Any] = {}
            # Tag the data with the rank
            data["rank"] = self._mpi_rank

            for detname in data_spec:
                # detector data will be a dict of field_names to the data
                # (e.g. "calib": (1,2,3,...))
                detector_data: Dict[str, Any] = get_data(data_spec[detname], event)
                data[detname] = detector_data
                if send_type_info:
                    data_type_info[detname] = {}
                    for attr_name, field_data in detector_data.items():
                        dtype: Type
                        rank: int
                        if isinstance(field_data, np.ndarray):
                            dtype = field_data.dtype.type
                            rank = field_data.ndim
                        elif isinstance(field_data, float):
                            dtype = np.float64
                            rank = 0
                        elif isinstance(field_data, int):
                            dtype = np.int64
                            rank = 0
                        else:
                            dtype = type(field_data)
                            rank = 0
                        data_type_info[detname][attr_name] = (dtype, rank)
                    for detname in data_spec:
                        calib_consts: Any = get_calib_constants(
                            data_spec[detname], event
                        )
                        if calib_consts is not None:
                            calib_dict[detname] = calib_consts
                            calib_detname: str = f"{detname}_calib"
                            data_type_info[calib_detname] = {}
                            for attr_name, field_data in calib_consts.items():
                                if isinstance(field_data, np.ndarray):
                                    dtype = field_data.dtype.type
                                    rank = field_data.ndim
                                elif isinstance(field_data, float):
                                    dtype = np.float64
                                    rank = 0
                                elif isinstance(field_data, int):
                                    dtype = np.int64
                                    rank = 0
                                else:
                                    dtype = type(field_data)
                                    rank = 0
                                data_type_info[calib_detname][attr_name] = (dtype, rank)

            data["timestamp"] = timestamp.time()
            if send_type_info:
                zmq_send.send_zipped_pickle(
                    {
                        "DATA_TYPE_INFO": data_type_info,
                        "rank": self._mpi_rank,
                    }
                )
                send_type_info = False

            if i == 0:
                # Send beginning timestamp - this will create config, beginrun,
                # beginstep, and enable on the client.
                start_dict: Dict[str, Any] = {
                    "start": True,
                    "config_timestamp": timestamp.time() - 10,
                    "rank": self._mpi_rank,
                }
                if calib_dict:
                    # If the requested detectors had calibration constants they will
                    # be attached to the BeginRun transition as part of the scan det
                    start_dict["calib_const"] = calib_dict
                    logger.info("[XTC1 Sender]: Starting sending..")
                zmq_send.send_zipped_pickle(start_dict)

            logger.debug(f"[XTC1 Sender]: event_num={event_num} ts={timestamp.time()}")

            # Send the dataset
            zmq_send.send_zipped_pickle(data)

        # Send end message
        done_dict: Dict[str, Any] = {"end": True, "rank": self._mpi_rank}
        zmq_send.send_zipped_pickle(done_dict)
        logger.info("[XTC1 Sender]: Sending complete!")

        zmq_send.close()

        self._result.task_status = TaskStatus.COMPLETED
        logger.info("XTC1 Sending Completed completed.")
