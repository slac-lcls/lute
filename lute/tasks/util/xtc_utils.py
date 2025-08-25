"""Reads in Xtc1 file in psana1 environment

Based on Mona's converter from https://github.com/monarin/xtc1to2

Classes: 
    PsanaImg: Image access layer in psana 1.
    
    PhotonEnergy: Retrieves photon energy in psana 1.

    PsanaGeometry: Reads lcls 1 style geometry files.
"""


import numpy as np
import psana
from PSCalib.GeometryAccess import GeometryAccess


class PsanaImg:
    """
    It serves as an image accessing layer based on the data management system
    psana in LCLS.
    """

    def __init__(self, exp, run, mode, detector_name):
        # Biolerplate code to access an image
        # Set up data source
        self.datasource_id = f"exp={exp}:run={run}:{mode}"
        self.datasource = psana.DataSource(self.datasource_id)
        self.run_current = next(self.datasource.runs())
        self.timestamps = self.run_current.times()

        # Set up detector
        self.detector = psana.Detector(detector_name)
        # set flag (for Chuck)
        self.detector.do_reshape_2d_to_3d(flag=True) 


    def get(self, event_num, calib=False):
        # Fetch the timestamp according to event number
        timestamp = self.timestamps[int(event_num)]

        # Access each event based on timestamp
        event = self.run_current.event(timestamp)

        # Fetch image data based on timestamp from detector
        if calib:
            img = self.detector.calib(event)
        else:
            img = self.detector.image(event)

        return img

    def timestamp(self, event_num):
        ts = self.timestamps[int(event_num)]
        return ts


class PsanaPhotonEnergy:
    """Uses psana1 ebeam and epicsStore to retrieve
    photon energy."""

    def __init__(self, exp, run, mode):
        # Biolerplate code to access an image
        # Set up data source
        self.datasource_id = f"exp={exp}:run={run}:{mode}"
        self.datasource = psana.DataSource(self.datasource_id)
        self.run_current = next(self.datasource.runs())
        self.timestamps = self.run_current.times()

        # Set up detector and epicsStore
        self.ebeam_det = psana.Detector("EBeam")
        self.es = self.datasource.env().epicsStore()

    def get(self, event_num):
        # Fetch the timestamp according to event number
        timestamp = self.timestamps[int(event_num)]

        # Access each event based on timestamp
        event = self.run_current.event(timestamp)

        # Fetch small wavelenthg (if any) as fallback plan
        try:
            wavelength = self.es.value("SIOC:SYS0:ML00:AO192")
        except:
            wavelength = 0

        # Try to get photon energy from ebeam if not
        # calculate it from wavelenght (if any)
        ebeam = self.ebeam_det.get(event)
        try:
            photonEnergy = ebeam.ebeamPhotonEnergy()
        except:
            photonEnergy = 0
            if wavelength > 0:
                h = 6.626070e-34  # J.m
                c = 2.99792458e8  # m/s
                joulesPerEv = 1.602176621e-19  # J/eV
                photonEnergy = (h / joulesPerEv * c) / (wavelength * 1e-9)

        return photonEnergy


class PsanaGeometry:
    """A getter that reads in lcls1-style geometry file.

    Use this access info from  geometry file (*-end.data).
    Available info is set as class attributes.
    """

    pixel_position = None
    pixel_index_map = None

    def __init__(self, geom):
        """Sets coordinate in real space (convert to m)."""
        cframe = 0  # fixed to psana style (1 is for lab conventions)
        geometry = GeometryAccess(geom, cframe=cframe)

        # Stores a tuple of x,y, and z coordinate arrays
        pixel_coords = geometry.get_pixel_coords(cframe=cframe)
        pixel_coord_indexes = geometry.get_pixel_coord_indexes(cframe=cframe)

        # Converts to metre unit for pixel coordinates
        temp = [np.asarray(t) * 1e-6 for t in pixel_coords]
        temp_index = [np.asarray(t) for t in pixel_coord_indexes]

        # The shape of each axis is represented by five numbers (for this det)
        # e.g. (1,2,2,512,512). We calculate no. of panels by multiplying
        # all numbers except the last two (#pixel_x, #pixel_y).
        panel_num = np.prod(temp[0].shape[:-2])

        shape = (panel_num, temp[0].shape[-2], temp[0].shape[-1])
        pixel_position = np.zeros(shape + (3,))     # x,y,z
        pixel_index_map = np.zeros(shape + (2,))    # x,y

        for n in range(3):
            pixel_position[..., n] = temp[n].reshape(shape)

        for n in range(2):
            pixel_index_map[..., n] = temp_index[n].reshape(shape)

        # Convert to zmq_pull type
        pixel_index_map = pixel_index_map.astype(np.int16)
        pixel_position = pixel_position.astype(np.float32)

        self.pixel_position = pixel_position
        self.pixel_index_map = pixel_index_map
