"""Code related to the detectors."""

import numpy as np
from pycbc import detector
from pycbc.detector import (
    _ground_detectors,
    coordinates,
    get_available_lal_detectors,
    meter,
)
from pycbc.libutils import import_optional


class MyDetector(detector.Detector):
    def __init__(self, _detector_name, reference_time=1126259462.0):
        """Create class representing a gravitational-wave detector
        Parameters
        ----------
        detector_name: str
            The two-character detector string, i.e. H1, L1, V1, K1, I1
        reference_time: float
            Default is time of GW150914. In this case, the earth's rotation
        will be estimated from a reference time. If 'None', we will
        calculate the time for each gps time requested explicitly
        using a slower but higher precision method.
        """
        if _detector_name.endswith("_lensed"):
            detector_name = _detector_name.split("_")[0]
        else:
            detector_name = _detector_name
        self.name = str(detector_name)

        lal_detectors = [pfx for pfx, name in get_available_lal_detectors()]
        if detector_name in _ground_detectors:
            self.info = _ground_detectors[detector_name]
            self.response = self.info["response"]
            self.location = self.info["location"]
        elif detector_name in lal_detectors:
            lalsim = import_optional("lalsimulation")
            self._lal = lalsim.DetectorPrefixToLALDetector(self.name)
            self.response = self._lal.response
            self.location = self._lal.location
        else:
            raise ValueError("Unkown detector {}".format(detector_name))

        loc = coordinates.EarthLocation(
            self.location[0], self.location[1], self.location[2], unit=meter
        )
        self.latitude = loc.lat.rad
        self.longitude = loc.lon.rad

        self.reference_time = reference_time
        self.sday = None
        self.gmst_reference = None


def calculate_antenna_pattern(detectors, ras, decs, trigger_time: float):
    """Calculate the antenna pattern functions for all detectors and sky positions.

    Args:
        trigger_time (float): Time at which the antenna patterns should be computed.
    """
    antenna_pattern = {}
    for ifo in detectors:
        curr_det = detectors[ifo]
        antenna_pattern[ifo] = []
        for ra, dec in zip(np.atleast_1d(ras), np.atleast_1d(decs)):
            antenna_pattern[ifo].append(
                curr_det.antenna_pattern(
                    ra,
                    dec,
                    polarization=0,
                    t_gps=trigger_time,
                )
            )
    return antenna_pattern
