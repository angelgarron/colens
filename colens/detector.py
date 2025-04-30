"""Code related to the detectors."""

import numpy as np


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
