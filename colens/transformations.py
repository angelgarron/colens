"""Functions for performing coordinate transformations."""

import numpy as np
from pycbc.detector import gmst_accurate


def spher_to_cart(sky_points):
    """Convert spherical coordinates to cartesian coordinates."""
    cart = np.zeros((len(sky_points), 3))
    cart[:, 0] = np.cos(sky_points[:, 0]) * np.cos(sky_points[:, 1])
    cart[:, 1] = np.sin(sky_points[:, 0]) * np.cos(sky_points[:, 1])
    cart[:, 2] = np.sin(sky_points[:, 1])
    return cart


def cart_to_spher(sky_points):
    """Convert cartesian coordinates to spherical coordinates."""
    spher = np.zeros((len(sky_points), 2))
    spher[:, 0] = np.arctan2(sky_points[:, 1], sky_points[:, 0])
    spher[:, 1] = np.arcsin(sky_points[:, 2])
    return spher


def ra_to_longitude(
    ra: float | np.ndarray, gps_time: float | np.ndarray
) -> float | np.ndarray:
    """Convert from right ascension to longitude at a given time.

    Args:
        ra (float | np.ndarray): Right ascension in the range [0, 2*np.pi).
        gps_time (float | np.ndarray): GPS time when the longitude will be computed.

    Returns:
        float | np.ndarray: Longitude.
    """
    longitude = ra - gmst_accurate(gps_time)
    return longitude


def longitude_to_ra(
    longitude: float | np.ndarray, gps_time: float | np.ndarray
) -> float | np.ndarray:
    """Convert from longitude to right ascension at a given time.

    Args:
        longitude (float | np.ndarray): Longitude.
        gps_time (float | np.ndarray): GPS time when the right ascension will be computed.

    Returns:
        float | np.ndarray: Right ascension.
    """
    ra = gmst_accurate(gps_time) + longitude
    return ra
