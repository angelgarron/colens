"""Functions for performing coordinate transformations."""

import numpy as np
from pycbc.detector import gmst_accurate
from scipy.spatial.transform import Rotation as R


class DimensionError(Exception):
    pass


def spher_to_cart(sky_points):
    """Convert spherical coordinates to cartesian coordinates."""
    shape = sky_points.shape
    if shape[-1] != 2:
        raise DimensionError("Last dimension should have size 2")
    cart = np.zeros((*shape[:-1], 3))
    cart[..., 0] = np.cos(sky_points[..., 0]) * np.cos(sky_points[..., 1])
    cart[..., 1] = np.sin(sky_points[..., 0]) * np.cos(sky_points[..., 1])
    cart[..., 2] = np.sin(sky_points[..., 1])
    return cart


def cart_to_spher(sky_points):
    """Convert cartesian coordinates to spherical coordinates, where spher[..., 0]
    is in the range [0, 2pi)."""
    shape = sky_points.shape
    if shape[-1] != 3:
        raise DimensionError("Last dimension should have size 3")
    spher = np.zeros((*sky_points.shape[:-1], 2))
    spher[..., 0] = np.arctan2(sky_points[..., 1], sky_points[..., 0]) % (2 * np.pi)
    spher[..., 1] = np.arcsin(sky_points[..., 2])
    return spher


def ra_to_longitude(
    ra: float | np.ndarray, gps_time: float | np.ndarray
) -> float | np.ndarray:
    """Convert from right ascension to longitude at a given time.

    Args:
        ra (float | np.ndarray): Right ascension in the range [0, 2*np.pi).
        gps_time (float | np.ndarray): GPS time when the longitude will be computed.

    Returns:
        float | np.ndarray: Longitude in radians.
    """
    longitude = ra - gmst_accurate(gps_time)
    return longitude


def longitude_to_ra(
    longitude: float | np.ndarray, gps_time: float | np.ndarray
) -> float | np.ndarray:
    """Convert from longitude to right ascension at a given time.

    Args:
        longitude (float | np.ndarray): Longitude in radians.
        gps_time (float | np.ndarray): GPS time when the right ascension will be computed.

    Returns:
        float | np.ndarray: Right ascension in radians.
    """
    ra = gmst_accurate(gps_time) + longitude
    return ra


def detector_network_to_geographical_coordinates(
    grid: np.ndarray,
    hanford_location_cart: np.ndarray,
    livingston_location_cart: np.ndarray,
    virgo_location_cart: np.ndarray,
    alpha_LV: float,
) -> np.ndarray:
    new_plane_normal = np.cross(
        livingston_location_cart - hanford_location_cart,
        virgo_location_cart - hanford_location_cart,
    )[0]
    new_plane_normal /= np.linalg.norm(new_plane_normal)
    y_axis = np.array([0.0, 1.0, 0.0])
    rotation_axis = np.cross(y_axis, new_plane_normal)
    rotation_axis /= np.linalg.norm(rotation_axis)
    rota = spher_to_cart(grid)
    zero_cart = np.array([1.0, 0.0, 0.0])

    n = np.arccos(np.dot(y_axis, new_plane_normal))
    u = rotation_axis * n
    r = R.from_rotvec(u)
    rota = r.apply(rota)
    zero_cart = r.apply(zero_cart)

    u = new_plane_normal * (np.pi / 2 - alpha_LV)
    r = R.from_rotvec(u)
    xaxis = r.apply(virgo_location_cart[0] - hanford_location_cart[0])
    xaxis = xaxis / np.linalg.norm(xaxis)
    n = np.arccos(np.dot(zero_cart, xaxis))
    u = new_plane_normal * n
    r = R.from_rotvec(u)
    rota = r.apply(rota)
    zero_cart = r.apply(zero_cart)

    spher = cart_to_spher(rota) * 180 / np.pi
    zero = cart_to_spher(zero_cart.reshape(1, -1)) * 180 / np.pi
    return spher
