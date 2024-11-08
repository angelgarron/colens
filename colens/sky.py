"""Functions to implement a sky grid for the search."""

import numpy as np
from scipy.spatial.transform import Rotation as R

from colens.transformations import cart_to_spher, spher_to_cart


def get_circular_sky_patch(
    ra: float, dec: float, sky_error: float, angular_spacing: float
) -> np.ndarray:
    """Compute the coordinates (in units of right ascention and declination) for a circular sky patch
    centered at (`ra`, `dec`) and of a radius given (in radians) by `sky_error`.
    Each point of a concentric ring in the sky grid is separated by `angular_spacing` (in radians).

    Args:
        ra (float): Right ascention of the center of the patch.
        dec (float): Declination of the center of the patch.
        sky_error (float): Radius (in radians) of the patch.
        angular_spacing (float): Distance between each point of a concentric ring (in radians).

    Returns:
        np.ndarray: Sky grid.
    """
    sky_points = np.zeros((1, 2))
    number_of_rings = int(sky_error / angular_spacing)
    # Generate the sky grid centered at the North pole
    for i in range(number_of_rings + 1):
        if i == 0:
            sky_points[0][0] = 0
            sky_points[0][1] = np.pi / 2
        else:
            number_of_points = int(2 * np.pi * i)
            for j in range(number_of_points):
                sky_points = np.row_stack(
                    (sky_points, np.array([j / i, np.pi / 2 - i * angular_spacing]))
                )
    # Convert spherical coordinates to cartesian coordinates
    cart = spher_to_cart(sky_points)
    grb = np.zeros((1, 2))
    grb[0] = ra, dec
    grb_cart = spher_to_cart(grb)
    north_pole = [0, 0, 1]
    ort = np.cross(grb_cart, north_pole)
    norm = np.linalg.norm(ort)
    ort /= norm
    n = -np.arccos(np.dot(grb_cart, north_pole))
    u = ort * n
    # Rotate the sky grid to the center of the external trigger error box
    r = R.from_rotvec(u)
    rota = r.apply(cart)
    # Convert cartesian coordinates back to spherical coordinates
    spher = cart_to_spher(rota)
    return np.array([spher[:, 0], spher[:, 1]])
