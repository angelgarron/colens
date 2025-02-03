"""Functions to implement a sky grid for the search."""

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
from scipy.spatial.transform import Rotation as R

from colens.transformations import cart_to_spher, spher_to_cart


@dataclass
class SkyCoordinates:
    ra: float
    dec: float


@dataclass
class SkyGrid:
    ra: Iterable
    dec: Iterable

    def __post_init__(self):
        if len(self.ra) != len(self.dec):
            raise ValueError("ra and dec must have the same length")

    def __getitem__(self, index):
        return SkyCoordinates(self.ra[index], self.dec[index])

    def __len__(self):
        return len(self.ra)


def get_circular_sky_patch(
    ra: float, dec: float, sky_error: float, angular_spacing: float
) -> SkyGrid:
    """Compute the coordinates (in units of right ascention and declination) for a circular sky patch
    centered at (`ra`, `dec`) and of a radius given (in radians) by `sky_error`.
    Each point of a concentric ring in the sky grid is separated by `angular_spacing` (in radians).

    Args:
        ra (float): Right ascention of the center of the patch.
        dec (float): Declination of the center of the patch.
        sky_error (float): Radius (in radians) of the patch.
        angular_spacing (float): Distance between each point of a concentric ring (in radians).

    Returns:
        SkyGrid: Sky grid.
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
    return SkyGrid(spher[:, 0], spher[:, 1])


def get_physically_admissible_time_delays(
    T_HL: float, T_HV: float, alpha_LV: float
) -> np.ndarray:
    TAU_HL, TAU_HV = np.meshgrid(
        np.arange(-T_HL, T_HL, 2 / 4096), np.arange(-T_HV, T_HV, 2 / 4096)
    )
    A_3 = np.array(
        [
            [T_HV**2 / T_HL**2, -T_HV / T_HL * np.cos(alpha_LV)],
            [-T_HV / T_HL * np.cos(alpha_LV), 1],
        ]
    )
    B_3 = T_HV**2 * np.sin(alpha_LV) ** 2
    mask = (
        (
            np.array([TAU_HL, TAU_HV]).transpose(1, 2, 0)
            * (A_3 @ (np.array([TAU_HL, TAU_HV]).transpose(1, 0, 2))).transpose(0, 2, 1)
        ).sum(axis=2)
    ) < B_3
    tau = np.array([TAU_HL[mask], TAU_HV[mask]]).T
    return tau


def project_time_delays_onto_celestial_sphere(
    tau: np.ndarray, T_HL, T_HV, alpha_LV
) -> tuple[np.ndarray, np.ndarray]:
    theta = np.arccos(-tau[:, 0] / T_HL)
    phi = np.arccos(
        -(T_HL * tau[:, 1] - T_HV * tau[:, 0] * np.cos(alpha_LV))
        / (T_HV * np.sqrt(T_HL**2 - tau[:, 0] ** 2) * np.sin(alpha_LV))
    )
    theta = np.pi / 2 - theta
    return phi, theta
