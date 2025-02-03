"""Functions to implement a sky grid for the search."""

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd
from pycbc.detector import Detector
from scipy.spatial.transform import Rotation as R

from colens.transformations import (
    cart_to_spher,
    detector_network_to_geographical_coordinates,
    ra_to_longitude,
    spher_to_cart,
)


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


def get_delays_for_sky_positions(
    grid,
    hanford_location_cart,
    livingston_location_cart,
    virgo_location_cart,
    time_to_center,
) -> np.ndarray:
    rota = spher_to_cart(grid * np.pi / 180)

    new_delays_x = -(
        (
            np.dot(
                livingston_location_cart[0].reshape(1, -1)
                - hanford_location_cart[0].reshape(1, -1),
                rota.T,
            )
        )
        * time_to_center
    )[0]
    new_delays_y = -(
        (
            np.dot(
                virgo_location_cart[0].reshape(1, -1)
                - hanford_location_cart[0].reshape(1, -1),
                rota.T,
            )
        )
        * time_to_center
    )[0]
    new_delays = np.array([new_delays_x, new_delays_y]).T

    return new_delays


def get_sky_grid_for_three_detectors() -> SkyGrid:
    detector_h1 = Detector("H1")
    detector_l1 = Detector("L1")
    detector_v1 = Detector("V1")

    hanford_location = (
        np.array([detector_h1.longitude, detector_h1.latitude]) * 180 / np.pi
    )
    livingston_location = (
        np.array([detector_l1.longitude, detector_l1.latitude]) * 180 / np.pi
    )
    virgo_location = (
        np.array([detector_v1.longitude, detector_v1.latitude]) * 180 / np.pi
    )

    TIME_GPS_PAST_SECONDS = 1185389807.298705

    hanford_location_cart = spher_to_cart(
        (
            np.array(
                [
                    ra_to_longitude(hanford_location[0], TIME_GPS_PAST_SECONDS),
                    hanford_location[1],
                ]
            )
            * np.pi
            / 180
        ).reshape(1, -1)
    )
    livingston_location_cart = spher_to_cart(
        (
            np.array(
                [
                    ra_to_longitude(livingston_location[0], TIME_GPS_PAST_SECONDS),
                    livingston_location[1],
                ]
            )
            * np.pi
            / 180
        ).reshape(1, -1)
    )
    virgo_location_cart = spher_to_cart(
        (
            np.array(
                [
                    ra_to_longitude(virgo_location[0], TIME_GPS_PAST_SECONDS),
                    virgo_location[1],
                ]
            )
            * np.pi
            / 180
        ).reshape(1, -1)
    )

    T_HL = detector_h1.light_travel_time_to_detector(detector_l1)
    T_HV = detector_h1.light_travel_time_to_detector(detector_v1)

    time_to_center = (
        T_HL
        / 2
        / np.sin(
            np.arccos(np.dot(livingston_location_cart[0], hanford_location_cart[0])) / 2
        )
    )

    # correction of time delays to facilitate checking correct result
    T_HL = (
        np.linalg.norm(livingston_location_cart[0] - hanford_location_cart[0])
        * time_to_center
    )
    T_HV = (
        np.linalg.norm(virgo_location_cart[0] - hanford_location_cart[0])
        * time_to_center
    )

    alpha_LV = np.arccos(
        np.dot(
            (virgo_location_cart[0] - hanford_location_cart[0])
            / np.linalg.norm(virgo_location_cart[0] - hanford_location_cart[0]),
            (livingston_location_cart[0] - hanford_location_cart[0])
            / np.linalg.norm(livingston_location_cart[0] - hanford_location_cart[0]),
        )
    )

    n_samples = 10000
    samples = pd.read_csv("./sky_position_samples.csv").to_numpy()[
        :n_samples
    ]  # this is in radians
    samples_cart = spher_to_cart(samples)

    samples_delays = get_delays_for_sky_positions(
        cart_to_spher(samples_cart) * 180 / np.pi,
        hanford_location_cart,
        livingston_location_cart,
        virgo_location_cart,
        time_to_center,
    )

    tau = get_physically_admissible_time_delays(T_HL, T_HV, alpha_LV)

    tau_mask = (
        (np.linalg.norm(tau[np.newaxis, :] - samples_delays[:, np.newaxis], axis=-1))
        < 0.0001
    ).any(axis=0)

    phi, theta = project_time_delays_onto_celestial_sphere(
        tau[tau_mask], T_HL, T_HV, alpha_LV
    )

    grid = np.array([-phi, theta]).T

    spher = detector_network_to_geographical_coordinates(
        grid,
        hanford_location_cart,
        livingston_location_cart,
        virgo_location_cart,
        alpha_LV,
    )

    spher = spher * np.pi / 180

    return SkyGrid(spher[:, 0], spher[:, 1])
