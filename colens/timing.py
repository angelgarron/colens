from typing import Iterable, Iterator, Mapping

import numpy as np
import scipy
from pycbc.detector import Detector

from colens.transformations import spher_to_cart

TIME_TO_CENTER = 6370e3 / 3e8

DETECTOR_H1 = Detector("H1")
DETECTOR_L1 = Detector("L1")
HANFORD_LOCATION_SPHER = np.array([DETECTOR_H1.longitude, DETECTOR_H1.latitude])
LIVINGSTON_LOCATION_SPHER = np.array([DETECTOR_L1.longitude, DETECTOR_L1.latitude])


def _voxel_down_sample(
    points: np.ndarray, voxel_size: int | float | Iterable[int | float]
) -> np.ndarray:
    if isinstance(voxel_size, (int, float)):
        voxel_size = [voxel_size] * points.shape[
            1
        ]  # Uniform voxel size for all dimensions

    voxel_size = np.array(voxel_size)
    voxel_indices = np.floor(points / voxel_size).astype(int)

    # Create a dictionary to store point indices per voxel
    voxel_dict: Mapping[tuple[int], list[float]] = {}
    for idx, voxel in enumerate(map(tuple, voxel_indices)):
        if voxel not in voxel_dict:
            voxel_dict[voxel] = []
        voxel_dict[voxel].append(idx)

    # Take as representative point the one that is closest to the voxel
    downsampled_indices = np.array(
        [
            voxel_dict[voxel][
                scipy.spatial.KDTree(points[voxel_dict[voxel]]).query(
                    voxel * voxel_size
                )[1]
            ]
            for voxel in voxel_dict
        ]
    )

    return downsampled_indices


def _get_t_prime(t_geocent_original, t_geocent_lensed, phi, theta):
    new_hanford_location_spher = HANFORD_LOCATION_SPHER + np.moveaxis(
        np.array([t_geocent_original, np.zeros_like(t_geocent_original)]), 0, -1
    )  # longitude of detector changes with geocent time
    new_hanford_location_cart = spher_to_cart(new_hanford_location_spher)
    new_livingston_location_spher = LIVINGSTON_LOCATION_SPHER + np.moveaxis(
        np.array([t_geocent_original, np.zeros_like(t_geocent_original)]), 0, -1
    )
    new_livingston_location_cart = spher_to_cart(new_livingston_location_spher)
    new_lensed_hanford_location_spher = HANFORD_LOCATION_SPHER + np.moveaxis(
        np.array([t_geocent_lensed, np.zeros_like(t_geocent_lensed)]), 0, -1
    )
    new_lensed_hanford_location_cart = spher_to_cart(new_lensed_hanford_location_spher)
    new_lensed_livingston_location_spher = LIVINGSTON_LOCATION_SPHER + np.moveaxis(
        np.array([t_geocent_lensed, np.zeros_like(t_geocent_lensed)]), 0, -1
    )
    new_lensed_livingston_location_cart = spher_to_cart(
        new_lensed_livingston_location_spher
    )

    p = spher_to_cart(np.moveaxis(np.array([phi, theta]), 0, -1))

    # times at which compute the coherent snr
    t_1_prime = (
        np.sum(-p * new_hanford_location_cart, axis=-1) * TIME_TO_CENTER
        + t_geocent_original
    )
    t_2_prime = (
        np.sum(-p * new_livingston_location_cart, axis=-1) * TIME_TO_CENTER
        + t_geocent_original
    )
    t_3_prime = (
        np.sum(-p * new_lensed_hanford_location_cart, axis=-1) * TIME_TO_CENTER
        + t_geocent_lensed
    )
    t_4_prime = (
        np.sum(-p * new_lensed_livingston_location_cart, axis=-1) * TIME_TO_CENTER
        + t_geocent_lensed
    )
    return t_1_prime, t_2_prime, t_3_prime, t_4_prime


def get_timing_iterator(
    time_gps_past_seconds: np.ndarray,
    time_gps_future_seconds: np.ndarray,
    ra: np.ndarray,
    dec: np.ndarray,
) -> Iterator[float]:
    t_geocent_original, t_geocent_lensed, phi, theta = np.meshgrid(
        time_gps_past_seconds,
        time_gps_future_seconds,
        ra,
        dec,
        indexing="ij",
    )
    t_1_prime, t_2_prime, t_3_prime, t_4_prime = _get_t_prime(
        t_geocent_original, t_geocent_lensed, phi, theta
    )
    t_prime_downsampled_indices = _voxel_down_sample(
        np.c_[
            t_1_prime.flatten(),
            t_2_prime.flatten(),
            t_3_prime.flatten(),
            t_4_prime.flatten(),
        ],
        0.0005,
    )
    for i, j, k, l in zip(
        t_geocent_original.flatten()[t_prime_downsampled_indices][:20],
        t_geocent_lensed.flatten()[t_prime_downsampled_indices][:20],
        theta.flatten()[t_prime_downsampled_indices][:20],
        phi.flatten()[t_prime_downsampled_indices][:20],
    ):
        yield i, j, k, l
