from typing import Iterable, Iterator, Mapping

import astropy
import numpy as np
import scipy
from pycbc.detector import Detector

from colens.transformations import geographical_to_celestial, spher_to_cart

DETECTOR_H1 = Detector("H1")
DETECTOR_L1 = Detector("L1")
HANFORD_LOCATION_SPHER = np.array([DETECTOR_H1.longitude, DETECTOR_H1.latitude])
LIVINGSTON_LOCATION_SPHER = np.array([DETECTOR_L1.longitude, DETECTOR_L1.latitude])
TIME_TO_CENTER_H1 = np.linalg.norm(DETECTOR_H1.location) / astropy.constants.c.value
TIME_TO_CENTER_L1 = np.linalg.norm(DETECTOR_L1.location) / astropy.constants.c.value


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
    voxel_dict: Mapping[tuple[int], list[int]] = {}
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


def _get_t_prime(
    grid_time_gps_past_seconds: np.ndarray,
    grid_time_gps_future_seconds: np.ndarray,
    grid_ra: np.ndarray,
    grid_dec: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    new_hanford_location_spher = geographical_to_celestial(
        HANFORD_LOCATION_SPHER, grid_time_gps_past_seconds
    )  # longitude of detector changes with geocent time
    new_hanford_location_cart = spher_to_cart(new_hanford_location_spher)
    new_livingston_location_spher = geographical_to_celestial(
        LIVINGSTON_LOCATION_SPHER, grid_time_gps_past_seconds
    )
    new_livingston_location_cart = spher_to_cart(new_livingston_location_spher)
    new_lensed_hanford_location_spher = geographical_to_celestial(
        HANFORD_LOCATION_SPHER, grid_time_gps_future_seconds
    )
    new_lensed_hanford_location_cart = spher_to_cart(new_lensed_hanford_location_spher)
    new_lensed_livingston_location_spher = geographical_to_celestial(
        LIVINGSTON_LOCATION_SPHER, grid_time_gps_future_seconds
    )
    new_lensed_livingston_location_cart = spher_to_cart(
        new_lensed_livingston_location_spher
    )

    p = spher_to_cart(np.moveaxis(np.array([grid_ra, grid_dec]), 0, -1))

    # Times at which compute the coherent snr
    # We need to rescale them by TIME_TO_CENTER because p and location vectors are normalized
    t_1_prime = (
        np.sum(-p * new_hanford_location_cart, axis=-1) * TIME_TO_CENTER_H1
        + grid_time_gps_past_seconds
    )
    t_2_prime = (
        np.sum(-p * new_livingston_location_cart, axis=-1) * TIME_TO_CENTER_L1
        + grid_time_gps_past_seconds
    )
    t_3_prime = (
        np.sum(-p * new_lensed_hanford_location_cart, axis=-1) * TIME_TO_CENTER_H1
        + grid_time_gps_future_seconds
    )
    t_4_prime = (
        np.sum(-p * new_lensed_livingston_location_cart, axis=-1) * TIME_TO_CENTER_L1
        + grid_time_gps_future_seconds
    )
    return t_1_prime, t_2_prime, t_3_prime, t_4_prime


def _get_meshgrid(*xi: np.ndarray) -> list[np.ndarray]:
    """Adapted from `np.meshgrid` to work when one of the
    arrays passed has dimension greater than 1. In that case,
    each column of that array will contribute as a coordiante vector,
    but without addind extra dimensions to the coordinate matrices returned.
    """
    s0 = (1,) * len(xi)
    output = []
    for i, x in enumerate(xi):
        if x.ndim > 1:
            for y in x.T:
                output.append(np.asanyarray(y).reshape(s0[:i] + (-1,) + s0[i + 1 :]))
        else:
            output.append(np.asanyarray(x).reshape(s0[:i] + (-1,) + s0[i + 1 :]))
    return np.broadcast_arrays(*output, subok=True)


def get_timing_iterator(
    time_gps_past_seconds: np.ndarray,
    time_gps_future_seconds: np.ndarray,
    ra: np.ndarray,
    dec: np.ndarray,
) -> Iterator[float]:
    grid_time_gps_future_seconds, grid_time_gps_past_seconds, grid_ra, grid_dec = (
        _get_meshgrid(
            time_gps_future_seconds,
            np.array([time_gps_past_seconds, ra, dec]).T,
        )
    )
    t_1_prime, t_2_prime, t_3_prime, t_4_prime = _get_t_prime(
        grid_time_gps_past_seconds, grid_time_gps_future_seconds, grid_ra, grid_dec
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
        grid_time_gps_past_seconds.flatten()[t_prime_downsampled_indices],
        grid_time_gps_future_seconds.flatten()[t_prime_downsampled_indices],
        grid_dec.flatten()[t_prime_downsampled_indices],
        grid_ra.flatten()[t_prime_downsampled_indices],
    ):
        yield i, j, k, l
