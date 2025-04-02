from typing import Iterable, Mapping

import numpy as np
import pandas as pd
import scipy

from colens.transformations import cart_to_spher, spher_to_cart


def _voxel_down_sample(
    points: np.ndarray, voxel_size: int | float | Iterable[int | float]
) -> np.ndarray:
    if isinstance(voxel_size, (int, float)):
        voxel_size = [voxel_size] * points.shape[
            1
        ]  # Uniform voxel size for all dimensions

    voxel_indices = np.floor(points / voxel_size).astype(int)

    # Create a dictionary to store point indices per voxel
    voxel_dict: Mapping[tuple[int], list[float]] = {}
    for idx, voxel in enumerate(map(tuple, voxel_indices)):
        if voxel not in voxel_dict:
            voxel_dict[voxel] = []
        voxel_dict[voxel].append(idx)

    # Take as representative point the one that is closest to the center of the voxel
    downsampled_indices = np.array(
        [
            voxel_dict[voxel][
                scipy.spatial.KDTree(points[voxel_dict[voxel]]).query(
                    (np.array([0.5] * points.shape[1]) + voxel) * voxel_size
                )[1]
            ]
            for voxel in voxel_dict
        ]
    )

    return downsampled_indices


def get_t_prime(t_g, t_g_L, phi, theta):
    theta_1 = 0.4
    theta_2 = 0.1
    phi_1 = 0.0
    phi_2 = 0.3
    hanford_location_spher = np.array([[phi_1, theta_1]])
    livingston_location_spher = np.array([[phi_2, theta_2]])

    hanford_location_cart = spher_to_cart(hanford_location_spher)
    livingston_location_cart = spher_to_cart(livingston_location_spher)

    lensed_hanford_location_cart = hanford_location_cart.copy()
    lensed_livingston_location_cart = livingston_location_cart.copy()
    time_to_center = 6370e3 / 3e8

    new_hanford_location_spher = cart_to_spher(hanford_location_cart)
    new_hanford_location_spher = new_hanford_location_spher + np.moveaxis(
        np.array([t_g, np.zeros_like(t_g)]), 0, -1
    )
    new_hanford_location_cart = spher_to_cart(new_hanford_location_spher)
    new_livingston_location_spher = cart_to_spher(livingston_location_cart)
    new_livingston_location_spher = new_livingston_location_spher + np.moveaxis(
        np.array([t_g, np.zeros_like(t_g)]), 0, -1
    )
    new_livingston_location_cart = spher_to_cart(new_livingston_location_spher)
    new_lensed_hanford_location_spher = cart_to_spher(lensed_hanford_location_cart)
    new_lensed_hanford_location_spher = new_lensed_hanford_location_spher + np.moveaxis(
        np.array([t_g_L, np.zeros_like(t_g_L)]), 0, -1
    )
    new_lensed_hanford_location_cart = spher_to_cart(new_lensed_hanford_location_spher)
    new_lensed_livingston_location_spher = cart_to_spher(
        lensed_livingston_location_cart
    )
    new_lensed_livingston_location_spher = (
        new_lensed_livingston_location_spher
        + np.moveaxis(np.array([t_g_L, np.zeros_like(t_g_L)]), 0, -1)
    )
    new_lensed_livingston_location_cart = spher_to_cart(
        new_lensed_livingston_location_spher
    )

    # the direction of the gw
    p = np.moveaxis(
        np.array(
            [
                np.cos(phi) * np.cos(theta),
                np.sin(phi) * np.cos(theta),
                np.ones_like(phi) * np.sin(theta),
            ]
        ),
        0,
        -1,
    )

    # times at which compute the coherent snr
    t_1_prime = np.sum(-p * new_hanford_location_cart, axis=-1) * time_to_center + t_g
    t_2_prime = (
        np.sum(-p * new_livingston_location_cart, axis=-1) * time_to_center + t_g
    )
    t_3_prime = (
        np.sum(-p * new_lensed_hanford_location_cart, axis=-1) * time_to_center + t_g_L
    )
    t_4_prime = (
        np.sum(-p * new_lensed_livingston_location_cart, axis=-1) * time_to_center
        + t_g_L
    )
    return t_1_prime, t_2_prime, t_3_prime, t_4_prime


def get_timing_iterator(time_gps_past_seconds, time_gps_future_seconds, delta_t):
    df = pd.read_csv("./sky_position_and_timing_samples.csv")
    t_g, t_g_L, phi, theta = np.meshgrid(
        np.arange(
            time_gps_past_seconds - 0.001, time_gps_past_seconds + 0.001, delta_t
        ),
        np.arange(
            time_gps_future_seconds - 0.001, time_gps_future_seconds + 0.001, delta_t
        ),
        df["ra"][:2].to_numpy(),
        df["dec"][:2].to_numpy(),
        indexing="ij",
    )
    t_1_prime, t_2_prime, t_3_prime, t_4_prime = get_t_prime(t_g, t_g_L, phi, theta)
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
        t_g.flatten()[t_prime_downsampled_indices][:20],
        t_g_L.flatten()[t_prime_downsampled_indices][:20],
        theta.flatten()[t_prime_downsampled_indices][:20],
        phi.flatten()[t_prime_downsampled_indices][:20],
    ):
        yield i, j, k, l
