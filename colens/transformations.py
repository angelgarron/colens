"""Functions for performing coordinate transformations."""

import numpy as np


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
