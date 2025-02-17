import numpy as np
import pytest

from colens.transformations import DimensionError, spher_to_cart


def test_spher_to_cart_shape_two():
    phi = 0
    theta = 1
    coordinates_spher = np.array([phi, theta])
    expected_x = np.cos(phi) * np.cos(theta)
    expected_y = np.sin(phi) * np.cos(theta)
    expected_z = np.sin(theta)
    expected_coordinates_cart = np.array([expected_x, expected_y, expected_z])
    np.testing.assert_equal(spher_to_cart(coordinates_spher), expected_coordinates_cart)


def test_spher_to_cart_shape_one_two():
    phi = 0
    theta = 1
    coordinates_spher = np.array([[phi, theta]])
    expected_x = np.cos(phi) * np.cos(theta)
    expected_y = np.sin(phi) * np.cos(theta)
    expected_z = np.sin(theta)
    expected_coordinates_cart = np.array([[expected_x, expected_y, expected_z]])
    np.testing.assert_equal(spher_to_cart(coordinates_spher), expected_coordinates_cart)


def test_spher_to_cart_shape_one_three():
    phi = 0
    theta = 1
    coordinates_spher = np.array([[phi, theta, 1.5]])
    expected_x = np.cos(phi) * np.cos(theta)
    expected_y = np.sin(phi) * np.cos(theta)
    expected_z = np.sin(theta)
    expected_coordinates_cart = np.array([[expected_x, expected_y, expected_z]])
    with pytest.raises(DimensionError):
        np.testing.assert_equal(
            spher_to_cart(coordinates_spher), expected_coordinates_cart
        )


def test_spher_to_cart_shape_two_two():
    phi = 0
    theta = 1
    coordinates_spher = np.array(
        [
            [phi, theta],
            [phi + 0.5, theta + 0.3],
        ]
    )
    expected_1_x = np.cos(phi) * np.cos(theta)
    expected_1_y = np.sin(phi) * np.cos(theta)
    expected_1_z = np.sin(theta)
    expected_2_x = np.cos(phi + 0.5) * np.cos(theta + 0.3)
    expected_2_y = np.sin(phi + 0.5) * np.cos(theta + 0.3)
    expected_2_z = np.sin(theta + 0.3)
    expected_coordinates_cart = np.array(
        [
            [expected_1_x, expected_1_y, expected_1_z],
            [expected_2_x, expected_2_y, expected_2_z],
        ]
    )
    np.testing.assert_equal(spher_to_cart(coordinates_spher), expected_coordinates_cart)


def test_spher_to_cart_shape_two_two_two():
    phi = 0
    theta = 1
    coordinates_spher = np.array(
        [
            [
                [phi, theta],
                [phi + 0.1, theta + 0.2],
            ],
            [
                [phi + 0.3, theta + 0.4],
                [phi + 0.5, theta + 0.6],
            ],
        ]
    )
    expected_1_1_x = np.cos(phi) * np.cos(theta)
    expected_1_1_y = np.sin(phi) * np.cos(theta)
    expected_1_1_z = np.sin(theta)
    expected_1_2_x = np.cos(phi + 0.1) * np.cos(theta + 0.2)
    expected_1_2_y = np.sin(phi + 0.1) * np.cos(theta + 0.2)
    expected_1_2_z = np.sin(theta + 0.2)
    expected_2_1_x = np.cos(phi + 0.3) * np.cos(theta + 0.4)
    expected_2_1_y = np.sin(phi + 0.3) * np.cos(theta + 0.4)
    expected_2_1_z = np.sin(theta + 0.4)
    expected_2_2_x = np.cos(phi + 0.5) * np.cos(theta + 0.6)
    expected_2_2_y = np.sin(phi + 0.5) * np.cos(theta + 0.6)
    expected_2_2_z = np.sin(theta + 0.6)
    expected_coordinates_cart = np.array(
        [
            [
                [expected_1_1_x, expected_1_1_y, expected_1_1_z],
                [expected_1_2_x, expected_1_2_y, expected_1_2_z],
            ],
            [
                [expected_2_1_x, expected_2_1_y, expected_2_1_z],
                [expected_2_2_x, expected_2_2_y, expected_2_2_z],
            ],
        ]
    )
    np.testing.assert_equal(spher_to_cart(coordinates_spher), expected_coordinates_cart)
