import numpy as np
from pycbc.detector import gmst_accurate

from colens.timing import _voxel_down_sample
from colens.transformations import geographical_to_celestial


def test_recover_initial_points():
    initial_points = np.array(
        [
            [-1, 1],
            [0, 1],
            [1, 1],
            [-1, 0],
            [0, 0],
            [1, 0],
        ]
    )
    mask = _voxel_down_sample(initial_points, 1)
    # all points fall in the voxels
    np.testing.assert_equal(mask, np.arange(len(initial_points)))


def test_recover_initial_points_except_one_in_the_middle():
    initial_points = np.array(
        [
            [-1, 1],
            [-0.7, 1],  # [-1, 1] is closer to the voxel
            [0, 1],
            [1, 1],
            [-1, 0],
            [0, 0],
            [1, 0],
        ]
    )
    expected = np.array(
        [
            [-1, 1],
            [0, 1],
            [1, 1],
            [-1, 0],
            [0, 0],
            [1, 0],
        ]
    )
    mask = _voxel_down_sample(initial_points, 1)
    np.testing.assert_equal(initial_points[mask], expected)


def test_recover_even_if_not_fall_in_voxel():
    # first point doesn't fall in voxel, but will be recovered
    # because the is no closer point
    initial_points = np.array(
        [
            [-0.9, 1],
            [-0.7, 1],  # [-0.9, 1] is closer to the voxel
            [0, 1],
            [1, 1],
            [-1, 0],
            [0, 0],
            [1, 0],
        ]
    )
    expected = np.array(
        [
            [-0.9, 1],
            [0, 1],
            [1, 1],
            [-1, 0],
            [0, 0],
            [1, 0],
        ]
    )
    mask = _voxel_down_sample(initial_points, 1)
    np.testing.assert_equal(initial_points[mask], expected)


def test_not_uniform_voxel_size():
    initial_points = np.array(
        [
            [-2, 1],
            [-1, 1],
            [0, 1],
            [1, 1],
            [2, 1],
            [-2, 0],
            [-1, 0],
            [0, 0],
            [1, 0],
            [2, 0],
        ]
    )
    expected = np.array(
        [
            [-2, 1],
            [0, 1],
            [2, 1],
            [-2, 0],
            [0, 0],
            [2, 0],
        ]
    )
    mask = _voxel_down_sample(initial_points, (2, 1))
    np.testing.assert_equal(initial_points[mask], expected)


def test_one_dimensional_initial_points():
    initial_points = np.array([-1, 0, 1, 2]).reshape(-1, 1)
    expected = np.array([-1, 0, 2]).reshape(-1, 1)
    mask = _voxel_down_sample(initial_points, 2)
    np.testing.assert_equal(initial_points[mask], expected)


rng = np.random.default_rng(1234)


def test_geographical_to_celestial_grid():
    t_gps = rng.uniform(low=0, high=10000, size=(5, 7, 13))
    geographical = np.array([1.1, 2.2])

    expected = np.zeros((5, 7, 13, 2))
    expected[..., 0] = np.ones((5, 7, 13)) * geographical[0] + gmst_accurate(t_gps)
    expected[..., 1] = np.ones((5, 7, 13)) * geographical[1]
    result = geographical_to_celestial(geographical, t_gps)
    np.testing.assert_allclose(result, expected)
