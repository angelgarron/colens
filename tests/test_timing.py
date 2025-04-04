import numpy as np

from colens.timing import _voxel_down_sample


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
