import numpy as np

from colens.timing import _get_meshgrid, _voxel_down_sample


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


def test_get_grid_mesh():
    time_past = np.array([10, 11, 12, 13])
    time_future = np.array([20, 21, 22, 23, 24])
    ra = np.array([1, 3, 5, 7])
    dec = np.array([2, 4, 6, 8])
    grid_time_future, grid_time_past, grid_ra, grid_dec = _get_meshgrid(
        time_future, np.array([time_past, ra, dec]).T
    )
    grid_time_past_expected = np.array(
        [
            [10, 11, 12, 13],
            [10, 11, 12, 13],
            [10, 11, 12, 13],
            [10, 11, 12, 13],
            [10, 11, 12, 13],
        ]
    )
    grid_time_future_expected = np.array(
        [
            [20, 20, 20, 20],
            [21, 21, 21, 21],
            [22, 22, 22, 22],
            [23, 23, 23, 23],
            [24, 24, 24, 24],
        ]
    )
    grid_ra_expected = np.array(
        [
            [1, 3, 5, 7],
            [1, 3, 5, 7],
            [1, 3, 5, 7],
            [1, 3, 5, 7],
            [1, 3, 5, 7],
        ]
    )
    grid_dec_expected = np.array(
        [
            [2, 4, 6, 8],
            [2, 4, 6, 8],
            [2, 4, 6, 8],
            [2, 4, 6, 8],
            [2, 4, 6, 8],
        ]
    )
    np.testing.assert_allclose(grid_time_past, grid_time_past_expected)
    np.testing.assert_allclose(grid_time_future, grid_time_future_expected)
    np.testing.assert_allclose(grid_ra, grid_ra_expected)
    np.testing.assert_allclose(grid_dec, grid_dec_expected)
