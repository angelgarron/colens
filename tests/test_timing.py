import numpy as np

from colens.timing import _voxel_down_sample


def test_recover_initial_points_when_match_desired():
    initial_points = np.array(
        [
            [-1, 1],
            [0, 1],
            [1, 1],
            [-1, 0],
            [0, 0],
            [1, 0],
            [-1, -1],
            [0, -1],
            [1, -1],
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
            [-1, -1],
            [0, -1],
            [1, -1],
        ]
    )
    mask = _voxel_down_sample(initial_points, 1)
    np.testing.assert_equal(initial_points[mask], expected)
