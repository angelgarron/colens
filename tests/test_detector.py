import numpy as np

from colens.detector import MyDetector, calculate_antenna_pattern
from colens.sky import SkyGrid


def test_calculate_antenna_pattern():
    sky_grid = SkyGrid([1, 2], [0.5, 1.5])
    trigger_time = 1126259462.4
    antenna_pattern = calculate_antenna_pattern(
        sky_grid, {"H1": trigger_time, "L1": trigger_time}
    )
    expected_antenna_pattern = {
        "H1": [
            MyDetector("H1").antenna_pattern(
                sky_grid[0].ra, sky_grid[0].dec, polarization=0, t_gps=trigger_time
            ),
            MyDetector("H1").antenna_pattern(
                sky_grid[1].ra, sky_grid[1].dec, polarization=0, t_gps=trigger_time
            ),
        ],
        "L1": [
            MyDetector("L1").antenna_pattern(
                sky_grid[0].ra, sky_grid[0].dec, polarization=0, t_gps=trigger_time
            ),
            MyDetector("L1").antenna_pattern(
                sky_grid[1].ra, sky_grid[1].dec, polarization=0, t_gps=trigger_time
            ),
        ],
    }
    np.testing.assert_equal(antenna_pattern, expected_antenna_pattern)
