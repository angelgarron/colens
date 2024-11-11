import numpy as np

from colens.background import get_time_delay_indices
from colens.detector import MyDetector
from colens.sky import SkyGrid


def test_get_time_delay_indices():
    trigger_times_seconds = {"H1_lensed": 15, "L1_lensed": 15, "H1": 13, "L1": 13}
    lensed_instruments = ["H1_lensed", "L1_lensed"]
    unlensed_instruments = ["H1", "L1"]
    sample_rate = 4096.0
    sky_grid = SkyGrid(ra=[1], dec=[1])
    time_slides, time_delay_idx = get_time_delay_indices(
        num_slides=5,
        slide_shift_seconds=1,
        lensed_instruments=lensed_instruments,
        unlensed_instruments=unlensed_instruments,
        sky_grid=sky_grid,
        trigger_times_seconds=trigger_times_seconds,
        sample_rate=sample_rate,
    )
    expected_time_slides = {
        "H1_lensed": np.array([0, 0, 0, 0, 0]),
        "L1_lensed": np.array([0, 1, 2, 3, 4]),
        "H1": np.array([0, 0, 0, 0, 0]),
        "L1": np.array([0, 0, 0, 0, 0]),
    }
    time_delay_idx_zerolag = {
        position_index: {
            ifo: MyDetector(ifo).time_delay_from_earth_center(
                sky_position.ra,
                sky_position.dec,
                trigger_times_seconds[ifo],
            )
            for ifo in unlensed_instruments + lensed_instruments
        }
        for position_index, sky_position in enumerate(sky_grid)
    }
    expected_time_delay_idx = {
        0: {
            0: {
                "H1": int(round((time_delay_idx_zerolag[0]["H1"] * sample_rate))),
                "L1": int(round((time_delay_idx_zerolag[0]["L1"] * sample_rate))),
                "H1_lensed": int(
                    round((time_delay_idx_zerolag[0]["H1"] * sample_rate))
                ),
                "L1_lensed": int(
                    round((time_delay_idx_zerolag[0]["L1"] * sample_rate))
                ),
            }
        },
        1: {
            0: {
                "H1": int(round((time_delay_idx_zerolag[0]["H1"] * sample_rate))),
                "L1": int(round((time_delay_idx_zerolag[0]["L1"] * sample_rate))),
                "H1_lensed": int(
                    round((time_delay_idx_zerolag[0]["H1"] * sample_rate))
                ),
                "L1_lensed": int(
                    round((time_delay_idx_zerolag[0]["L1"] + 1) * sample_rate)
                ),
            }
        },
        2: {
            0: {
                "H1": int(round((time_delay_idx_zerolag[0]["H1"] * sample_rate))),
                "L1": int(round((time_delay_idx_zerolag[0]["L1"] * sample_rate))),
                "H1_lensed": int(
                    round((time_delay_idx_zerolag[0]["H1"] * sample_rate))
                ),
                "L1_lensed": int(
                    round((time_delay_idx_zerolag[0]["L1"] + 2) * sample_rate)
                ),
            }
        },
        3: {
            0: {
                "H1": int(round((time_delay_idx_zerolag[0]["H1"] * sample_rate))),
                "L1": int(round((time_delay_idx_zerolag[0]["L1"] * sample_rate))),
                "H1_lensed": int(
                    round((time_delay_idx_zerolag[0]["H1"] * sample_rate))
                ),
                "L1_lensed": int(
                    round((time_delay_idx_zerolag[0]["L1"] + 3) * sample_rate)
                ),
            }
        },
        4: {
            0: {
                "H1": int(round((time_delay_idx_zerolag[0]["H1"] * sample_rate))),
                "L1": int(round((time_delay_idx_zerolag[0]["L1"] * sample_rate))),
                "H1_lensed": int(
                    round((time_delay_idx_zerolag[0]["H1"] * sample_rate))
                ),
                "L1_lensed": int(
                    round((time_delay_idx_zerolag[0]["L1"] + 4) * sample_rate)
                ),
            }
        },
    }
    np.testing.assert_equal(time_slides, expected_time_slides)
    np.testing.assert_equal(time_delay_idx, expected_time_delay_idx)
