import numpy as np

from colens.background import get_time_delay_indices


def test_get_time_delay_indices():
    lensed_instruments = ["H1_lensed", "L1_lensed"]
    unlensed_instruments = ["H1", "L1"]
    sample_rate = 4096.0
    time_delay_zerolag_seconds = {
        0: {"H1": 0.25, "L1": 0.26, "H1_lensed": 0.25, "L1_lensed": 0.26}
    }
    time_slides_seconds, time_delay_idx = get_time_delay_indices(
        num_slides=5,
        slide_shift_seconds=1,
        lensed_instruments=lensed_instruments,
        unlensed_instruments=unlensed_instruments,
        sample_rate=sample_rate,
        time_delay_zerolag_seconds=time_delay_zerolag_seconds,
    )
    expected_time_slides = {
        "H1_lensed": np.array([0, 0, 0, 0, 0]),
        "L1_lensed": np.array([0, 1, 2, 3, 4]),
        "H1": np.array([0, 0, 0, 0, 0]),
        "L1": np.array([0, 0, 0, 0, 0]),
    }
    expected_time_delay_idx = {
        0: {
            0: {
                "H1": int(round((time_delay_zerolag_seconds[0]["H1"] * sample_rate))),
                "L1": int(round((time_delay_zerolag_seconds[0]["L1"] * sample_rate))),
                "H1_lensed": int(
                    round((time_delay_zerolag_seconds[0]["H1"] * sample_rate))
                ),
                "L1_lensed": int(
                    round((time_delay_zerolag_seconds[0]["L1"] * sample_rate))
                ),
            }
        },
        1: {
            0: {
                "H1": int(round((time_delay_zerolag_seconds[0]["H1"] * sample_rate))),
                "L1": int(round((time_delay_zerolag_seconds[0]["L1"] * sample_rate))),
                "H1_lensed": int(
                    round((time_delay_zerolag_seconds[0]["H1"] * sample_rate))
                ),
                "L1_lensed": int(
                    round((time_delay_zerolag_seconds[0]["L1"] + 1) * sample_rate)
                ),
            }
        },
        2: {
            0: {
                "H1": int(round((time_delay_zerolag_seconds[0]["H1"] * sample_rate))),
                "L1": int(round((time_delay_zerolag_seconds[0]["L1"] * sample_rate))),
                "H1_lensed": int(
                    round((time_delay_zerolag_seconds[0]["H1"] * sample_rate))
                ),
                "L1_lensed": int(
                    round((time_delay_zerolag_seconds[0]["L1"] + 2) * sample_rate)
                ),
            }
        },
        3: {
            0: {
                "H1": int(round((time_delay_zerolag_seconds[0]["H1"] * sample_rate))),
                "L1": int(round((time_delay_zerolag_seconds[0]["L1"] * sample_rate))),
                "H1_lensed": int(
                    round((time_delay_zerolag_seconds[0]["H1"] * sample_rate))
                ),
                "L1_lensed": int(
                    round((time_delay_zerolag_seconds[0]["L1"] + 3) * sample_rate)
                ),
            }
        },
        4: {
            0: {
                "H1": int(round((time_delay_zerolag_seconds[0]["H1"] * sample_rate))),
                "L1": int(round((time_delay_zerolag_seconds[0]["L1"] * sample_rate))),
                "H1_lensed": int(
                    round((time_delay_zerolag_seconds[0]["H1"] * sample_rate))
                ),
                "L1_lensed": int(
                    round((time_delay_zerolag_seconds[0]["L1"] + 4) * sample_rate)
                ),
            }
        },
    }
    np.testing.assert_equal(time_slides_seconds, expected_time_slides)
    np.testing.assert_equal(time_delay_idx, expected_time_delay_idx)
