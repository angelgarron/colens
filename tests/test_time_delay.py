import numpy as np

from colens.background import get_time_delay_indices, get_time_slides_seconds


def test_get_time_slides_seconds():
    num_slides = 5
    slide_shift_seconds = 1
    unlensed_instruments = ["H1", "L1"]
    lensed_instruments = ["H1_lensed", "L1_lensed"]
    time_slides_seconds = get_time_slides_seconds(
        num_slides, slide_shift_seconds, unlensed_instruments, lensed_instruments
    )

    expected_time_slides_seconds = {
        "H1_lensed": np.array([0, 0, 0, 0, 0]),
        "L1_lensed": np.array([0, 1, 2, 3, 4]),
        "H1": np.array([0, 0, 0, 0, 0]),
        "L1": np.array([0, 0, 0, 0, 0]),
    }
    np.testing.assert_equal(time_slides_seconds, expected_time_slides_seconds)


def test_get_time_delay_indices():
    sample_rate = 4096.0
    time_delay_zerolag_seconds = {
        0: {"A": 0.25, "B": 0.26, "C": 0.27},
        1: {"A": 0.28, "B": 0.29, "C": 0.3},
    }
    time_slides_seconds = {
        "A": np.array([1, 2, 3, 9]),
        "B": np.array([0, 4, 0, 2]),
        "C": np.array([7, 5, 0, 43]),
    }
    time_delay_idx = get_time_delay_indices(
        sample_rate=sample_rate,
        time_delay_zerolag_seconds=time_delay_zerolag_seconds,
        time_slides_seconds=time_slides_seconds,
    )
    expected_time_delay_idx = [
        [
            {
                "A": round(((0.25 + 1) * sample_rate)),
                "B": round(((0.26) * sample_rate)),
                "C": round(((0.27 + 7) * sample_rate)),
            },
            {
                "A": round(((0.28 + 1) * sample_rate)),
                "B": round(((0.29) * sample_rate)),
                "C": round(((0.3 + 7) * sample_rate)),
            },
        ],
        [
            {
                "A": round(((0.25 + 2) * sample_rate)),
                "B": round(((0.26 + 4) * sample_rate)),
                "C": round(((0.27 + 5) * sample_rate)),
            },
            {
                "A": round(((0.28 + 2) * sample_rate)),
                "B": round(((0.29 + 4) * sample_rate)),
                "C": round(((0.3 + 5) * sample_rate)),
            },
        ],
        [
            {
                "A": round(((0.25 + 3) * sample_rate)),
                "B": round(((0.26) * sample_rate)),
                "C": round(((0.27) * sample_rate)),
            },
            {
                "A": round(((0.28 + 3) * sample_rate)),
                "B": round(((0.29) * sample_rate)),
                "C": round(((0.3) * sample_rate)),
            },
        ],
        [
            {
                "A": round(((0.25 + 9) * sample_rate)),
                "B": round(((0.26 + 2) * sample_rate)),
                "C": round(((0.27 + 43) * sample_rate)),
            },
            {
                "A": round(((0.28 + 9) * sample_rate)),
                "B": round(((0.29 + 2) * sample_rate)),
                "C": round(((0.3 + 43) * sample_rate)),
            },
        ],
    ]
    np.testing.assert_equal(time_delay_idx, expected_time_delay_idx)
