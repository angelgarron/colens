"""Functions related to the background estimation."""

import numpy as np


def slide_limiter(
    segment_length_seconds: float, slide_shift_seconds: float, nifos: int
) -> np.int32:
    """
    Compute the number of shortslides used by the coherent
    matched filter statistic to obtain as most background triggers as
    possible.

    It bounds the number of slides to avoid counting triggers more than once.
    If the data is not time slid, there is a single slide for the zero-lag.

    Args:
        segment_length_seconds (float): The length (in seconds) of each segment.
        slide_shift_seconds (float): The time difference (in seconds) between slides.
        nifos (int): The number of detectors for which we want to compute slides.

    Returns:
        np.int32: Number of time slides that are going to be performed.
    """
    upp_seconds = segment_length_seconds
    stride_dur_seconds = segment_length_seconds / 2
    num_slides = np.int32(
        1 + np.floor(stride_dur_seconds / (slide_shift_seconds * (nifos - 1)))
    )  # the "1 +" is to account for the zero-lag slide
    assert num_slides * slide_shift_seconds <= upp_seconds, (
        "the combination (slideshift, segment_dur)"
        f" = ({slide_shift_seconds:.2f},{stride_dur_seconds*2:.2f})"
        f" goes over the allowed upper bound {upp_seconds}"
    )
    return num_slides


def get_time_slides_seconds(
    num_slides: int,
    slide_shift_seconds: float,
    instruments: list[str],
):
    """Create a dictionary of time slide shifts; IFO 0 is unshifted.

    Args:
        num_slides (int): Number of time slides we wish to compute.
        slide_shift_seconds (float): The time difference (in seconds) between slides.
        instruments (list[str]): List of instruments.
    """
    slide_ids = np.arange(num_slides)
    time_slides_seconds = [
        slide_shift_seconds * slide_ids * ifo_idx for ifo_idx in range(len(instruments))
    ]
    return time_slides_seconds
