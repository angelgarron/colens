"""Functions related to the background estimation."""

import numpy as np


def slide_limiter(segment_length, slide_shift, instruments):
    """
    This function computes the number of shortslides used by the coherent
    matched filter statistic to obtain as most background triggers as
    possible.

    It bounds the number of slides to avoid counting triggers more than once.
    If the data is not time slid, there is a single slide for the zero-lag.
    """
    low, upp = 1, segment_length
    n_ifos = len(instruments)
    stride_dur = segment_length / 2
    num_slides = np.int32(1 + np.floor(stride_dur / (slide_shift * (n_ifos - 1))))
    assert np.logical_and(num_slides >= low, num_slides <= upp), (
        "the combination (slideshift, segment_dur)"
        f" = ({slide_shift:.2f},{stride_dur*2:.2f})"
        f" goes over the allowed upper bound {upp}"
    )
    return num_slides
