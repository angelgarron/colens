"""Functions related to the background estimation."""

import numpy as np

from colens.detector import MyDetector


def slide_limiter(
    segment_length_seconds: int | float, slide_shift_seconds: int | float, nifos: int
) -> np.int32:
    """
    This function computes the number of shortslides used by the coherent
    matched filter statistic to obtain as most background triggers as
    possible.

    It bounds the number of slides to avoid counting triggers more than once.
    If the data is not time slid, there is a single slide for the zero-lag.

    Args:
        segment_length (int | float): The length (in seconds) of each segment.
        slide_shift (int | float): The interval (in seconds) of the slides.
        nifos (int): The number of detectors for which you want to compute slides.

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


def get_time_delay_indices(
    num_slides,
    SLIDE_SHIFT,
    LENSED_INSTRUMENTS,
    UNLENSED_INSTRUMENTS,
    sky_positions,
    TRIGGER_TIMES,
    INSTRUMENTS,
    sky_pos_indices,
    SAMPLE_RATE,
):
    # Create a dictionary of time slide shifts; IFO 0 is unshifted
    # ANGEL: Just lensed detectors are shifted
    slide_ids = np.arange(num_slides)
    time_slides = {
        ifo: SLIDE_SHIFT * slide_ids * ifo_idx
        for ifo_idx, ifo in enumerate(LENSED_INSTRUMENTS)
    }
    time_slides.update(
        {ifo: time_slides[LENSED_INSTRUMENTS[0]] for ifo in UNLENSED_INSTRUMENTS}
    )
    # Given the time delays wrt to IFO 0 in time_slides, create a dictionary
    # for time delay indices evaluated wrt the geocenter, in units of samples,
    # i.e. (time delay from geocenter + time slide)*sampling_rate
    time_delay_idx_zerolag = {
        position_index: {
            ifo: MyDetector(ifo).time_delay_from_earth_center(
                sky_positions[0][position_index],
                sky_positions[1][position_index],
                TRIGGER_TIMES[ifo],
            )
            for ifo in INSTRUMENTS
        }
        for position_index in sky_pos_indices
    }
    time_delay_idx = {
        slide: {
            position_index: {
                ifo: int(
                    round(
                        (
                            time_delay_idx_zerolag[position_index][ifo]
                            + time_slides[ifo][slide]
                        )
                        * SAMPLE_RATE
                    )
                )
                for ifo in INSTRUMENTS
            }
            for position_index in sky_pos_indices
        }
        for slide in slide_ids
    }
    return time_slides, time_delay_idx
