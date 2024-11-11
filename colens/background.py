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
        segment_length_seconds (int | float): The length (in seconds) of each segment.
        slide_shift_seconds (int | float): The interval (in seconds) of the slides.
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


def get_time_delay_at_zerolag_seconds(trigger_times_seconds, sky_grid, instruments):
    time_delay_zerolag_seconds = {
        position_index: {
            ifo: MyDetector(ifo).time_delay_from_earth_center(
                sky_position.ra,
                sky_position.dec,
                trigger_times_seconds[ifo],
            )
            for ifo in instruments
        }
        for position_index, sky_position in enumerate(sky_grid)
    }
    return time_delay_zerolag_seconds


def get_time_delay_indices(
    num_slides,
    slide_shift_seconds,
    lensed_instruments,
    unlensed_instruments,
    sky_grid,
    sample_rate,
    time_delay_zerolag_seconds,
):
    # Create a dictionary of time slide shifts; IFO 0 is unshifted
    # ANGEL: Just lensed detectors are shifted
    slide_ids = np.arange(num_slides)
    time_slides_seconds = {
        ifo: slide_shift_seconds * slide_ids * ifo_idx
        for ifo_idx, ifo in enumerate(lensed_instruments)
    }
    time_slides_seconds.update(
        {
            ifo: time_slides_seconds[lensed_instruments[0]]
            for ifo in unlensed_instruments
        }
    )
    # Given the time delays wrt to IFO 0 in time_slides, create a dictionary
    # for time delay indices evaluated wrt the geocenter, in units of samples,
    # i.e. (time delay from geocenter + time slide)*sampling_rate
    time_delay_idx = {
        slide: {
            position_index: {
                ifo: int(
                    round(
                        (
                            time_delay_zerolag_seconds[position_index][ifo]
                            + time_slides_seconds[ifo][slide]
                        )
                        * sample_rate
                    )
                )
                for ifo in unlensed_instruments + lensed_instruments
            }
            for position_index in range(len(sky_grid))
        }
        for slide in slide_ids
    }
    return time_slides_seconds, time_delay_idx
