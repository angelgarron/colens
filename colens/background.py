"""Functions related to the background estimation."""

import numpy as np

from colens.sky import SkyGrid


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


def get_time_delay_at_zerolag_seconds(
    trigger_times_seconds: float,
    sky_grid: SkyGrid,
    detectors,
):
    """Compute the difference of arrival time between the earth center and each one of the `instruments` of a signal
    coming from each point in `sky_grid`, .i.e. (t_{instrument}-t_{center}).

    Args:
        trigger_times_seconds (float): Trigger time (in seconds).
        sky_grid (SkyGrid): The sky grid containing the positions in the sky for which the time delay \
        should be computed.
    """
    time_delay_zerolag_seconds = [
        {
            ifo: detectors[ifo].time_delay_from_earth_center(
                sky_position.ra,
                sky_position.dec,
                trigger_times_seconds,
            )
            for ifo in detectors
        }
        for sky_position in sky_grid
    ]
    return time_delay_zerolag_seconds


def get_time_slides_seconds(
    num_slides: int,
    slide_shift_seconds: float,
    unlensed_instruments: list[str],
    lensed_instruments: list[str],
):
    """Create a dictionary of time slide shifts; IFO 0 is unshifted.
    Just the lensed detectors are shifted.

    Args:
        num_slides (int): Number of time slides we wish to compute.
        slide_shift_seconds (float): The time difference (in seconds) between slides.
        unlensed_instruments (list[str]): List of unlensed instruments.
        lensed_instruments (list[str]): List of lensed instruments.
    """
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
    return time_slides_seconds


def get_time_delay_indices(
    sample_rate: float,
    time_delay_zerolag_seconds: list[dict[str, float]],
    time_slides_seconds: dict[str, np.ndarray],
):
    """Given the time delays wrt to IFO 0 in time_slides, create a dictionary
    for time delay indices evaluated wrt the geocenter, in units of samples,
    i.e. (time delay from geocenter + time slide)*sampling_rate

    Args:
        sample_rate (float): Sample rate (in Hertz).
        time_delay_zerolag_seconds (list[dict[str, float]]): Time difference (in seconds) \
        of arrival time between earth center and instruments of a grid of sky positions.
        time_slides_seconds (dict[str, np.ndarray]): Time slides (in seconds) for each instrument.
    """
    slide_ids = np.arange(len(list(time_slides_seconds.values())[0]))
    time_delay_idx = [
        [
            {
                ifo: round(
                    (
                        time_delay_zerolag_at_sky_position_seconds[ifo]
                        + time_slides_seconds[ifo][slide]
                    )
                    * sample_rate
                )
                for ifo in time_delay_zerolag_at_sky_position_seconds
            }
            for time_delay_zerolag_at_sky_position_seconds in time_delay_zerolag_seconds
        ]
        for slide in slide_ids
    ]
    return time_delay_idx
