import numpy as np
from scipy.interpolate import interp1d


def _interpolate_timeseries_at(time, timeseries, index, margin):
    value = interp1d(
        np.array(timeseries.sample_times)[index - margin : index + margin],
        np.array(timeseries)[index - margin : index + margin],
    )(time)
    return value


def _get_index(
    trigger_time_seconds,
    gps_start_seconds,
    sample_rate,
    time_delay_idx,
    cumulative_index,
):
    index_trigger = int(
        (trigger_time_seconds - gps_start_seconds) * sample_rate
        + time_delay_idx
        - cumulative_index
    )
    return index_trigger


def get_snr(
    time_delay_zerolag_seconds,
    timeseries,
    trigger_time_seconds,
    gps_start_seconds,
    sample_rate,
    time_delay_idx,
    cumulative_index,
    time_slides_seconds,
    margin,
):
    index_trigger = _get_index(
        trigger_time_seconds,
        gps_start_seconds,
        sample_rate,
        time_delay_idx,
        cumulative_index,
    )
    snr_at_trigger = timeseries[index_trigger]
    return snr_at_trigger


def get_snr_interpolated(
    time_delay_zerolag_seconds,
    timeseries,
    trigger_time_seconds,
    gps_start_seconds,
    sample_rate,
    time_delay_idx,
    cumulative_index,
    time_slides_seconds,
    margin,
):
    index_trigger = _get_index(
        trigger_time_seconds,
        gps_start_seconds,
        sample_rate,
        time_delay_idx,
        cumulative_index,
    )
    snr_at_trigger = _interpolate_timeseries_at(
        time=trigger_time_seconds
        + time_delay_zerolag_seconds
        + time_slides_seconds
        - gps_start_seconds,
        timeseries=timeseries,
        index=index_trigger,
        margin=margin,
    )
    return snr_at_trigger
