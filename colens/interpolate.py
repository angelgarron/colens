import numpy as np
from scipy.interpolate import interp1d


def interpolate_timeseries_at(time, timeseries, index, margin):
    value = interp1d(
        np.array(timeseries.sample_times)[index - margin : index + margin],
        np.array(timeseries)[index - margin : index + margin],
    )(time)
    return value


def get_index(
    original_trigger_time_seconds,
    gps_start_seconds,
    sample_rate,
    unlensed_time_delay_idx,
    cumulative_index,
):
    index_trigger_H1_original = int(
        (original_trigger_time_seconds - gps_start_seconds) * sample_rate
        + unlensed_time_delay_idx
        - cumulative_index
    )
    return index_trigger_H1_original


def get_snr(
    unlensed_time_delay_zerolag_seconds,
    timeseries,
    original_trigger_time_seconds,
    gps_start_seconds,
    sample_rate,
    unlensed_time_delay_idx,
    cumulative_index,
    time_slides_seconds,
    margin,
):
    index_trigger_H1_original = get_index(
        original_trigger_time_seconds,
        gps_start_seconds,
        sample_rate,
        unlensed_time_delay_idx,
        cumulative_index,
    )
    snr_H1_at_trigger_original = timeseries[index_trigger_H1_original]
    return snr_H1_at_trigger_original


def get_snr_interpolated(
    unlensed_time_delay_zerolag_seconds,
    timeseries,
    original_trigger_time_seconds,
    gps_start_seconds,
    sample_rate,
    unlensed_time_delay_idx,
    cumulative_index,
    time_slides_seconds,
    margin,
):
    index_trigger_H1_original = get_index(
        original_trigger_time_seconds,
        gps_start_seconds,
        sample_rate,
        unlensed_time_delay_idx,
        cumulative_index,
    )
    snr_H1_at_trigger_original = interpolate_timeseries_at(
        time=original_trigger_time_seconds
        + unlensed_time_delay_zerolag_seconds
        + time_slides_seconds
        - gps_start_seconds,
        timeseries=timeseries,
        index=index_trigger_H1_original,
        margin=margin,
    )
    return snr_H1_at_trigger_original
