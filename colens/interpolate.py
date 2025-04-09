import numpy as np


def _lagrange_interpolate_timeseries_at(time, timeseries, index):
    x0, x1, x2 = np.array(timeseries.sample_times)[index - 1 : index + 2]
    y0, y1, y2 = np.array(timeseries)[index - 1 : index + 2]
    return (
        y0 * (time - x1) * (time - x2) / ((x0 - x1) * (x0 - x2))
        + y1 * (time - x0) * (time - x2) / ((x1 - x0) * (x1 - x2))
        + y2 * (time - x0) * (time - x1) / ((x2 - x0) * (x2 - x1))
    )


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
):
    index_trigger = _get_index(
        trigger_time_seconds,
        gps_start_seconds,
        sample_rate,
        time_delay_idx,
        cumulative_index,
    )
    snr_at_trigger = _lagrange_interpolate_timeseries_at(
        time=trigger_time_seconds
        + time_delay_zerolag_seconds
        + time_slides_seconds
        - gps_start_seconds,
        timeseries=timeseries,
        index=index_trigger,
    )
    return snr_at_trigger


def get_snr_interpolated_numpy(
    time_delay_zerolag_seconds,
    timeseries,
    trigger_time_seconds,
    gps_start_seconds,
    sample_rate,
    time_delay_idx,
    cumulative_index,
    time_slides_seconds,
):
    return np.interp(
        trigger_time_seconds,
        np.array(timeseries.sample_times),
        np.array(timeseries),
    )
