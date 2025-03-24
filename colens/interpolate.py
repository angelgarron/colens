import numpy as np
from scipy.interpolate import interp1d


def interpolate_timeseries_at(time, timeseries, index, margin):
    value = interp1d(
        np.array(timeseries.sample_times)[index - margin : index + margin],
        np.array(timeseries)[index - margin : index + margin],
    )(time)
    return value
