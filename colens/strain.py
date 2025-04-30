"""Functions related to the processing of the strain."""

from pycbc import DYN_RANGE_FAC
from pycbc.filter import highpass, resample_to_delta_t
from pycbc.types import float32
from pycbc.types.timeseries import TimeSeries


def process_strain_dict(
    strain_dict: dict[str, TimeSeries],
    strain_high_pass_hertz: float,
    sample_rate: int,
    pad_seconds: float,
) -> None:
    """Modify (in-place) the `strain_dict` to apply filters to the timeseries
    and discard `pad_seconds` of corrupted data at the start and end after filtering and
    also applying the `DYN_RANGE_FAC`.

    Args:
        strain_dict (dict[str, TimeSeries]): Dictionary of timeseries for each detector.
        strain_high_pass_hertz (float): Lower frequency for the high-pass filters.
        sample_rate (int): Sample rate to resample the timeseries.
        pad_seconds (float): Padding (in seconds) that should be added at the start and \
        end of each timeseries.
    """
    for ifo in strain_dict:
        strain_dict[ifo] = process_strain(
            strain_dict[ifo], strain_high_pass_hertz, sample_rate, pad_seconds
        )


def process_strain(
    strain,
    strain_high_pass_hertz: float,
    sample_rate: int,
    pad_seconds: float,
) -> None:
    strain_tmp = highpass(strain, frequency=strain_high_pass_hertz)
    strain_tmp = resample_to_delta_t(strain_tmp, 1.0 / sample_rate, method="ldas")
    strain_tmp = (strain_tmp * DYN_RANGE_FAC).astype(float32)
    strain_tmp = highpass(strain_tmp, frequency=strain_high_pass_hertz)
    start = int(pad_seconds * sample_rate)
    end = int(len(strain_tmp) - sample_rate * pad_seconds)
    strain_tmp = strain_tmp[start:end]
    return strain_tmp
