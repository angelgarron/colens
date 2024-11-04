"""Functions to manage loading data."""

from pycbc.frame import read_frame


def get_strain_dict_from_files(
    frame_files, channels, instruments, gps_start, gps_end, pad
):
    strain_dict = dict()
    for ifo in instruments:
        strain_tmp = read_frame(
            frame_files[ifo],
            channels[ifo],
            start_time=gps_start[ifo] - pad,
            end_time=gps_end[ifo] + pad,
            sieve=None,
        )
        strain_dict[ifo] = strain_tmp
    return strain_dict
