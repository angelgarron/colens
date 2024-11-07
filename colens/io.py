"""Functions to manage loading and saving data."""

import h5py
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


def create_filter_bank(
    mass1: float,
    mass2: float,
    spin1z: float,
    spin2z: float,
    bank_file: str,
    approximant: str,
    low_frequency_cutoff: float,
) -> None:
    """Create a file with the template bank.

    Args:
        mass1 (float): Parameter.
        mass2 (float): Parameter.
        spin1z (float): Parameter.
        spin2z (float): Parameter.
        bank_file (str): File name where to store the template bank.
        approximant (str): Name of the waveform to use.
        low_frequency_cutoff (float): Low frequency cutoff.
    """
    with h5py.File(bank_file, "w") as file:
        for key, value in {
            "appoximant": approximant,
            "f_lower": low_frequency_cutoff,
            "mass1": mass1,
            "mass2": mass2,
            "spin1z": spin1z,
            "spin2z": spin2z,
        }.items():
            file.create_dataset(
                key, data=[value], compression="gzip", compression_opts=9, shuffle=True
            )
