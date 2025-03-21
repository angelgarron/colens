"""Functions to manage loading and saving data."""

import json
from dataclasses import asdict, dataclass, field

import h5py
from pycbc.frame import read_frame

from colens.data import NumpyArrayEncoder


@dataclass
class PerDetectorOutput:
    snr_real: list = field(default_factory=list)
    snr_imag: list = field(default_factory=list)
    sigma: list = field(default_factory=list)
    chisq_dof: list = field(default_factory=list)
    chisq: list = field(default_factory=list)


@dataclass
class Output:
    H1: PerDetectorOutput
    L1: PerDetectorOutput
    H1_lensed: PerDetectorOutput
    L1_lensed: PerDetectorOutput
    original_trigger_time_seconds: list = field(default_factory=list)
    lensed_trigger_time_seconds: list = field(default_factory=list)
    time_slide_index: list = field(default_factory=list)
    ra: list = field(default_factory=list)
    dec: list = field(default_factory=list)
    rho_coinc: list = field(default_factory=list)
    rho_coh: list = field(default_factory=list)
    network_chisq_values: list = field(default_factory=list)
    reweighted_snr: list = field(default_factory=list)
    reweighted_by_null_snr: list = field(default_factory=list)

    def write_to_json(self, output_file):
        with open(output_file, "w") as file:
            json.dump(asdict(self), file, cls=NumpyArrayEncoder, indent=4)


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
    reference_frequency: float,
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
        reference_frequency (float): Reference frequency.
    """
    with h5py.File(bank_file, "w") as file:
        for key, value in {
            "appoximant": approximant,
            "f_lower": low_frequency_cutoff,
            "mass1": mass1,
            "mass2": mass2,
            "spin1z": spin1z,
            "spin2z": spin2z,
            "delta_f": 0.0625,
            "f_final": 2048.0,
            "f_ref": reference_frequency,
        }.items():
            file.create_dataset(
                key, data=[value], compression="gzip", compression_opts=9, shuffle=True
            )
