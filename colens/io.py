"""Functions to manage loading and saving data."""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from pycbc.frame import read_frame


class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


@dataclass
class PerDetectorOutput:
    name: str
    snr_real: list = field(default_factory=list)
    snr_imag: list = field(default_factory=list)
    sigma: list = field(default_factory=list)
    chisq_dof: list = field(default_factory=list)
    chisq: list = field(default_factory=list)


@dataclass
class Output:
    original_output: list[PerDetectorOutput] = field(default_factory=list)
    lensed_output: list[PerDetectorOutput] = field(default_factory=list)
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


def get_bilby_posteriors(filename: Path | str, approximant: str) -> pd.DataFrame:
    with h5py.File(filename) as file:
        return pd.DataFrame(np.array(file[approximant]["posterior_samples"]))
