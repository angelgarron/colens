"""Dataclasses to store the configuration read from a yaml file."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Iterable, Type, TypeVar

import numpy as np
import yaml


class UnknownSectionError(Exception):
    def __init__(self, unknown_section: Iterable[str]):
        self.message = f"The following sections are not known: {', '.join(sorted(unknown_section))}"
        super().__init__(self.message)


class UnknownSubsectionError(Exception):
    def __init__(self, unknown_subsection: Iterable[str]):
        self.message = f"The following subsections are not known: {', '.join(sorted(unknown_subsection))}"
        super().__init__(self.message)


def _check_unknown_entries(expected, given, error):
    unknown_entry = set(given) - set(expected)
    if unknown_entry:
        raise error(unknown_entry)


def _construct_subsection_dict(match_args: tuple[str], obj: dict) -> dict:
    _check_unknown_entries(match_args, obj.keys(), UnknownSubsectionError)
    subsection_dict = dict()
    not_overwritten = []
    for key in match_args:
        try:
            subsection_dict[key] = obj[key]
        except KeyError:
            not_overwritten.append(key)
            continue
    if len(not_overwritten) > 0:
        print(f"Leaving {', '.join(not_overwritten)} with their default values.")
    return subsection_dict


T = TypeVar("T")


@dataclass
class SubsectionConfiguration:
    @classmethod
    def from_dict(cls: Type[T], obj: dict) -> T:
        return cls(**_construct_subsection_dict(cls.__match_args__, obj))

    def asdict(self) -> dict:
        return asdict(self)


@dataclass
class Data(SubsectionConfiguration):
    frame_files: dict
    channels: dict
    posteriors_file: str


@dataclass
class Injection(SubsectionConfiguration):
    time_gps_past_seconds: float
    time_gps_future_seconds: float
    gps_start_seconds: dict = field(init=False)
    gps_end_seconds: dict = field(init=False)
    trig_start_time_seconds: dict = field(init=False)
    trig_end_time_seconds: dict = field(init=False)
    sample_rate: float
    unlensed_instruments: list
    lensed_instruments: list
    segment_length_seconds: float
    slide_shift_seconds: float
    low_frequency_cutoff: float
    reference_frequency: float
    sngl_snr_threshold: float
    downsample_factor: int
    upsample_threshold: float
    upsample_method: str
    order: str
    taper_template: bool
    approximant: str
    coinc_threshold: float
    do_null_cut: bool
    null_min: float
    null_grad: float
    null_step: float
    cluster_window: float
    pad_seconds: float
    segment_start_pad_seconds: float
    segment_end_pad_seconds: float

    def __post_init__(self):
        self.gps_start_seconds = {
            "past": int(self.time_gps_past_seconds) - 192 - self.pad_seconds,
            "future": int(self.time_gps_future_seconds) - 192 - self.pad_seconds,
        }
        self.gps_end_seconds = {
            "past": int(self.time_gps_past_seconds) + 192 + self.pad_seconds,
            "future": int(self.time_gps_future_seconds) + 192 + self.pad_seconds,
        }

        self.trig_start_time_seconds = {
            "past": self.gps_start_seconds["past"] + self.segment_start_pad_seconds,
            "future": self.gps_start_seconds["future"] + self.segment_start_pad_seconds,
        }
        self.trig_end_time_seconds = {
            "past": self.gps_end_seconds["past"] - self.segment_end_pad_seconds,
            "future": self.gps_end_seconds["future"] - self.segment_end_pad_seconds,
        }


@dataclass
class Chisq(SubsectionConfiguration):
    chisq_bins: str
    autochi_stride: int
    autochi_number_points: int
    autochi_onesided: str
    autochi_two_phase: bool
    autochi_reverse_template: bool
    autochi_max_valued: str
    autochi_max_valued_dof: str
    chisq_index: float
    chisq_nhigh: float


@dataclass
class Psd(SubsectionConfiguration):
    psd_segment_stride_seconds: float
    psd_segment_length_seconds: float
    psd_num_segments: int
    strain_high_pass_hertz: float
    psd_model: str = None
    psd_file: str = None
    asd_file: str = None
    psd_estimation: str = None
    psd_inverse_length: float = None
    # aliases that pycbc functions will recognize
    psd_segment_stride: float = field(init=False)
    psd_segment_length: float = field(init=False)

    def __post_init__(self):
        self.psd_segment_stride = self.psd_segment_stride_seconds
        self.psd_segment_length = self.psd_segment_length_seconds


@dataclass
class Output(SubsectionConfiguration):
    output_file_name: str


@dataclass
class InjectionParameters(SubsectionConfiguration):
    mass_1: float
    mass_2: float
    a_1: float
    a_2: float
    tilt_1: float
    tilt_2: float
    phi_12: float
    phi_jl: float
    luminosity_distance: float
    theta_jn: float
    psi: float
    phase: float
    geocent_time: float
    ra: float
    dec: float


@dataclass
class Configuration:
    data: Data
    injection: Injection
    chisq: Chisq
    psd: Psd
    output: Output
    injection_parameters: InjectionParameters

    @classmethod
    def from_dict(cls: Type[Configuration], obj: dict) -> Configuration:
        _check_unknown_entries(cls.__match_args__, obj.keys(), UnknownSectionError)
        return cls(
            data=Data.from_dict(obj["data"]),
            injection=Injection.from_dict(obj["injection"]),
            chisq=Chisq.from_dict(obj["chisq"]),
            psd=Psd.from_dict(obj["psd"]),
            output=Output.from_dict(obj["output"]),
            injection_parameters=InjectionParameters.from_dict(
                obj["injection_parameters"]
            ),
        )


def read_configuration_from(filename: str) -> Configuration:
    with open(filename, "r") as file:
        config = yaml.safe_load(file)
    return Configuration.from_dict(config)
