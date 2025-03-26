from __future__ import annotations

from dataclasses import dataclass, field
from typing import Type

import numpy as np
import yaml


@dataclass
class Data:
    frame_files: dict
    channels: dict

    @classmethod
    def from_dict(cls: Type[Data], obj: dict) -> Data:
        return cls(
            frame_files=obj["frame_files"],
            channels=obj["channels"],
        )


@dataclass
class Injection:
    time_gps_past_seconds: float
    time_gps_future_seconds: float
    gps_start_seconds: dict = field(init=False)
    gps_end_seconds: dict = field(init=False)
    trig_start_time_seconds: dict = field(init=False)
    trig_end_time_seconds: dict = field(init=False)
    ra: float
    dec: float
    sample_rate: float
    unlensed_instruments: list
    lensed_instruments: list
    instruments: list = field(init=False)
    segment_length_seconds: float
    slide_shift_seconds: float
    low_frequency_cutoff: float
    reference_frequency: float
    sngl_snr_threshold: float
    downsample_factor: int
    upsample_threshold: float
    upsample_method: str
    bank_file: str
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
        self.instruments = self.lensed_instruments + self.unlensed_instruments
        self.instruments.sort()

        trigger_times_seconds = {
            "H1": self.time_gps_past_seconds,
            "L1": self.time_gps_past_seconds,
            "H1_lensed": self.time_gps_future_seconds,
            "L1_lensed": self.time_gps_future_seconds,
        }
        self.gps_start_seconds = dict()
        self.gps_end_seconds = dict()
        for ifo in self.instruments:
            self.gps_start_seconds[ifo] = (
                int(trigger_times_seconds[ifo]) - 192 - self.pad_seconds
            )
            self.gps_end_seconds[ifo] = (
                int(trigger_times_seconds[ifo]) + 192 + self.pad_seconds
            )

        self.trig_start_time_seconds = dict()
        self.trig_end_time_seconds = dict()
        for ifo in self.instruments:
            self.trig_start_time_seconds[ifo] = (
                self.gps_start_seconds[ifo] + self.segment_start_pad_seconds
            )
            self.trig_end_time_seconds[ifo] = (
                self.gps_end_seconds[ifo] - self.segment_end_pad_seconds
            )

    @classmethod
    def from_dict(cls: Type[Injection], obj: dict) -> Injection:
        return cls(
            time_gps_past_seconds=obj["time_gps_past_seconds"],
            time_gps_future_seconds=obj["time_gps_future_seconds"],
            ra=obj["ra"],
            dec=obj["dec"],
            sample_rate=obj["sample_rate"],
            unlensed_instruments=obj["unlensed_instruments"],
            lensed_instruments=obj["lensed_instruments"],
            segment_length_seconds=obj["segment_length_seconds"],
            slide_shift_seconds=obj["slide_shift_seconds"],
            low_frequency_cutoff=obj["low_frequency_cutoff"],
            reference_frequency=obj["reference_frequency"],
            sngl_snr_threshold=obj["sngl_snr_threshold"],
            downsample_factor=obj["downsample_factor"],
            upsample_threshold=obj["upsample_threshold"],
            upsample_method=obj["upsample_method"],
            bank_file=obj["bank_file"],
            order=obj["order"],
            taper_template=obj["taper_template"],
            approximant=obj["approximant"],
            coinc_threshold=obj["coinc_threshold"],
            do_null_cut=obj["do_null_cut"],
            null_min=obj["null_min"],
            null_grad=obj["null_grad"],
            null_step=obj["null_step"],
            cluster_window=obj["cluster_window"],
            pad_seconds=obj["pad_seconds"],
            segment_start_pad_seconds=obj["segment_start_pad_seconds"],
            segment_end_pad_seconds=obj["segment_end_pad_seconds"],
        )


@dataclass
class Chisq:
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

    @classmethod
    def from_dict(cls: Type[Chisq], obj: dict) -> Chisq:
        return cls(
            chisq_bins=obj["chisq_bins"],
            autochi_stride=obj["autochi_stride"],
            autochi_number_points=obj["autochi_number_points"],
            autochi_onesided=obj["autochi_onesided"],
            autochi_two_phase=obj["autochi_two_phase"],
            autochi_reverse_template=obj["autochi_reverse_template"],
            autochi_max_valued=obj["autochi_max_valued"],
            autochi_max_valued_dof=obj["autochi_max_valued_dof"],
            chisq_index=obj["chisq_index"],
            chisq_nhigh=obj["chisq_nhigh"],
        )


@dataclass
class Psd:
    psd_segment_stride_seconds: float
    psd_segment_length_seconds: float
    psd_num_segments: int
    strain_high_pass_hertz: float

    @classmethod
    def from_dict(cls: Type[Psd], obj: dict) -> Psd:
        return cls(
            psd_segment_stride_seconds=obj["psd_segment_stride_seconds"],
            psd_segment_length_seconds=obj["psd_segment_length_seconds"],
            psd_num_segments=obj["psd_num_segments"],
            strain_high_pass_hertz=obj["strain_high_pass_hertz"],
        )


@dataclass
class SkyPatch:
    angular_spacing: float
    sky_error: float

    @classmethod
    def from_dict(cls: Type[SkyPatch], obj: dict) -> SkyPatch:
        return cls(
            angular_spacing=obj["angular_spacing"] * np.pi / 180,
            sky_error=obj["sky_error"] * np.pi / 180,
        )


@dataclass
class Output:
    output_file_name: str

    @classmethod
    def from_dict(cls: Type[SkyPatch], obj: dict) -> SkyPatch:
        return cls(
            output_file_name=obj["output_file_name"],
        )


@dataclass
class InjectionParameters:
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

    @classmethod
    def from_dict(cls: Type[InjectionParameters], obj: dict) -> InjectionParameters:
        return cls(
            mass_1=obj["mass_1"],
            mass_2=obj["mass_2"],
            a_1=obj["a_1"],
            a_2=obj["a_2"],
            tilt_1=obj["tilt_1"],
            tilt_2=obj["tilt_2"],
            phi_12=obj["phi_12"],
            phi_jl=obj["phi_jl"],
            luminosity_distance=obj["luminosity_distance"],
            theta_jn=obj["theta_jn"],
            psi=obj["psi"],
            phase=obj["phase"],
            geocent_time=obj["geocent_time"],
            ra=obj["ra"],
            dec=obj["dec"],
        )


@dataclass
class Configuration:
    data: Data
    injection: Injection
    chisq: Chisq
    psd: Psd
    sky_patch: SkyPatch
    output: Output
    injection_parameters: InjectionParameters

    @classmethod
    def from_dict(cls: Type[Configuration], obj: dict) -> Configuration:
        return cls(
            data=Data.from_dict(obj["data"]),
            injection=Injection.from_dict(obj["injection"]),
            chisq=Chisq.from_dict(obj["chisq"]),
            psd=Psd.from_dict(obj["psd"]),
            sky_patch=SkyPatch.from_dict(obj["sky_patch"]),
            output=Output.from_dict(obj["output"]),
            injection_parameters=InjectionParameters.from_dict(
                obj["injection_parameters"]
            ),
        )


def read_configuration_from(filename: str) -> Configuration:
    with open(filename, "r") as file:
        config = yaml.safe_load(file)
    return Configuration.from_dict(config)
