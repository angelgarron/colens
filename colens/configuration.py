from __future__ import annotations

import json
from configparser import ConfigParser
from dataclasses import dataclass, field
from typing import Type

import numpy as np


@dataclass
class Data:
    frame_files: dict
    channels: dict

    @classmethod
    def from_dict(cls: Type[Data], obj: ConfigParser) -> Data:
        return cls(
            frame_files=json.loads(obj["frame_files"]),
            channels=json.loads(obj["channels"]),
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
    def from_dict(cls: Type[Injection], obj: ConfigParser) -> Injection:
        return cls(
            time_gps_past_seconds=obj.getfloat("time_gps_past_seconds"),
            time_gps_future_seconds=obj.getfloat("time_gps_future_seconds"),
            ra=obj.getfloat("ra"),
            dec=obj.getfloat("dec"),
            sample_rate=obj.getfloat("sample_rate"),
            unlensed_instruments=json.loads(obj["unlensed_instruments"]),
            lensed_instruments=json.loads(obj["lensed_instruments"]),
            segment_length_seconds=obj.getfloat("segment_length_seconds"),
            slide_shift_seconds=obj.getfloat("slide_shift_seconds"),
            low_frequency_cutoff=obj.getfloat("low_frequency_cutoff"),
            reference_frequency=obj.getfloat("reference_frequency"),
            sngl_snr_threshold=obj.getfloat("sngl_snr_threshold"),
            downsample_factor=obj.getint("downsample_factor"),
            upsample_threshold=obj.getfloat("upsample_threshold"),
            upsample_method=obj["upsample_method"],
            bank_file=obj["bank_file"],
            order=obj["order"],
            taper_template=obj["taper_template"],
            approximant=obj["approximant"],
            coinc_threshold=obj.getfloat("coinc_threshold"),
            do_null_cut=obj.getboolean("do_null_cut"),
            null_min=obj.getfloat("null_min"),
            null_grad=obj.getfloat("null_grad"),
            null_step=obj.getfloat("null_step"),
            cluster_window=obj.getfloat("cluster_window"),
            pad_seconds=obj.getfloat("pad_seconds"),
            segment_start_pad_seconds=obj.getfloat("segment_start_pad_seconds"),
            segment_end_pad_seconds=obj.getfloat("segment_end_pad_seconds"),
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
    def from_dict(cls: Type[Chisq], obj: ConfigParser) -> Chisq:
        return cls(
            chisq_bins=obj["chisq_bins"],
            autochi_stride=obj.getint("autochi_stride"),
            autochi_number_points=obj.getint("autochi_number_points"),
            autochi_onesided=obj["autochi_onesided"],
            autochi_two_phase=obj.getboolean("autochi_two_phase"),
            autochi_reverse_template=obj.getboolean("autochi_reverse_template"),
            autochi_max_valued=obj["autochi_max_valued"],
            autochi_max_valued_dof=obj["autochi_max_valued_dof"],
            chisq_index=obj.getfloat("chisq_index"),
            chisq_nhigh=obj.getfloat("chisq_nhigh"),
        )


@dataclass
class Psd:
    psd_segment_stride_seconds: float
    psd_segment_length_seconds: float
    psd_num_segments: int
    strain_high_pass_hertz: float

    @classmethod
    def from_dict(cls: Type[Psd], obj: ConfigParser) -> Psd:
        return cls(
            psd_segment_stride_seconds=obj.getfloat("psd_segment_stride_seconds"),
            psd_segment_length_seconds=obj.getfloat("psd_segment_length_seconds"),
            psd_num_segments=obj.getint("psd_num_segments"),
            strain_high_pass_hertz=obj.getfloat("strain_high_pass_hertz"),
        )


@dataclass
class SkyPatch:
    angular_spacing: float
    sky_error: float

    @classmethod
    def from_dict(cls: Type[SkyPatch], obj: ConfigParser) -> SkyPatch:
        return cls(
            angular_spacing=obj.getfloat("angular_spacing") * np.pi / 180,
            sky_error=obj.getfloat("sky_error") * np.pi / 180,
        )


@dataclass
class Output:
    output_file_name: str

    @classmethod
    def from_dict(cls: Type[SkyPatch], obj: ConfigParser) -> SkyPatch:
        return cls(
            output_file_name=obj["output_file_name"],
        )


@dataclass
class Configuration:
    data: Data
    injection: Injection
    chisq: Chisq
    psd: Psd
    sky_patch: SkyPatch
    output: Output

    @classmethod
    def from_dict(cls: Type[Configuration], obj: ConfigParser) -> Configuration:
        return cls(
            data=Data.from_dict(obj["data"]),
            injection=Injection.from_dict(obj["injection"]),
            chisq=Chisq.from_dict(obj["chisq"]),
            psd=Psd.from_dict(obj["psd"]),
            sky_patch=SkyPatch.from_dict(obj["sky_patch"]),
            output=Output.from_dict(obj["output"]),
        )


def read_configuration_from(filename: str) -> Configuration:
    config = ConfigParser()
    config.read(filename)
    return Configuration.from_dict(config)
