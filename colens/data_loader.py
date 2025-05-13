import logging
from copy import copy

import numpy as np
from pycbc import DYN_RANGE_FAC
from pycbc.filter import MatchedFilterControl
from pycbc.psd import associate_psds_to_segments
from pycbc.strain import StrainSegments
from pycbc.types import complex64, float32, zeros

from colens.bank import MyFilterBank
from colens.injection import (
    get_ifos_with_simulated_noise,
    get_strain_list_from_bilby_simulation,
)
from colens.strain import process_strain, process_strain_dict


class DataLoader:
    def __init__(
        self,
        conf,
        output_data,
        instruments,
        lensed_or_unlensed_output,
        time_gps_seconds,
        gps_start_seconds,
        gps_end_seconds,
        trig_start_time_seconds,
        trig_end_time_seconds,
    ):
        self.conf = conf
        self.output_data = output_data
        self.instruments = instruments
        self.lensed_or_unlensed_output = lensed_or_unlensed_output
        self.time_gps_seconds = time_gps_seconds
        self.gps_start_seconds = gps_start_seconds
        self.gps_end_seconds = gps_end_seconds
        self.trig_start_time_seconds = trig_start_time_seconds
        self.trig_end_time_seconds = trig_end_time_seconds
        self.delta_f = (
            1.0 / self.conf.injection.segment_length_seconds
        )  # frequency step of the fourier transform of each segment
        self.segment_length = int(
            self.conf.injection.segment_length_seconds * self.conf.injection.sample_rate
        )  # number of samples of each segment
        self.frequency_length = int(
            self.segment_length // 2 + 1
        )  # number of samples of the fourier transform of each segment
        self.template_mem = zeros(self.segment_length, dtype=complex64)
        logging.info("Read in template bank")
        self.template = self.create_template_bank()[0]
        # TODO loop over segments (or maybe we just create a big segment)
        self.segment_index = 0
        self.matched_filters = []
        self.injection_parameters = copy(conf.injection_parameters)
        for ifo in self.instruments:
            self.single_detector_setup(ifo)
        self.single_segment_setup()

    def single_detector_setup(self, ifo):
        self.injection_parameters.geocent_time = self.time_gps_seconds
        strain = self.create_injections(ifo)
        strain = process_strain(
            strain,
            self.conf.psd.strain_high_pass_hertz,
            self.conf.injection.sample_rate,
            self.conf.injection.pad_seconds,
        )

        segments = self.get_segments(strain)
        matched_filter = self.get_matched_filter(segments)
        self.matched_filters.append(matched_filter)

    def single_segment_setup(self):
        self.sigma = []
        self.snrs = []
        for i in range(len(self.instruments)):
            sigmasq = self.template.sigmasq(
                self.matched_filters[i].segments[self.segment_index].psd
            )
            sigma = np.sqrt(sigmasq)
            self.sigma.append(sigma)
            snr_ts, norm, corr, ind, snrv = self.matched_filters[
                i
            ].matched_filter_and_cluster(self.segment_index, sigmasq, window=0)
            self.snrs.append(
                snr_ts[self.matched_filters[i].segments[self.segment_index].analyze]
                * norm
            )
            self.lensed_or_unlensed_output[i].sigma.append(sigma)

    def get_segments(self, strain):
        segments = StrainSegments(
            strain,
            segment_length=self.conf.injection.segment_length_seconds,
            segment_start_pad=self.conf.injection.segment_start_pad_seconds,
            segment_end_pad=self.conf.injection.segment_end_pad_seconds,
            trigger_start=self.trig_start_time_seconds,
            trigger_end=self.trig_end_time_seconds,
            filter_inj_only=False,
            allow_zero_padding=False,
        ).fourier_segments()

        associate_psds_to_segments(
            self.conf.psd,
            segments,
            strain,
            self.frequency_length,
            self.delta_f,
            self.conf.injection.low_frequency_cutoff,
            dyn_range_factor=DYN_RANGE_FAC,
            precision="single",
        )

        # Overwhiten segments
        for seg in segments:
            seg /= seg.psd

        return segments

    def get_matched_filter(self, segment):
        return MatchedFilterControl(
            self.conf.injection.low_frequency_cutoff,
            None,
            self.conf.injection.sngl_snr_threshold,
            self.segment_length,
            self.delta_f,
            complex64,
            segment,
            self.template_mem,
            use_cluster=False,
            downsample_factor=self.conf.injection.downsample_factor,
            upsample_threshold=self.conf.injection.upsample_threshold,
            upsample_method=self.conf.injection.upsample_method,
        )

    def create_injections(self, ifo):
        # The extra padding we are adding here is going to get removed after highpassing
        return get_strain_list_from_bilby_simulation(
            self.injection_parameters.asdict(),
            [ifo],
            start_time=self.gps_start_seconds - self.conf.injection.pad_seconds,
            end_time=self.gps_end_seconds + self.conf.injection.pad_seconds,
            low_frequency_cutoff=self.conf.injection.low_frequency_cutoff,
            reference_frequency=self.conf.injection.reference_frequency,
            sampling_frequency=self.conf.injection.sample_rate,
            seed=self.gps_start_seconds,
            approximant=self.conf.injection.approximant,
            get_ifos_function=get_ifos_with_simulated_noise,
        )[0]

    def create_template_bank(self) -> MyFilterBank:
        template_parameters = {
            "mass1": np.array([79.45]),
            "mass2": np.array([48.50]),
            "spin1z": np.array([0.60]),
            "spin2z": np.array([0.05]),
            "f_final": np.array([2048.0]),
            "f_ref": np.array([self.conf.injection.reference_frequency]),
        }
        return MyFilterBank(
            filter_length=self.frequency_length,
            delta_f=self.delta_f,
            dtype=complex64,
            template_parameters=template_parameters,
            low_frequency_cutoff=self.conf.injection.low_frequency_cutoff,
            phase_order=self.conf.injection.order,
            approximant=self.conf.injection.approximant,
            out=self.template_mem,
        )
