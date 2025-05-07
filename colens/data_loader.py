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
    def __init__(self, conf, output_data):
        self.conf = conf
        self.output_data = output_data
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
        self.sigma = []
        self.snrs_original = []
        self.snrs_lensed = []
        self.segments_original = []
        self.segments_lensed = []
        self.injection_parameters = copy(conf.injection_parameters)
        for ifo in conf.injection.unlensed_instruments:
            self.single_detector_setup(ifo, False)
        for ifo in conf.injection.lensed_instruments:
            self.single_detector_setup(ifo, True)

    def single_detector_setup(self, ifo, lensed):
        if lensed:
            self.injection_parameters.geocent_time = (
                self.conf.injection.time_gps_future_seconds
            )
            ifo_real_name = ifo[:2]
        else:
            ifo_real_name = ifo
        strain = self.create_injections(
            ifo_real_name,
            self.conf.injection.gps_start_seconds[ifo],
            self.conf.injection.gps_end_seconds[ifo],
        )
        strain = process_strain(
            strain,
            self.conf.psd.strain_high_pass_hertz,
            self.conf.injection.sample_rate,
            self.conf.injection.pad_seconds,
        )

        segments = self.get_segments(strain, ifo)
        matched_filter = self.get_matched_filter(segments)
        sigmasq = self.template.sigmasq(segments[self.segment_index].psd)
        sigma = np.sqrt(sigmasq)
        self.sigma.append(sigma)
        self.output_data.__getattribute__(ifo).sigma.append(sigma)
        snr_ts, norm, corr, ind, snrv = matched_filter.matched_filter_and_cluster(
            self.segment_index, sigmasq, window=0
        )
        if lensed:
            self.segments_lensed.append(segments)
            self.snrs_lensed.append(
                snr_ts[matched_filter.segments[self.segment_index].analyze] * norm
            )
        else:
            self.segments_original.append(segments)
            self.snrs_original.append(
                snr_ts[matched_filter.segments[self.segment_index].analyze] * norm
            )

    def get_segments(self, strain, ifo):
        segments = StrainSegments(
            strain,
            segment_length=self.conf.injection.segment_length_seconds,
            segment_start_pad=self.conf.injection.segment_start_pad_seconds,
            segment_end_pad=self.conf.injection.segment_end_pad_seconds,
            trigger_start=self.conf.injection.trig_start_time_seconds[ifo],
            trigger_end=self.conf.injection.trig_end_time_seconds[ifo],
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

    def create_injections(self, ifo_real_name, gps_start_seconds, gps_end_seconds):
        # The extra padding we are adding here is going to get removed after highpassing
        return get_strain_list_from_bilby_simulation(
            self.injection_parameters.asdict(),
            [ifo_real_name],
            start_time=gps_start_seconds - self.conf.injection.pad_seconds,
            end_time=gps_end_seconds + self.conf.injection.pad_seconds,
            low_frequency_cutoff=self.conf.injection.low_frequency_cutoff,
            reference_frequency=self.conf.injection.reference_frequency,
            sampling_frequency=self.conf.injection.sample_rate,
            seed=gps_start_seconds,
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
