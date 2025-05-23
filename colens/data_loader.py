import logging
from copy import copy

import numpy as np
from pycbc import DYN_RANGE_FAC
from pycbc.filter import MatchedFilterControl
from pycbc.psd import associate_psds_to_segments
from pycbc.strain import StrainSegments
from pycbc.types import complex64

from colens.injection import (
    get_ifos_with_simulated_noise,
    get_strain_list_from_bilby_simulation,
)
from colens.strain import process_strain, process_strain_dict


class DataLoader:
    def __init__(
        self,
        conf,
        instruments,
        per_detector_output,
        time_gps_seconds,
        gps_start_seconds,
        gps_end_seconds,
        delta_f,
        segment_length,
        frequency_length,
        template_mem,
        template,
    ):
        self.conf = conf
        self.instruments = instruments
        self.per_detector_output = per_detector_output
        self.time_gps_seconds = time_gps_seconds
        self.gps_start_seconds = gps_start_seconds
        self.gps_end_seconds = gps_end_seconds
        self.delta_f = delta_f
        self.segment_length = segment_length
        self.frequency_length = frequency_length
        self.template_mem = template_mem
        self.template = template
        self.matched_filters = []
        self.injection_parameters = copy(conf.injection_parameters)
        for ifo in self.instruments:
            self.single_detector_setup(ifo)

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

    def single_segment_setup(self, segment_index):
        self.sigma = []
        self.snrs = []
        for i in range(len(self.instruments)):
            sigmasq = self.template.sigmasq(
                self.matched_filters[i].segments[segment_index].psd
            )
            sigma = np.sqrt(sigmasq)
            self.sigma.append(sigma)
            snr_ts, norm, corr, ind, snrv = self.matched_filters[
                i
            ].matched_filter_and_cluster(segment_index, sigmasq, window=0)
            self.snrs.append(
                snr_ts[self.matched_filters[i].segments[segment_index].analyze] * norm
            )
            self.per_detector_output[i].sigma.append(sigma)

    def get_segments(self, strain):
        segments = StrainSegments(
            strain,
            segment_length=self.conf.injection.segment_length_seconds,
            segment_start_pad=self.conf.injection.segment_start_pad_seconds,
            segment_end_pad=self.conf.injection.segment_end_pad_seconds,
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
