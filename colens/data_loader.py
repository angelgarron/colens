import logging
from copy import copy

import numpy as np
from pycbc import DYN_RANGE_FAC
from pycbc.detector import Detector
from pycbc.filter import MatchedFilterControl
from pycbc.psd import associate_psds_to_segments
from pycbc.strain import StrainSegments
from pycbc.types import complex64, float32, zeros

from colens.background import (
    get_time_delay_at_zerolag_seconds,
    get_time_delay_indices,
    get_time_slides_seconds,
)
from colens.bank import MyFilterBank
from colens.detector import calculate_antenna_pattern
from colens.injection import (
    get_ifos_with_simulated_noise,
    get_strain_list_from_bilby_simulation,
)
from colens.io import get_bilby_posteriors, get_strain_dict_from_files
from colens.strain import process_strain, process_strain_dict
from colens.timing import get_timing_iterator


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
        bank = self.create_template_bank()
        logging.info("Full template bank size: %d", len(bank))
        self.template = bank[0]
        # TODO loop over segments (or maybe we just create a big segment)
        self.segment_index = 0
        self.sigma = []
        self.snr_dict = dict()
        self.segments = dict()
        self.injection_parameters = copy(conf.injection_parameters)
        self.unlensed_detectors = dict()
        self.lensed_detectors = dict()
        for ifo in conf.injection.unlensed_instruments:
            self.single_detector_setup(ifo, False)
            self.unlensed_detectors[ifo] = Detector(ifo)
        for ifo in conf.injection.lensed_instruments:
            self.single_detector_setup(ifo, True)
            self.lensed_detectors[ifo] = Detector(ifo[:2])
        self.get_timing_iterator()
        # self.num_slides = slide_limiter(
        #     conf.injection.segment_length_seconds,
        #     conf.injection.slide_shift_seconds,
        #     len(conf.injection.lensed_instruments),
        # )
        self.num_slides = 1
        self.time_slides_seconds = get_time_slides_seconds(
            self.num_slides,
            self.conf.injection.slide_shift_seconds,
            list(self.unlensed_detectors),
            list(self.lensed_detectors),
        )

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
        self.segments[ifo] = self.get_segments(strain, ifo)
        matched_filter = self.get_matched_filter(self.segments[ifo])
        sigmasq = self.template.sigmasq(self.segments[ifo][self.segment_index].psd)
        sigma = np.sqrt(sigmasq)
        self.sigma.append(sigma)
        self.output_data.__getattribute__(ifo).sigma.append(sigma)
        snr_ts, norm, corr, ind, snrv = matched_filter.matched_filter_and_cluster(
            self.segment_index, sigmasq, window=0
        )
        self.snr_dict[ifo] = (
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

    def get_timing_iterator(self):
        df = get_bilby_posteriors(self.conf.data.posteriors_file)[1000:1100]
        logging.info("Generating timing iterator")
        self.timing_iterator = get_timing_iterator(
            df["geocent_time"].to_numpy(),
            np.arange(
                self.conf.injection.time_gps_future_seconds - 0.1,
                self.conf.injection.time_gps_future_seconds + 0.1,
                self.snr_dict["H1"]._delta_t,
            ),
            df["ra"].to_numpy(),
            df["dec"].to_numpy(),
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

    def calculate_antenna_pattern(
        self,
        ra,
        dec,
        original_trigger_time_seconds,
        lensed_trigger_time_seconds,
    ):
        self.unlensed_antenna_pattern = calculate_antenna_pattern(
            self.unlensed_detectors,
            ra,
            dec,
            original_trigger_time_seconds,
        )
        self.lensed_antenna_pattern = calculate_antenna_pattern(
            self.lensed_detectors,
            ra,
            dec,
            lensed_trigger_time_seconds,
        )

    def get_time_delay_at_zerolag_seconds(
        self,
        original_trigger_time_seconds,
        lensed_trigger_time_seconds,
        ra,
        dec,
    ):
        self.unlensed_time_delay_zerolag_seconds = get_time_delay_at_zerolag_seconds(
            original_trigger_time_seconds,
            ra,
            dec,
            self.unlensed_detectors,
        )
        self.lensed_time_delay_zerolag_seconds = get_time_delay_at_zerolag_seconds(
            lensed_trigger_time_seconds,
            ra,
            dec,
            self.lensed_detectors,
        )

    def get_time_delay_indices(
        self,
        SAMPLE_RATE,
    ):
        self.unlensed_time_delay_idx = get_time_delay_indices(
            SAMPLE_RATE,
            self.unlensed_time_delay_zerolag_seconds,
            self.time_slides_seconds,
        )
        self.lensed_time_delay_idx = get_time_delay_indices(
            SAMPLE_RATE,
            self.lensed_time_delay_zerolag_seconds,
            self.time_slides_seconds,
        )

    def get_snr_at_trigger_original(
        self,
        get_snr,
        sky_position_index,
        original_trigger_time_seconds,
        time_slide_index,
    ):
        self.snr_at_trigger_original = [
            get_snr(
                time_delay_zerolag_seconds=self.unlensed_time_delay_zerolag_seconds[
                    sky_position_index
                ][ifo],
                timeseries=self.snr_dict[ifo],
                trigger_time_seconds=original_trigger_time_seconds,
                gps_start_seconds=self.conf.injection.gps_start_seconds[ifo],
                sample_rate=self.conf.injection.sample_rate,
                time_delay_idx=self.unlensed_time_delay_idx[time_slide_index][
                    sky_position_index
                ][ifo],
                cumulative_index=self.segments[ifo][
                    self.segment_index
                ].cumulative_index,
                time_slides_seconds=self.time_slides_seconds[ifo][time_slide_index],
            )
            for ifo in self.unlensed_detectors
        ]

    def get_snr_at_trigger_lensed(
        self,
        get_snr,
        sky_position_index,
        lensed_trigger_time_seconds,
        time_slide_index,
    ):
        self.snr_at_trigger_lensed = [
            get_snr(
                time_delay_zerolag_seconds=self.lensed_time_delay_zerolag_seconds[
                    sky_position_index
                ][ifo],
                timeseries=self.snr_dict[ifo],
                trigger_time_seconds=lensed_trigger_time_seconds,
                gps_start_seconds=self.conf.injection.gps_start_seconds[ifo],
                sample_rate=self.conf.injection.sample_rate,
                time_delay_idx=self.lensed_time_delay_idx[time_slide_index][
                    sky_position_index
                ][ifo],
                cumulative_index=self.segments[ifo][
                    self.segment_index
                ].cumulative_index,
                time_slides_seconds=self.time_slides_seconds[ifo][time_slide_index],
            )
            for ifo in self.lensed_detectors
        ]
