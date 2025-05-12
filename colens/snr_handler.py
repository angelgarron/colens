import logging

import numpy as np
from pycbc.detector import Detector

from colens.background import (
    get_time_delay_at_zerolag_seconds,
    get_time_delay_indices,
    get_time_slides_seconds,
)


class SNRHandler:
    def __init__(
        self,
        conf,
        get_snr,
        sigma,
        snrs,
        segments,
        instruments,
        lensed,
    ):
        self.conf = conf
        self.get_snr = get_snr
        self.instruments = instruments
        self.lensed = lensed
        # TODO loop over segments (or maybe we just create a big segment)
        self.segment_index = 0
        self.sky_position_index = 0
        self.sigma = sigma
        self.snrs = snrs
        self.segments = segments
        self.detectors = dict()
        for ifo, ifo_real_name in zip(self.instruments, ["H1", "L1"]):
            self.detectors[ifo] = Detector(ifo_real_name)
        # self.num_slides = slide_limiter(
        #     conf.injection.segment_length_seconds,
        #     conf.injection.slide_shift_seconds,
        #     len(conf.injection.lensed_instruments),
        # )
        self.num_slides = 1
        self.time_slides_seconds = get_time_slides_seconds(
            self.num_slides,
            self.conf.injection.slide_shift_seconds,
            conf.injection.unlensed_instruments,
            conf.injection.lensed_instruments,
        )
        self.time_slide_index = 0

    def _get_snr_at_trigger(
        self,
        get_snr,
        sky_position_index,
        trigger_time_seconds,
        time_slide_index,
        detectors,
        time_delay_zerolag_seconds,
        time_delay_idx,
        snrs,
        segments,
    ):
        return [
            get_snr(
                time_delay_zerolag_seconds=time_delay_zerolag_seconds[
                    sky_position_index
                ][ifo],
                timeseries=snrs[i],
                trigger_time_seconds=trigger_time_seconds,
                gps_start_seconds=self.conf.injection.gps_start_seconds[ifo],
                sample_rate=self.conf.injection.sample_rate,
                time_delay_idx=time_delay_idx[time_slide_index][sky_position_index][
                    ifo
                ],
                cumulative_index=segments[i][self.segment_index].cumulative_index,
                time_slides_seconds=self.time_slides_seconds[ifo][time_slide_index],
            )
            for i, ifo in enumerate(detectors)
        ]

    def first_function(self, time_gps_seconds):
        self.lensed_trigger_time_seconds = time_gps_seconds

    def second_function(self, time_gps_seconds, ra, dec):
        self.ra = ra
        self.dec = dec
        if self.lensed:
            self.trigger_time_seconds = self.lensed_trigger_time_seconds
        else:
            self.trigger_time_seconds = time_gps_seconds
        self.time_delay_zerolag_seconds = get_time_delay_at_zerolag_seconds(
            self.trigger_time_seconds,
            self.ra,
            self.dec,
            self.detectors,
        )
        self.time_delay_idx = get_time_delay_indices(
            self.conf.injection.sample_rate,
            self.time_delay_zerolag_seconds,
            self.time_slides_seconds,
        )
        self.snr_at_trigger = self._get_snr_at_trigger(
            self.get_snr,
            self.sky_position_index,
            self.trigger_time_seconds,
            self.time_slide_index,
            self.detectors,
            self.time_delay_zerolag_seconds,
            self.time_delay_idx,
            self.snrs,
            self.segments,
        )
        self.fp = []
        self.fc = []
        for detector in self.detectors.values():
            fp, fc = detector.antenna_pattern(
                self.ra,
                self.dec,
                polarization=0,
                t_gps=self.trigger_time_seconds,
            )
            self.fp.append(fp)
            self.fc.append(fc)
