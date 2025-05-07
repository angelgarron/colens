import logging
from itertools import groupby
from operator import itemgetter

import numpy as np
from pycbc.detector import Detector

from colens.background import (
    get_time_delay_at_zerolag_seconds,
    get_time_delay_indices,
    get_time_slides_seconds,
)
from colens.io import get_bilby_posteriors
from colens.timing import get_timing_iterator


class SNRHandler:
    def __init__(
        self,
        conf,
        get_snr,
        sigma,
        snrs_lensed,
        snrs_original,
        segments_lensed,
        segments_original,
    ):
        self.conf = conf
        self.get_snr = get_snr
        # TODO loop over segments (or maybe we just create a big segment)
        self.segment_index = 0
        self.sky_position_index = 0
        self.sigma = sigma
        self.snrs_lensed = snrs_lensed
        self.snrs_original = snrs_original
        self.segments_lensed = segments_lensed
        self.segments_original = segments_original
        self.unlensed_detectors = dict()
        self.lensed_detectors = dict()
        for ifo in conf.injection.unlensed_instruments:
            self.unlensed_detectors[ifo] = Detector(ifo)
        for ifo in conf.injection.lensed_instruments:
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
        self.time_slide_index = 0

    def get_timing_iterator(self):
        df = get_bilby_posteriors(self.conf.data.posteriors_file)[1000:1100]
        self.time_gps_past_seconds_array = df["geocent_time"].to_numpy()
        self.time_gps_future_seconds_array = np.arange(
            self.conf.injection.time_gps_future_seconds - 0.1,
            self.conf.injection.time_gps_future_seconds + 0.1,
            1 / self.conf.injection.sample_rate,
        )
        self.ra_array = df["ra"].to_numpy()
        self.dec_array = df["dec"].to_numpy()
        logging.info("Generating timing iterator")
        self.timing_iterator = _create_iterator(
            get_timing_iterator(
                self.time_gps_past_seconds_array,
                self.time_gps_future_seconds_array,
                self.ra_array,
                self.dec_array,
            ),
            [self.first_function, self.second_function],
        )

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

    def first_function(self, arg):
        self.lensed_trigger_time_seconds = self.time_gps_future_seconds_array[arg]

    def second_function(self, arg):
        self.ra = self.ra_array[arg]
        self.dec = self.dec_array[arg]
        self.original_trigger_time_seconds = self.time_gps_past_seconds_array[arg]
        self.unlensed_time_delay_zerolag_seconds = get_time_delay_at_zerolag_seconds(
            self.original_trigger_time_seconds,
            self.ra,
            self.dec,
            self.unlensed_detectors,
        )
        self.lensed_time_delay_zerolag_seconds = get_time_delay_at_zerolag_seconds(
            self.lensed_trigger_time_seconds,
            self.ra,
            self.dec,
            self.lensed_detectors,
        )
        self.unlensed_time_delay_idx = get_time_delay_indices(
            self.conf.injection.sample_rate,
            self.unlensed_time_delay_zerolag_seconds,
            self.time_slides_seconds,
        )
        self.lensed_time_delay_idx = get_time_delay_indices(
            self.conf.injection.sample_rate,
            self.lensed_time_delay_zerolag_seconds,
            self.time_slides_seconds,
        )
        self.snr_at_trigger_original = self._get_snr_at_trigger(
            self.get_snr,
            self.sky_position_index,
            self.original_trigger_time_seconds,
            self.time_slide_index,
            self.unlensed_detectors,
            self.unlensed_time_delay_zerolag_seconds,
            self.unlensed_time_delay_idx,
            self.snrs_original,
            self.segments_original,
        )
        self.snr_at_trigger_lensed = self._get_snr_at_trigger(
            self.get_snr,
            self.sky_position_index,
            self.lensed_trigger_time_seconds,
            self.time_slide_index,
            self.lensed_detectors,
            self.lensed_time_delay_zerolag_seconds,
            self.lensed_time_delay_idx,
            self.snrs_lensed,
            self.segments_lensed,
        )
        self.snr_at_trigger = self.snr_at_trigger_original + self.snr_at_trigger_lensed
        self.fp = []
        self.fc = []
        for detector in self.unlensed_detectors.values():
            fp, fc = detector.antenna_pattern(
                self.ra,
                self.dec,
                polarization=0,
                t_gps=self.original_trigger_time_seconds,
            )
            self.fp.append(fp)
            self.fc.append(fc)
        for detector in self.lensed_detectors.values():
            fp, fc = detector.antenna_pattern(
                self.ra,
                self.dec,
                polarization=0,
                t_gps=self.lensed_trigger_time_seconds,
            )
            self.fp.append(fp)
            self.fc.append(fc)


def _create_iterator(generator, functions):
    def inner(gen, func_idx):
        for i, group in groupby(gen, key=itemgetter(func_idx)):
            functions[func_idx](i)
            if func_idx < len(functions) - 2:  # TODO find a better way to do this
                yield from inner(group, func_idx + 1)
            else:  # pause iteration on the innermost for loop
                for i_, group_ in groupby(group, key=itemgetter(func_idx + 1)):
                    functions[func_idx + 1](i_)
                    yield

    iterator = inner(generator, 0)
    return iterator
