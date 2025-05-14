import logging

import numpy as np
from pycbc.detector import Detector


class SNRHandler:
    def __init__(
        self,
        conf,
        get_snr,
        data_loader,
        instruments,
        time_slides_seconds,
        gps_start_seconds,
    ):
        self.conf = conf
        self.get_snr = get_snr
        self.instruments = instruments
        self.time_slides_seconds = time_slides_seconds
        self.gps_start_seconds = gps_start_seconds
        self.sky_position_index = 0
        self.data_loader = data_loader
        self.detectors = dict()
        for ifo in self.instruments:
            self.detectors[ifo] = Detector(ifo)
        self.time_slide_index = 0

    def segment_setup(self):
        self.data_loader.single_segment_setup(self.segment_index)
        self.sigma = self.data_loader.sigma
        self.snrs = self.data_loader.snrs

    def _set_snr_at_trigger(self):
        self.snr_at_trigger = [
            self.get_snr(
                time_delay_zerolag_seconds=self.time_delay_zerolag_seconds[i],
                timeseries=self.snrs[i],
                trigger_time_seconds=self.trigger_time_seconds,
                gps_start_seconds=self.gps_start_seconds,
                sample_rate=self.conf.injection.sample_rate,
                time_delay_idx=self.time_delay_idx[self.time_slide_index][i],
                cumulative_index=self.data_loader.matched_filters[i]
                .segments[self.segment_index]
                .cumulative_index,
                time_slides_seconds=self.time_slides_seconds[i][self.time_slide_index],
            )
            for i in range(len(self.detectors))
        ]

    def _set_time_delay_at_zerolag_seconds(self, ra, dec):
        """Compute the difference of arrival time between the earth center and each one of the `instruments` of a signal
        coming from each point in `sky_grid`, .i.e. (t_{instrument} - t_{center}).
        """
        self.time_delay_zerolag_seconds = [
            detector.time_delay_from_earth_center(
                ra,
                dec,
                self.trigger_time_seconds,
            )
            for detector in self.detectors.values()
        ]

    def _set_time_delay_indices(self):
        slide_ids = np.arange(len(self.time_slides_seconds[0]))
        self.time_delay_idx = [
            [
                round(
                    (
                        self.time_delay_zerolag_seconds[i]
                        + self.time_slides_seconds[i][slide]
                    )
                    * self.conf.injection.sample_rate
                )
                for i in range(len(self.time_delay_zerolag_seconds))
            ]
            for slide in slide_ids
        ]

    def _set_antenna_patterns(self, ra, dec):
        self.fp = []
        self.fc = []
        for detector in self.detectors.values():
            fp, fc = detector.antenna_pattern(
                ra,
                dec,
                polarization=0,
                t_gps=self.trigger_time_seconds,
            )
            self.fp.append(fp)
            self.fc.append(fc)

    def set_trigger_time(self, time_gps_seconds):
        self.trigger_time_seconds = time_gps_seconds

    def second_function(self, ra, dec):
        self._set_time_delay_at_zerolag_seconds(ra, dec)
        self._set_time_delay_indices()
        self._set_antenna_patterns(ra, dec)

    def third_function(self, time_slide_index):
        self.time_slide_index = time_slide_index
        self._set_snr_at_trigger()

    def fourth_function(self, segment_index):
        self.segment_index = segment_index
        self.segment_setup()
