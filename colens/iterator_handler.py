import logging
from itertools import groupby
from operator import itemgetter

import numpy as np

from colens.io import get_bilby_posteriors
from colens.timing import get_timing_iterator


class IteratorHandler:
    def __init__(
        self,
        conf,
        snr_handler,
        snr_handler_lensed,
    ):
        self.conf = conf
        self.snr_handler = snr_handler
        self.snr_handler_lensed = snr_handler_lensed
        self.get_timing_iterator()

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

    def first_function(self, arg):
        self.snr_handler_lensed.set_trigger_time(
            self.time_gps_future_seconds_array[arg]
        )

    def second_function(self, arg):
        self.snr_handler.set_trigger_time(self.time_gps_past_seconds_array[arg])
        self.snr_handler.second_function(
            self.ra_array[arg],
            self.dec_array[arg],
        )
        self.snr_handler_lensed.second_function(
            self.ra_array[arg],
            self.dec_array[arg],
        )


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
