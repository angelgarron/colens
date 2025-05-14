import logging
from itertools import groupby
from operator import itemgetter

import numpy as np

from colens.io import get_bilby_posteriors
from colens.timing import get_iteration_indexes


class IteratorHandler:
    def __init__(self, conf, snr_handler, snr_handler_lensed, num_slides, output_data):
        self.conf = conf
        self.snr_handler = snr_handler
        self.snr_handler_lensed = snr_handler_lensed
        self.num_slides = num_slides
        self.output_data = output_data
        self.get_timing_iterator()
        self.snr_handler.on_changed_segment_index(0)

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
        iteration_indexes = get_iteration_indexes(
            self.time_gps_future_seconds_array,
            self.time_gps_past_seconds_array,
            self.ra_array,
            self.dec_array,
        )

        def generator():
            for i in range(1):
                for args in zip(*iteration_indexes):
                    for j in range(self.num_slides):
                        yield (i,) + args + (j,)

        self.timing_iterator = _create_iterator(
            generator(),
            [
                self.on_changed_segment_index,
                self.on_changed_lensed_time,
                self.on_changed_posterior,
                self.on_changed_time_slide_index,
            ],
        )

    def on_changed_lensed_time(self, arg):
        self.snr_handler_lensed.set_trigger_time(
            self.time_gps_future_seconds_array[arg]
        )

    def on_changed_posterior(self, arg):
        self.ra = self.ra_array[arg]
        self.dec = self.dec_array[arg]

        self.snr_handler.set_trigger_time(self.time_gps_past_seconds_array[arg])

        self.snr_handler.on_changed_posterior(self.ra, self.dec)
        self.snr_handler_lensed.on_changed_posterior(self.ra, self.dec)

    def on_changed_time_slide_index(self, arg):
        self.snr_handler.on_changed_time_slide_index(arg)
        self.snr_handler_lensed.on_changed_time_slide_index(arg)

        self.write_output()

    def on_changed_segment_index(self, arg):
        self.snr_handler_lensed.on_changed_segment_index(arg)

    def write_output(self):
        self.output_data.ra.append(self.ra)
        self.output_data.dec.append(self.dec)


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
