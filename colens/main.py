import logging

import numpy as np
from pycbc import init_logging

from colens.background import get_time_slides_seconds
from colens.configuration import read_configuration_from
from colens.data_loader import DataLoader
from colens.fstatistic import get_two_f
from colens.interpolate import get_snr, get_snr_interpolated
from colens.io import Output, PerDetectorOutput
from colens.iterator_handler import IteratorHandler
from colens.runner import Runner
from colens.snr_handler import SNRHandler


def main():
    init_logging(True)
    conf = read_configuration_from("config.yaml")
    output_data = Output()
    for ifo in conf.injection.unlensed_instruments:
        output_data.original_output.append(PerDetectorOutput(ifo))
    for ifo in conf.injection.lensed_instruments:
        output_data.lensed_output.append(PerDetectorOutput(ifo))

    data_loader = DataLoader(
        conf,
        output_data,
        conf.injection.unlensed_instruments,
        output_data.original_output,
        conf.injection.time_gps_past_seconds,
        conf.injection.gps_start_seconds["past"],
        conf.injection.gps_end_seconds["past"],
        conf.injection.trig_start_time_seconds["past"],
        conf.injection.trig_end_time_seconds["past"],
    )
    data_loader_lensed = DataLoader(
        conf,
        output_data,
        conf.injection.lensed_instruments,
        output_data.lensed_output,
        conf.injection.time_gps_future_seconds,
        conf.injection.gps_start_seconds["future"],
        conf.injection.gps_end_seconds["future"],
        conf.injection.trig_start_time_seconds["future"],
        conf.injection.trig_end_time_seconds["future"],
    )
    # num_slides = slide_limiter(
    #     conf.injection.segment_length_seconds,
    #     conf.injection.slide_shift_seconds,
    #     len(conf.injection.lensed_instruments),
    # )
    num_slides = 2
    time_slides_seconds_lensed = get_time_slides_seconds(
        num_slides,
        conf.injection.slide_shift_seconds,
        conf.injection.lensed_instruments,
    )
    time_slides_seconds_unlensed = [
        np.zeros(num_slides) for _ in conf.injection.unlensed_instruments
    ]
    snr_handler = SNRHandler(
        conf,
        get_snr,
        data_loader.sigma,
        data_loader.snrs,
        data_loader.segments,
        conf.injection.unlensed_instruments,
        time_slides_seconds_unlensed,
        conf.injection.gps_start_seconds["past"],
    )
    snr_handler_lensed = SNRHandler(
        conf,
        get_snr,
        data_loader_lensed.sigma,
        data_loader_lensed.snrs,
        data_loader_lensed.segments,
        conf.injection.lensed_instruments,
        time_slides_seconds_lensed,
        conf.injection.gps_start_seconds["future"],
    )
    iterator_handler = IteratorHandler(
        conf, snr_handler, snr_handler_lensed, num_slides, output_data
    )

    logging.info("Starting the filtering...")
    runner = Runner(
        get_two_f,
        output_data,
        snr_handler,
        snr_handler_lensed,
        iterator_handler.timing_iterator,
    )
    runner.run()
    logging.info("Filtering completed")
    logging.info(f"Saving results to {conf.output.output_file_name}")
    output_data.write_to_json(conf.output.output_file_name)


if __name__ == "__main__":
    main()
