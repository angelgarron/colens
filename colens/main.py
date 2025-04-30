import logging

import numpy as np
from pycbc import init_logging

from colens.background import slide_limiter
from colens.brute_force_filter import brute_force_filter_template
from colens.configuration import read_configuration_from
from colens.data_loader import DataLoader
from colens.detector import MyDetector
from colens.fstatistic import get_two_f
from colens.interpolate import get_snr, get_snr_interpolated
from colens.io import Output


def main():
    init_logging(True)
    conf = read_configuration_from("config.yaml")
    output_data = Output()

    lensed_detectors = {
        ifo: MyDetector(ifo) for ifo in conf.injection.lensed_instruments
    }
    unlensed_detectors = {
        ifo: MyDetector(ifo) for ifo in conf.injection.unlensed_instruments
    }

    num_slides = slide_limiter(
        conf.injection.segment_length_seconds,
        conf.injection.slide_shift_seconds,
        len(conf.injection.lensed_instruments),
    )
    num_slides = 1

    data_loader = DataLoader(conf, output_data)

    logging.info("Starting the filtering...")
    brute_force_filter_template(
        lensed_detectors,
        unlensed_detectors,
        num_slides,
        conf.injection.slide_shift_seconds,
        conf.injection.sample_rate,
        conf.injection.gps_start_seconds,
        get_two_f,
        output_data,
        get_snr,
        data_loader,
    )

    logging.info("Filtering completed")
    logging.info(f"Saving results to {conf.output.output_file_name}")
    output_data.write_to_json(conf.output.output_file_name)


if __name__ == "__main__":
    main()
