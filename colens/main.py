import logging

import numpy as np
from pycbc import init_logging
from pycbc.types import complex64, float32, zeros

from colens.background import get_time_slides_seconds
from colens.bank import MyFilterBank
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

    delta_f = (
        1.0 / conf.injection.segment_length_seconds
    )  # frequency step of the fourier transform of each segment
    segment_length = int(
        conf.injection.segment_length_seconds * conf.injection.sample_rate
    )  # number of samples of each segment
    frequency_length = int(
        segment_length // 2 + 1
    )  # number of samples of the fourier transform of each segment
    template_mem = zeros(segment_length, dtype=complex64)
    template_parameters = {
        "mass1": np.array([79.45]),
        "mass2": np.array([48.50]),
        "spin1z": np.array([0.60]),
        "spin2z": np.array([0.05]),
        "f_final": np.array([2048.0]),
        "f_ref": np.array([conf.injection.reference_frequency]),
    }
    template = MyFilterBank(
        filter_length=frequency_length,
        delta_f=delta_f,
        dtype=complex64,
        template_parameters=template_parameters,
        low_frequency_cutoff=conf.injection.low_frequency_cutoff,
        phase_order=conf.injection.order,
        approximant=conf.injection.approximant,
        out=template_mem,
    )[0]

    data_loader = DataLoader(
        conf,
        conf.injection.unlensed_instruments,
        output_data.original_output,
        conf.injection.time_gps_past_seconds,
        conf.injection.gps_start_seconds["past"],
        conf.injection.gps_end_seconds["past"],
        conf.injection.trig_start_time_seconds["past"],
        conf.injection.trig_end_time_seconds["past"],
        delta_f,
        segment_length,
        frequency_length,
        template_mem,
        template,
    )
    data_loader_lensed = DataLoader(
        conf,
        conf.injection.lensed_instruments,
        output_data.lensed_output,
        conf.injection.time_gps_future_seconds,
        conf.injection.gps_start_seconds["future"],
        conf.injection.gps_end_seconds["future"],
        conf.injection.trig_start_time_seconds["future"],
        conf.injection.trig_end_time_seconds["future"],
        delta_f,
        segment_length,
        frequency_length,
        template_mem,
        template,
    )
    # num_slides = slide_limiter(
    #     conf.injection.segment_length_seconds,
    #     conf.injection.slide_shift_seconds,
    #     len(conf.injection.lensed_instruments),
    # )
    num_slides = 1
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
        data_loader,
        conf.injection.unlensed_instruments,
        time_slides_seconds_unlensed,
        conf.injection.gps_start_seconds["past"],
    )
    snr_handler_lensed = SNRHandler(
        conf,
        get_snr,
        data_loader_lensed,
        conf.injection.lensed_instruments,
        time_slides_seconds_lensed,
        conf.injection.gps_start_seconds["future"],
    )
    iterator_handler = IteratorHandler(
        conf, snr_handler, snr_handler_lensed, num_slides
    )

    logging.info("Starting the filtering...")
    runner = Runner(
        conf,
        get_two_f,
        output_data,
        snr_handler,
        snr_handler_lensed,
        iterator_handler,
    )
    runner.run()
    logging.info("Filtering completed")
    logging.info(f"Saving results to {conf.output.output_file_name}")
    output_data.write_to_json(conf.output.output_file_name)


if __name__ == "__main__":
    main()
