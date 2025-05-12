import logging

from pycbc import init_logging

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
        output_data.original_output.append(PerDetectorOutput())
    for ifo in conf.injection.lensed_instruments:
        output_data.lensed_output.append(PerDetectorOutput())

    data_loader = DataLoader(
        conf,
        output_data,
        conf.injection.unlensed_instruments,
        output_data.original_output,
        conf.injection.time_gps_past_seconds,
    )
    data_loader_lensed = DataLoader(
        conf,
        output_data,
        conf.injection.lensed_instruments,
        output_data.lensed_output,
        conf.injection.time_gps_future_seconds,
    )
    snr_handler = SNRHandler(
        conf,
        get_snr,
        data_loader.sigma,
        data_loader.snrs,
        data_loader.segments,
        conf.injection.unlensed_instruments,
        False,
    )
    snr_handler_lensed = SNRHandler(
        conf,
        get_snr,
        data_loader_lensed.sigma,
        data_loader_lensed.snrs,
        data_loader_lensed.segments,
        conf.injection.lensed_instruments,
        True,
    )
    iterator_handler = IteratorHandler(conf, snr_handler, snr_handler_lensed)

    logging.info("Starting the filtering...")
    runner = Runner(
        get_two_f, output_data, snr_handler, snr_handler_lensed, iterator_handler
    )
    runner.run()
    logging.info("Filtering completed")
    logging.info(f"Saving results to {conf.output.output_file_name}")
    output_data.write_to_json(conf.output.output_file_name)


if __name__ == "__main__":
    main()
