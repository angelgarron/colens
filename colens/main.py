import logging

from pycbc import init_logging

from colens.configuration import read_configuration_from
from colens.data_loader import DataLoader
from colens.fstatistic import get_two_f
from colens.interpolate import get_snr, get_snr_interpolated
from colens.io import Output
from colens.runner import Runner
from colens.snr_handler import SNRHandler


def main():
    init_logging(True)
    conf = read_configuration_from("config.yaml")
    output_data = Output()

    data_loader = DataLoader(conf, output_data)
    snr_handler = SNRHandler(
        conf,
        get_snr,
        data_loader.sigma,
        data_loader.snrs_lensed,
        data_loader.snrs_original,
        data_loader.segments_lensed,
        data_loader.segments_original,
    )

    logging.info("Starting the filtering...")
    runner = Runner(get_two_f, output_data, snr_handler)
    runner.run()
    logging.info("Filtering completed")
    logging.info(f"Saving results to {conf.output.output_file_name}")
    output_data.write_to_json(conf.output.output_file_name)


if __name__ == "__main__":
    main()
