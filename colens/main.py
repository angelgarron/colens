import logging

import numpy as np
from pycbc import init_logging

from colens.configuration import read_configuration_from
from colens.data_loader import DataLoader
from colens.fstatistic import get_two_f
from colens.interpolate import get_snr, get_snr_interpolated
from colens.io import Output
from colens.runner import Runner


def main():
    init_logging(True)
    conf = read_configuration_from("config.yaml")
    output_data = Output()

    data_loader = DataLoader(conf, output_data)

    logging.info("Starting the filtering...")
    runner = Runner(get_two_f, get_snr, output_data, data_loader)
    runner.run()
    logging.info("Filtering completed")
    logging.info(f"Saving results to {conf.output.output_file_name}")
    output_data.write_to_json(conf.output.output_file_name)


if __name__ == "__main__":
    main()
