import logging
from functools import partial

import numpy as np
import pandas as pd
from pycbc import DYN_RANGE_FAC, init_logging, vetoes, waveform
from pycbc.filter import MatchedFilterControl
from pycbc.psd import associate_psds_to_segments
from pycbc.strain import StrainSegments
from pycbc.types import complex64, float32, zeros

from colens import configuration
from colens.background import slide_limiter
from colens.bank import MyFilterBank
from colens.brute_force_filter import brute_force_filter_template
from colens.configuration import read_configuration_from
from colens.detector import MyDetector
from colens.filter import filter_ifos, get_sigmasq
from colens.fstatistic import get_two_f
from colens.injection import (
    get_ifos_with_simulated_noise,
    get_strain_list_from_bilby_simulation,
)
from colens.interpolate import get_snr, get_snr_interpolated
from colens.io import (
    Output,
    PerDetectorOutput,
    get_bilby_posteriors,
    get_strain_dict_from_files,
)
from colens.strain import process_strain_dict
from colens.timing import get_timing_iterator


def create_template_bank(
    conf_injection: configuration.Injection, template_mem, frequency_length, delta_f
) -> MyFilterBank:
    template_parameters = {
        "mass1": np.array([79.45]),
        "mass2": np.array([48.50]),
        "spin1z": np.array([0.60]),
        "spin2z": np.array([0.05]),
        "f_final": np.array([2048.0]),
        "f_ref": np.array([conf_injection.reference_frequency]),
    }
    return MyFilterBank(
        filter_length=frequency_length,
        delta_f=delta_f,
        dtype=complex64,
        template_parameters=template_parameters,
        low_frequency_cutoff=conf_injection.low_frequency_cutoff,
        phase_order=conf_injection.order,
        approximant=conf_injection.approximant,
        out=template_mem,
    )


def create_injections(
    injection_parameters: configuration.InjectionParameters,
    conf_injection: configuration.Injection,
):
    # The extra padding we are adding here is going to get removed after highpassing
    return_value = get_strain_list_from_bilby_simulation(
        injection_parameters.asdict(),
        ["H1", "L1"],
        start_time=conf_injection.gps_start_seconds["H1"] - conf_injection.pad_seconds,
        end_time=conf_injection.gps_end_seconds["H1"] + conf_injection.pad_seconds,
        low_frequency_cutoff=conf_injection.low_frequency_cutoff,
        reference_frequency=conf_injection.reference_frequency,
        sampling_frequency=conf_injection.sample_rate,
        seed=1,
        approximant=conf_injection.approximant,
        get_ifos_function=get_ifos_with_simulated_noise,
    )
    strain_dict = dict(zip(["H1", "L1"], return_value))
    # the lensed image
    injection_parameters.geocent_time = conf_injection.time_gps_future_seconds
    return_value = get_strain_list_from_bilby_simulation(
        injection_parameters.asdict(),
        ["H1", "L1"],
        start_time=conf_injection.gps_start_seconds["H1_lensed"]
        - conf_injection.pad_seconds,
        end_time=conf_injection.gps_end_seconds["H1_lensed"]
        + conf_injection.pad_seconds,
        low_frequency_cutoff=conf_injection.low_frequency_cutoff,
        reference_frequency=conf_injection.reference_frequency,
        sampling_frequency=conf_injection.sample_rate,
        seed=2,
        approximant=conf_injection.approximant,
        get_ifos_function=get_ifos_with_simulated_noise,
    )
    strain_dict.update(
        dict(
            zip(
                ["H1_lensed", "L1_lensed"],
                return_value,
            )
        )
    )
    return strain_dict


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

    logging.info("Injecting simulated signals on gaussian noise")
    strain_dict = create_injections(conf.injection_parameters, conf.injection)

    process_strain_dict(
        strain_dict,
        conf.psd.strain_high_pass_hertz,
        conf.injection.sample_rate,
        conf.injection.pad_seconds,
    )

    num_slides = slide_limiter(
        conf.injection.segment_length_seconds,
        conf.injection.slide_shift_seconds,
        len(conf.injection.lensed_instruments),
    )
    num_slides = 1

    # Create a dictionary of Python slice objects that indicate where the segments
    # start and end for each detector timeseries.
    segments = dict()
    for ifo in conf.injection.instruments:
        segments[ifo] = StrainSegments(
            strain_dict[ifo],
            segment_length=conf.injection.segment_length_seconds,
            segment_start_pad=conf.injection.segment_start_pad_seconds,
            segment_end_pad=conf.injection.segment_end_pad_seconds,
            trigger_start=conf.injection.trig_start_time_seconds[ifo],
            trigger_end=conf.injection.trig_end_time_seconds[ifo],
            filter_inj_only=False,
            allow_zero_padding=False,
        ).fourier_segments()

    delta_f = (
        1.0 / conf.injection.segment_length_seconds
    )  # frequency step of the fourier transform of each segment
    segment_length = int(
        conf.injection.segment_length_seconds * conf.injection.sample_rate
    )  # number of samples of each segment
    frequency_length = int(
        segment_length // 2 + 1
    )  # number of samples of the fourier transform of each segment

    logging.info("Associating PSDs to the fourier segments")
    for ifo in conf.injection.instruments:
        associate_psds_to_segments(
            conf.psd,
            segments[ifo],
            strain_dict[ifo],
            frequency_length,
            delta_f,
            conf.injection.low_frequency_cutoff,
            dyn_range_factor=DYN_RANGE_FAC,
            precision="single",
        )

    logging.info("Setting up MatchedFilterControl at each IFO")
    template_mem = zeros(segment_length, dtype=complex64)

    matched_filter = {
        ifo: MatchedFilterControl(
            conf.injection.low_frequency_cutoff,
            None,
            conf.injection.sngl_snr_threshold,
            segment_length,
            delta_f,
            complex64,
            segments[ifo],
            template_mem,
            use_cluster=False,
            downsample_factor=conf.injection.downsample_factor,
            upsample_threshold=conf.injection.upsample_threshold,
            upsample_method=conf.injection.upsample_method,
        )
        for ifo in conf.injection.instruments
    }

    logging.info("Initializing signal-based vetoes: power")
    power_chisq = vetoes.SingleDetPowerChisq(conf.chisq.chisq_bins)

    logging.info("Overwhitening frequency-domain data segments")
    for ifo in conf.injection.instruments:
        for seg in segments[ifo]:
            seg /= seg.psd

    logging.info("Read in template bank")
    bank = create_template_bank(conf.injection, template_mem, frequency_length, delta_f)

    logging.info("Full template bank size: %d", len(bank))

    logging.info("Starting the filtering...")
    for t_num, template in enumerate(bank):
        # TODO loop over segments (or maybe we just create a big segment)
        # get the single detector snrs
        segment_index = 0
        sigmasq = get_sigmasq(
            segments, template, conf.injection.instruments, segment_index
        )
        sigma = [np.sqrt(sigmasq[ifo]) for ifo in conf.injection.lensed_instruments]
        sigma += [np.sqrt(sigmasq[ifo]) for ifo in conf.injection.unlensed_instruments]

        snr_dict, norm_dict, corr_dict, idx, snr = filter_ifos(
            conf.injection.instruments, sigmasq, matched_filter, segment_index
        )

        for i, ifo in enumerate(conf.injection.lensed_instruments):
            output_data.__getattribute__(ifo).sigma.append(sigma[i])

        for i, ifo in enumerate(conf.injection.unlensed_instruments):
            output_data.__getattribute__(ifo).sigma.append(sigma[i + 2])

        df = get_bilby_posteriors(conf.data.posteriors_file)[1000:1100]
        logging.info("Generating timing iterator")
        timing_iterator = get_timing_iterator(
            df["geocent_time"].to_numpy(),
            np.arange(
                conf.injection.time_gps_future_seconds - 0.1,
                conf.injection.time_gps_future_seconds + 0.1,
                snr_dict["H1"]._delta_t,
            ),
            df["ra"].to_numpy(),
            df["dec"].to_numpy(),
        )
        logging.info("Filtering template %d/%d", t_num + 1, len(bank))
        brute_force_filter_template(
            lensed_detectors,
            unlensed_detectors,
            segments,
            num_slides,
            conf.injection.slide_shift_seconds,
            conf.injection.sample_rate,
            conf.injection.gps_start_seconds,
            get_two_f,
            output_data,
            get_snr,
            sigma,
            snr_dict,
            segment_index,
            timing_iterator,
        )

    logging.info("Filtering completed")
    logging.info(f"Saving results to {conf.output.output_file_name}")
    output_data.write_to_json(conf.output.output_file_name)


if __name__ == "__main__":
    main()
