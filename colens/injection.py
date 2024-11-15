"""Functions used to inject simulated signals."""

import bilby
from pycbc.types.timeseries import TimeSeries


def get_strain_list_from_simulation(
    injection_parameters, ifo_names, start_time, end_time, low_frequency_cutoff, seed
):
    sampling_frequency = 4096.0
    duration = end_time - start_time

    # Set up a random seed for result reproducibility.  This is optional!
    bilby.core.utils.random.seed(seed)

    waveform_arguments = dict(
        waveform_approximant="IMRPhenomD",
        reference_frequency=50.0,
        minimum_frequency=low_frequency_cutoff,
    )

    waveform_generator = bilby.gw.WaveformGenerator(
        duration=duration,
        sampling_frequency=sampling_frequency,
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        waveform_arguments=waveform_arguments,
    )

    ifos = bilby.gw.detector.InterferometerList(ifo_names)
    ifos.set_strain_data_from_power_spectral_densities(
        sampling_frequency=sampling_frequency,
        duration=duration,
        start_time=start_time,
    )

    ifos.inject_signal(
        waveform_generator=waveform_generator, parameters=injection_parameters
    )
    strains = []
    for i in range(len(ifo_names)):
        strain_tmp = TimeSeries(
            initial_array=ifos[i].time_domain_strain,
            delta_t=ifos[i].time_array[1] - ifos[i].time_array[0],
            epoch=start_time,
        )
        strains.append(strain_tmp)
    optimal_snrs = [ifos.meta_data[ifo_name]["optimal_SNR"] for ifo_name in ifo_names]

    matched_filter_snrs = [
        ifos.meta_data[ifo_name]["matched_filter_SNR"] for ifo_name in ifo_names
    ]

    return optimal_snrs, matched_filter_snrs, strains
