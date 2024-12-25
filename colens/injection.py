"""Functions used to inject simulated signals."""

import bilby
import numpy as np
from pycbc.detector import Detector
from pycbc.types.timeseries import TimeSeries
from pycbc.waveform import get_fd_waveform, get_td_waveform


def get_strain_list_from_simulation(
    injection_parameters,
    ifo_names,
    start_time,
    end_time,
    low_frequency_cutoff,
    reference_frequency,
    sampling_frequency,
    seed,
    approximant,
    inject_from_pycbc=False,
    is_zero_noise=False,
):
    duration = end_time - start_time

    # Set up a random seed for result reproducibility.  This is optional!
    bilby.core.utils.random.seed(seed)

    waveform_arguments = dict(
        waveform_approximant=approximant,
        reference_frequency=reference_frequency,
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
    if is_zero_noise:
        ifos.set_strain_data_from_zero_noise(
            sampling_frequency=sampling_frequency,
            duration=duration,
            start_time=start_time,
        )
    else:
        ifos.set_strain_data_from_power_spectral_densities(
            sampling_frequency=sampling_frequency,
            duration=duration,
            start_time=start_time,
        )

    if not inject_from_pycbc:
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
        if inject_from_pycbc:
            signal = _get_signal_from_pycbc(
                injection_parameters,
                low_frequency_cutoff,
                reference_frequency,
                sampling_frequency,
                approximant,
                ifo_names[i],
            )
            strain_tmp = strain_tmp.inject(signal)
        strains.append(strain_tmp)

    return strains


def _get_signal_from_pycbc(
    injection_parameters,
    low_frequency_cutoff,
    reference_frequency,
    sampling_frequency,
    approximant,
    ifo,
):
    hp, hc = get_fd_waveform(
        approximant=approximant,
        mass1=injection_parameters["mass_1"],
        mass2=injection_parameters["mass_2"],
        spin1z=injection_parameters["a_1"],
        spin2z=injection_parameters["a_2"],
        inclination=injection_parameters["theta_jn"],
        coa_phase=injection_parameters["phase"],
        distance=injection_parameters["luminosity_distance"],
        f_ref=reference_frequency,
        delta_f=0.0625,
        f_lower=low_frequency_cutoff,
        f_final=2048.0,
    )

    hp = hp.to_timeseries(delta_t=1.0 / sampling_frequency)
    hc = hc.to_timeseries(delta_t=1.0 / sampling_frequency)
    det = Detector(ifo)

    declination = injection_parameters["dec"]
    right_ascension = injection_parameters["ra"]
    polarization = injection_parameters["psi"]

    signal = det.project_wave(
        hp,
        hc,
        right_ascension,
        declination,
        polarization,
        method="constant",
        reference_time=injection_parameters["geocent_time"],
    )
    # we don't want the zero time to be at the end of the array
    signal = signal.cyclic_time_shift(-10)
    signal.start_time += injection_parameters["geocent_time"]
    return signal
