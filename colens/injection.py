"""Functions used to inject simulated signals."""

import dataclasses

import bilby
from pycbc.detector import Detector
from pycbc.waveform import get_fd_waveform, get_td_waveform


def _get_strains_from(ifos, ifo_names):
    strains = []
    for i in range(len(ifo_names)):
        strain_tmp = ifos[i].strain_data.to_pycbc_timeseries()
        strains.append(strain_tmp)
    return strains


def get_ifos_without_noise(ifo_names, sampling_frequency, duration, start_time):
    ifos = bilby.gw.detector.InterferometerList(ifo_names)
    ifos.set_strain_data_from_zero_noise(
        sampling_frequency=sampling_frequency,
        duration=duration,
        start_time=start_time,
    )
    return ifos


def get_ifos_with_simulated_noise(ifo_names, sampling_frequency, duration, start_time):
    ifos = bilby.gw.detector.InterferometerList(ifo_names)
    ifos.set_strain_data_from_power_spectral_densities(
        sampling_frequency=sampling_frequency,
        duration=duration,
        start_time=start_time,
    )
    return ifos


def inject_real_noise(strains, ifo_names, noise: dict):
    for i in range(len(ifo_names)):
        strains[i] = strains[i].inject(noise[ifo_names[i]])


def get_strain_list_from_bilby_simulation(
    injection_parameters,
    ifo_names,
    start_time,
    end_time,
    low_frequency_cutoff,
    reference_frequency,
    sampling_frequency,
    seed,
    approximant,
    get_ifos_function,
):
    # Set up a random seed for result reproducibility.  This is optional!
    bilby.core.utils.random.seed(seed)
    duration = end_time - start_time

    ifos = get_ifos_function(ifo_names, sampling_frequency, duration, start_time)

    waveform_arguments = dict(
        waveform_approximant=approximant,
        reference_frequency=reference_frequency,
        minimum_frequency=low_frequency_cutoff,
    )

    waveform_generator = bilby.gw.WaveformGenerator(
        duration=duration,
        sampling_frequency=sampling_frequency,
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        # frequency_domain_source_model=strongly_lensed_BBH_waveform,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        waveform_arguments=waveform_arguments,
    )

    ifos.inject_signal(
        waveform_generator=waveform_generator,
        parameters=dataclasses.asdict(injection_parameters),
    )
    strains = _get_strains_from(ifos, ifo_names)

    return strains


def get_strain_list_from_pycbc_simulation(
    injection_parameters,
    ifo_names,
    start_time,
    end_time,
    low_frequency_cutoff,
    reference_frequency,
    sampling_frequency,
    seed,
    approximant,
    get_ifos_function,
):
    duration = end_time - start_time

    # Set up a random seed for result reproducibility.  This is optional!
    bilby.core.utils.random.seed(seed)

    ifos = get_ifos_function(ifo_names, sampling_frequency, duration, start_time)

    strains = _get_strains_from(ifos, ifo_names)
    for i in range(len(ifo_names)):
        signal = _get_signal_from_pycbc(
            dataclasses.asdict(injection_parameters),
            low_frequency_cutoff,
            reference_frequency,
            sampling_frequency,
            approximant,
            ifo_names[i],
        )
        strains[i] = strains[i].inject(signal)

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

    signal = Detector(ifo).project_wave(
        hp=hp,
        hc=hc,
        ra=injection_parameters["ra"],
        dec=injection_parameters["dec"],
        polarization=injection_parameters["psi"],
        method="constant",
        reference_time=injection_parameters["geocent_time"],
    )
    # we don't want the zero time to be at the end of the array
    signal = signal.cyclic_time_shift(-10)
    signal.start_time += injection_parameters["geocent_time"]
    return signal
