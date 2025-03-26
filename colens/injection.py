"""Functions used to inject simulated signals."""

import bilby
import numpy as np
from pycbc.detector import Detector
from pycbc.types.timeseries import TimeSeries
from pycbc.waveform import get_fd_waveform, get_td_waveform

from colens.io import get_strain_dict_from_files

FRAME_FILES = {
    "H1": "/home/angel/Documents/pycbc_checks/H-H1_GWOSC_4KHZ_R1-1185387760-4096.gwf",
    "L1": "/home/angel/Documents/pycbc_checks/L-L1_GWOSC_4KHZ_R1-1185387760-4096.gwf",
    "H1_lensed": "/home/angel/Documents/pycbc_checks/H-H1_GWOSC_O2_4KHZ_R1-1185435648-4096.gwf",
    "L1_lensed": "/home/angel/Documents/pycbc_checks/L-L1_GWOSC_O2_4KHZ_R1-1185435648-4096.gwf",
}
CHANNELS = {
    "H1": "H1:GWOSC-4KHZ_R1_STRAIN",
    "L1": "L1:GWOSC-4KHZ_R1_STRAIN",
    "H1_lensed": "H1:GWOSC-4KHZ_R1_STRAIN",
    "L1_lensed": "L1:GWOSC-4KHZ_R1_STRAIN",
}
PAD_SECONDS = 8


# Function copied from https://github.com/ricokaloklo/hanabi
def strongly_lensed_BBH_waveform(
    frequency_array,
    mass_1,
    mass_2,
    luminosity_distance,
    a_1,
    tilt_1,
    phi_12,
    a_2,
    tilt_2,
    phi_jl,
    theta_jn,
    phase,
    morse_phase,
    **kwargs
):
    frequency_domain_source_model = bilby.gw.source.lal_binary_black_hole

    # Actually generate the waveform by calling the generator
    wf = frequency_domain_source_model(
        frequency_array,
        mass_1,
        mass_2,
        luminosity_distance,
        a_1,
        tilt_1,
        phi_12,
        a_2,
        tilt_2,
        phi_jl,
        theta_jn,
        phase,
        **kwargs
    )

    # Apply a phase shift per polarization
    # TODO accounting for the morse phase shift (hanabi wasn't accounting for sign(f))
    # should I multiply by np.sign(frequency_array)?
    wf["plus"] = np.exp(-1j * morse_phase) * np.ones_like(wf["plus"]) * wf["plus"]
    wf["cross"] = np.exp(-1j * morse_phase) * np.ones_like(wf["cross"]) * wf["cross"]

    return wf


def get_strains_from(ifos, ifo_names, start_time):
    strains = []
    for i in range(len(ifo_names)):
        strain_tmp = TimeSeries(
            initial_array=ifos[i].time_domain_strain,
            delta_t=ifos[i].time_array[1] - ifos[i].time_array[0],
            epoch=start_time,
        )
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
    is_real_noise=False,
    suffix="",
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
        waveform_generator=waveform_generator, parameters=injection_parameters
    )
    strains = get_strains_from(ifos, ifo_names, start_time)
    if is_real_noise:
        for i in range(len(ifo_names)):
            noise = get_strain_dict_from_files(
                FRAME_FILES,
                CHANNELS,
                [ifo_names[i] + suffix],
                {ifo_names[i] + suffix: start_time},
                {ifo_names[i] + suffix: end_time},
                PAD_SECONDS,
            )[ifo_names[i] + suffix]
            strains[i] = strains[i].inject(noise)

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
    is_real_noise=False,
    suffix="",
):
    duration = end_time - start_time

    # Set up a random seed for result reproducibility.  This is optional!
    bilby.core.utils.random.seed(seed)

    ifos = get_ifos_function(ifo_names, sampling_frequency, duration, start_time)

    strains = get_strains_from(ifos, ifo_names, start_time)
    if is_real_noise:
        for i in range(len(ifo_names)):
            noise = get_strain_dict_from_files(
                FRAME_FILES,
                CHANNELS,
                [ifo_names[i] + suffix],
                {ifo_names[i] + suffix: start_time},
                {ifo_names[i] + suffix: end_time},
                PAD_SECONDS,
            )[ifo_names[i] + suffix]
            strains[i] = strains[i].inject(noise)
    for i in range(len(ifo_names)):
        signal = _get_signal_from_pycbc(
            injection_parameters,
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
