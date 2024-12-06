"""Functions used to inject simulated signals."""

import bilby
from pycbc.detector import Detector
from pycbc.types.timeseries import TimeSeries
from pycbc.waveform import get_td_waveform


def get_strain_list_from_simulation(
    injection_parameters,
    ifo_names,
    start_time,
    end_time,
    low_frequency_cutoff,
    seed,
    approximant,
    inject_from_pycbc=False,
):
    sampling_frequency = 4096.0
    duration = end_time - start_time

    # Set up a random seed for result reproducibility.  This is optional!
    bilby.core.utils.random.seed(seed)

    waveform_arguments = dict(
        waveform_approximant=approximant,
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
    ifos.set_strain_data_from_zero_noise(
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
        signal = _get_strain_list_from_pycbc_injection(
            injection_parameters,
            low_frequency_cutoff,
            approximant,
            injection_parameters["geocent_time"],
            ifo_names[i],
        )
        strain_tmp = TimeSeries(
            initial_array=ifos[i].time_domain_strain,
            delta_t=ifos[i].time_array[1] - ifos[i].time_array[0],
            epoch=start_time,
        )
        if inject_from_pycbc:
            strain_tmp = strain_tmp.inject(signal)
        strains.append(strain_tmp)

    return strains


def _get_strain_list_from_pycbc_injection(
    injection_parameters,
    low_frequency_cutoff,
    approximant,
    end_time,
    ifo,
):
    hp, hc = get_td_waveform(
        approximant=approximant,
        mass1=injection_parameters["mass_1"],
        mass2=injection_parameters["mass_2"],
        spinz1=injection_parameters["a_1"],
        spinz2=injection_parameters["a_2"],
        inclination=injection_parameters["theta_jn"],
        coa_phase=injection_parameters["phase"],
        delta_t=1.0 / 4096,
        f_lower=low_frequency_cutoff,
    )
    hp.start_time += end_time
    hc.start_time += end_time

    declination = injection_parameters["dec"]
    right_ascension = injection_parameters["ra"]
    polarization = injection_parameters["psi"]

    det = Detector(ifo)
    signal = det.project_wave(hp, hc, right_ascension, declination, polarization)
    return signal / injection_parameters["luminosity_distance"]
