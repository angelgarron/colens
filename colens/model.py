import bilby
import numpy as np


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
