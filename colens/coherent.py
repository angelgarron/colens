import numpy as np


def coherent_snr(
    snr_H1_at_trigger_original,
    snr_L1_at_trigger_original,
    snr_H1_at_trigger_lensed,
    snr_L1_at_trigger_lensed,
    projection_matrix,
):
    # Calculate rho_coh
    snr_array = np.array(
        [
            [snr_H1_at_trigger_original],
            [snr_H1_at_trigger_lensed],
            [snr_L1_at_trigger_original],
            [snr_L1_at_trigger_lensed],
        ]
    )
    snr_proj = np.inner(snr_array.conj().transpose(), projection_matrix)
    rho_coh2 = sum(snr_proj.transpose() * snr_array)
    rho_coh = abs(np.sqrt(rho_coh2))
    return rho_coh
