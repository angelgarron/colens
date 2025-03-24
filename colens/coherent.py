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


def coherent_statistic_adapter(
    snr_H1_at_trigger_original,
    snr_L1_at_trigger_original,
    snr_H1_at_trigger_lensed,
    snr_L1_at_trigger_lensed,
    sigma,
    fp,
    fc,
    instruments,
):
    snr_dict_at_trigger = {
        "H1": snr_H1_at_trigger_original,
        "H1_lensed": snr_H1_at_trigger_lensed,
        "L1": snr_L1_at_trigger_original,
        "L1_lensed": snr_L1_at_trigger_lensed,
    }

    w_p = np.array([sigma[ifo] * fp[ifo] for ifo in instruments])
    w_c = np.array([sigma[ifo] * fc[ifo] for ifo in instruments])

    A = np.dot(w_p, w_p)
    B = np.dot(w_c, w_c)
    C = np.dot(w_p, w_c)
    M_mu_nu = np.array(
        [
            [A, C, 0, 0],
            [C, B, 0, 0],
            [0, 0, A, C],
            [0, 0, C, B],
        ]
    )

    x_mu = np.zeros(4)
    x_mu[0] = sum(
        [fp[ifo] * snr_dict_at_trigger[ifo].real * sigma[ifo] for ifo in instruments]
    )
    x_mu[1] = sum(
        [fc[ifo] * snr_dict_at_trigger[ifo].real * sigma[ifo] for ifo in instruments]
    )
    x_mu[2] = -sum(
        [fp[ifo] * snr_dict_at_trigger[ifo].imag * sigma[ifo] for ifo in instruments]
    )
    x_mu[3] = -sum(
        [fc[ifo] * snr_dict_at_trigger[ifo].imag * sigma[ifo] for ifo in instruments]
    )
    return M_mu_nu, x_mu
