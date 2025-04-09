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


def coherent_statistic_adapter(snr_at_trigger, sigma, fp, fc):
    snr_at_trigger = np.asarray(snr_at_trigger)
    sigma = np.asarray(sigma)
    fp = np.asarray(fp)
    fc = np.asarray(fc)
    w_p = sigma * fp
    w_c = sigma * fc

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

    x_mu = [
        sum(w_p * snr_at_trigger.real),
        sum(w_c * snr_at_trigger.real),
        -sum(w_p * snr_at_trigger.imag),
        -sum(w_c * snr_at_trigger.imag),
    ]
    return M_mu_nu, x_mu
