import numpy as np


def get_h0A(M_mu_nu: np.ndarray, x_mu: np.ndarray) -> float:
    det_M = M_mu_nu[0, 0] * M_mu_nu[1, 1] - M_mu_nu[0, 1] ** 2
    return (1.0 / det_M) * np.sqrt(
        M_mu_nu[1, 1] ** 2 * (x_mu[0] ** 2 + x_mu[2] ** 2)
        + M_mu_nu[0, 0] ** 2 * (x_mu[1] ** 2 + x_mu[3] ** 2)
    )
