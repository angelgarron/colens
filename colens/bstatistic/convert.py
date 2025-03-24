import numpy as np

from colens.recover_parameters import XLALAmplitudeVect2Params


def convert_M_to_M_tilde(M_mu_nu: np.ndarray) -> np.ndarray:
    A = M_mu_nu[0, 0]
    B = M_mu_nu[1, 1]
    C = M_mu_nu[0, 1]
    I = A + B
    J = A + B
    K = 2 * C
    L = A - B
    return np.array(
        [
            [I, 0, L, -K],
            [0, I, K, L],
            [L, K, J, 0],
            [-K, L, 0, J],
        ]
    )


def convert_A_to_A_tilde(
    A_sup_mu_1: float, A_sup_mu_2: float, A_sup_mu_3: float, A_sup_mu_4: float
) -> tuple[float]:
    A_sup_mu_tilde_1 = (A_sup_mu_1 + A_sup_mu_4) / 2
    A_sup_mu_tilde_2 = (A_sup_mu_2 - A_sup_mu_3) / 2
    A_sup_mu_tilde_3 = (A_sup_mu_1 - A_sup_mu_4) / 2
    A_sup_mu_tilde_4 = (-A_sup_mu_2 - A_sup_mu_3) / 2
    return A_sup_mu_tilde_1, A_sup_mu_tilde_2, A_sup_mu_tilde_3, A_sup_mu_tilde_4


def convert_A_to_polar_coordinates(
    A_sup_mu_1: float, A_sup_mu_2: float, A_sup_mu_3: float, A_sup_mu_4: float
) -> tuple[float]:
    aPlus, aCross, psi, phi0 = XLALAmplitudeVect2Params(
        A_sup_mu_1, A_sup_mu_2, A_sup_mu_3, A_sup_mu_4
    )
    A_r = (aPlus + aCross) / 2
    A_l = (aPlus - aCross) / 2
    phi_r = phi0 + 2 * psi
    phi_l = phi0 - 2 * psi
    return A_r, A_l, phi_r, phi_l
