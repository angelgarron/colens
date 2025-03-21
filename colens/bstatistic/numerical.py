import numpy as np
import vegas

from colens.recover_parameters import XLALAmplitudeParams2Vect


def numerical_bstatistic(
    M_mu_nu: np.ndarray, x_mu: np.ndarray, nitn: int = 10, neval: int = 1000
) -> float:
    det_M = M_mu_nu[0, 0] * M_mu_nu[1, 1] - M_mu_nu[0, 1] ** 2
    h0A = (1.0 / det_M) * np.sqrt(
        M_mu_nu[1, 1] ** 2 * (x_mu[0] ** 2 + x_mu[2] ** 2)
        + M_mu_nu[0, 0] ** 2 * (x_mu[1] ** 2 + x_mu[3] ** 2)
    )

    def bstat_integrand(params: list[float]) -> float:
        h0, cosi, psi, phi0 = params
        A_1, A_2, A_3, A_4 = XLALAmplitudeParams2Vect(h0, cosi, psi, phi0)
        A_sup_mu = np.array([A_1, A_2, A_3, A_4])
        Ax = np.dot(A_sup_mu, x_mu)
        rho2 = np.dot(A_sup_mu, M_mu_nu @ A_sup_mu)
        lnL = Ax - 0.5 * rho2
        return np.exp(lnL)

    integ = vegas.Integrator(
        [
            [0, 3 * h0A],  # h0
            [-1, 1],  # cosi
            [0, np.pi],  # psi
            [0, 2 * np.pi],  # phi0
        ]
    )

    result = integ(bstat_integrand, nitn=nitn, neval=neval)
    print(result.summary())
    print("result = %s    Q = %.2f" % (result, result.Q))
    return result.val
