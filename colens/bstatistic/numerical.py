import numpy as np
import scipy
import vegas

from colens.bstatistic.limits import get_h0A
from colens.recover_parameters import XLALAmplitudeParams2Vect


def numerical_bstatistic_4d(
    M_mu_nu: np.ndarray, x_mu: np.ndarray, nitn: int = 10, neval: int = 1000
) -> float:
    h0A = get_h0A(M_mu_nu, x_mu)

    def bstat_integrand(params: list[float]) -> float:
        h0, cosi, psi, phi0 = params
        A_1, A_2, A_3, A_4, aPlus, aCross = XLALAmplitudeParams2Vect(
            h0, cosi, psi, phi0
        )
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
    return result.val / (2 * np.pi**2 * 3 * h0A)


def numerical_bstatistic_2d(
    M_mu_nu: np.ndarray, x_mu: np.ndarray, nitn: int = 10, neval: int = 1000
) -> float:
    h0A = get_h0A(M_mu_nu, x_mu)

    def bstat_integrand(params: list[float]) -> float:
        eta = params[0]
        psi = params[1]

        etaSQ = eta**2
        etaSQp1SQ = (1.0 + etaSQ) ** 2

        sin2psi = np.sin(2.0 * psi)
        cos2psi = np.cos(2.0 * psi)
        sin2psiSQ = sin2psi**2
        cos2psiSQ = cos2psi**2

        al1 = 0.25 * etaSQp1SQ * cos2psiSQ + etaSQ * sin2psiSQ
        al2 = 0.25 * etaSQp1SQ * sin2psiSQ + etaSQ * cos2psiSQ
        al3 = 0.25 * (1.0 - etaSQ) ** 2 * sin2psi * cos2psi
        al4 = 0.5 * eta * (1.0 + etaSQ)

        gammaSQ = al1 * M_mu_nu[0, 0] + al2 * M_mu_nu[1, 1] + 2.0 * al3 * M_mu_nu[0, 1]

        qSQ = (
            al1 * (x_mu[0] ** 2 + x_mu[2] ** 2)
            + al2 * (x_mu[1] ** 2 + x_mu[3] ** 2)
            + 2.0 * al3 * (x_mu[0] * x_mu[1] + x_mu[2] * x_mu[3])
            + 2.0 * al4 * (x_mu[0] * x_mu[3] - x_mu[1] * x_mu[2])
        )

        Xi = 0.25 * qSQ / gammaSQ

        return np.exp(Xi) * gammaSQ ** (-0.5) * scipy.special.i0(Xi)

    integ = vegas.Integrator(
        [
            [-1, 1],  # cosi
            [-np.pi / 4, np.pi / 4],  # psi
        ]
    )

    result = integ(bstat_integrand, nitn=nitn, neval=neval)
    print(result.summary())
    print("result = %s    Q = %.2f" % (result, result.Q))
    # use extra factor 2 to integrate psi from -pi/4 to pi/4 instead of 0 to pi
    # one could also leave all prefactors, because the statistical power is not going to change
    # this way, the result would be the same as Bstat from synthesizeBstatMC.c
    return result.val * 2 / (2 * np.pi**2 * 3 * h0A)
