"""Different functions for computing approximations to the B-statistic."""

import numpy as np
from scipy.special import gamma as gamma_func
from scipy.special import hyp1f1 as hyp1f1_func

from colens.bstatistic.convert import (
    convert_A_to_A_tilde,
    convert_A_to_polar_coordinates,
    convert_M_to_M_tilde,
)
from colens.fstatistic import get_maximizing_A, get_two_f


def whelan2014(M_mu_nu: np.ndarray, x_mu: np.ndarray, h0A: float) -> float:
    A_sup_mu_max = get_maximizing_A(M_mu_nu, x_mu)
    A_sup_mu_max_tilde = convert_A_to_A_tilde(A_sup_mu_max)
    k = (A_sup_mu_max_tilde[0] ** 2 + A_sup_mu_max_tilde[1] ** 2) * (
        A_sup_mu_max_tilde[2] ** 2 + A_sup_mu_max_tilde[3] ** 2
    )
    alpha = -3 / 4 * np.log(k)
    alpha_mu = (
        (-3 / 4 / k)
        * 2
        * A_sup_mu_max_tilde
        * np.array(
            [
                A_sup_mu_max_tilde[2] ** 2 + A_sup_mu_max_tilde[3] ** 2,
                A_sup_mu_max_tilde[2] ** 2 + A_sup_mu_max_tilde[3] ** 2,
                A_sup_mu_max_tilde[0] ** 2 + A_sup_mu_max_tilde[1] ** 2,
                A_sup_mu_max_tilde[0] ** 2 + A_sup_mu_max_tilde[1] ** 2,
            ]
        )
    )
    # fmt: off
    alpha_mu_nu = (
        -3
        / 4
        / k
        * 2
        * np.array(
            [
                [A_sup_mu_max_tilde[2] ** 2 + A_sup_mu_max_tilde[3] ** 2, 0, 2 * A_sup_mu_max_tilde[0] * A_sup_mu_max_tilde[2], 2 * A_sup_mu_max_tilde[0] * A_sup_mu_max_tilde[3]],
                [0, A_sup_mu_max_tilde[2] ** 2 + A_sup_mu_max_tilde[3] ** 2, 2 * A_sup_mu_max_tilde[1] * A_sup_mu_max_tilde[2], 2 * A_sup_mu_max_tilde[1] * A_sup_mu_max_tilde[3]],
                [2 * A_sup_mu_max_tilde[2] * A_sup_mu_max_tilde[0], 2 * A_sup_mu_max_tilde[2] * A_sup_mu_max_tilde[1], A_sup_mu_max_tilde[0] ** 2 + A_sup_mu_max_tilde[1] ** 2, 0],
                [2 * A_sup_mu_max_tilde[3] * A_sup_mu_max_tilde[0], 2 * A_sup_mu_max_tilde[3] * A_sup_mu_max_tilde[1], 0, A_sup_mu_max_tilde[0] ** 2 + A_sup_mu_max_tilde[1] ** 2],
            ]
        )
    )
    # fmt: on
    N_mu_nu = M_mu_nu - alpha_mu_nu
    f = get_two_f(M_mu_nu, x_mu) / 2
    return (
        np.exp(f + alpha)
        / (8 * np.pi**2 * 3 * h0A)
        * np.sqrt((2 * np.pi) ** 4 / np.linalg.det(N_mu_nu))
        * np.exp(0.5 * np.dot(alpha_mu, np.linalg.inv(N_mu_nu) @ alpha_mu))
    )


def dhurandhar2017(M_mu_nu: np.ndarray, x_mu: np.ndarray, h0A: float) -> float:
    f = get_two_f(M_mu_nu, x_mu) / 2
    A_sup_mu_max = get_maximizing_A(M_mu_nu, x_mu)
    A_r_max, A_l_max, phi_r_max, phi_l_max = convert_A_to_polar_coordinates(
        A_sup_mu_max[0], A_sup_mu_max[1], A_sup_mu_max[2], A_sup_mu_max[3]
    )
    M_mu_nu_tilde = convert_M_to_M_tilde(M_mu_nu)
    return (
        np.exp(f)
        / (2 * 3 * h0A)
        / (((A_r_max * A_l_max) ** (3 / 2)) * np.linalg.det(M_mu_nu_tilde) ** 0.5)
    )


def bero2018(M_mu_nu: np.ndarray, x_mu: np.ndarray, h0A: float) -> float:
    M_mu_nu_tilde = convert_M_to_M_tilde(M_mu_nu)
    I = M_mu_nu_tilde[0, 0]
    J = M_mu_nu_tilde[2, 2]
    L = M_mu_nu_tilde[0, 2]
    K = -M_mu_nu_tilde[0, 3]
    A_sup_mu_max = get_maximizing_A(M_mu_nu, x_mu)
    A_r_max, A_l_max, phi_r_max, phi_l_max = convert_A_to_polar_coordinates(
        A_sup_mu_max[0], A_sup_mu_max[1], A_sup_mu_max[2], A_sup_mu_max[3]
    )
    result = (
        1
        / (8 * np.pi**2 * 3 * h0A)
        * 2
        * np.pi
        * gamma_func(1 / 4)
        / (2 ** (3 / 4) * I ** (1 / 4))
        * 2
        * np.pi
        * gamma_func(1 / 4)
        / (2 ** (3 / 4) * J ** (1 / 4))
        * hyp1f1_func(0.25, 1, I * A_r_max**2 / 2)
        * hyp1f1_func(0.25, 1, J * A_l_max**2 / 2)
        * (
            1
            + (K * np.sin(phi_r_max - phi_l_max) + L * np.cos(phi_r_max - phi_l_max))
            * A_r_max
            * A_l_max
            * (
                0.25
                * hyp1f1_func(5 / 4, 2, I * A_r_max**2 / 2)
                / hyp1f1_func(0.25, 1, I * A_r_max**2 / 2)
                + 0.25
                * hyp1f1_func(5 / 4, 2, J * A_l_max**2 / 2)
                / hyp1f1_func(0.25, 1, J * A_l_max**2 / 2)
                - 1
                / 16
                * hyp1f1_func(5 / 4, 2, I * A_r_max**2 / 2)
                / hyp1f1_func(0.25, 1, I * A_r_max**2 / 2)
                * hyp1f1_func(5 / 4, 2, J * A_l_max**1 / 2)
                / hyp1f1_func(0.25, 1, J * A_l_max**2 / 2)
            )
        )
    )
    return result


def prix2024(M_mu_nu: np.ndarray, x_mu: np.ndarray, H: float = 1.0) -> float:
    f_a = (x_mu[0] ** 2 + x_mu[2] ** 2) / (2 * M_mu_nu[0, 0])
    f_b = (x_mu[1] ** 2 + x_mu[3] ** 2) / (2 * M_mu_nu[1, 1])
    return 1 + H**2 / 10 * (
        M_mu_nu[0, 0] * (2 * f_a - 2) + M_mu_nu[1, 1] * (2 * f_b - 2)
    )
