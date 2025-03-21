import numpy as np


def get_maximizing_A(M_mu_nu: np.ndarray, x_mu: np.ndarray) -> np.ndarray:
    M_sup_mu_sup_nu = np.linalg.inv(M_mu_nu)
    return M_sup_mu_sup_nu @ x_mu


def get_two_f(M_mu_nu: np.ndarray, x_mu: np.ndarray) -> float:
    A_sup_mu_max = get_maximizing_A(M_mu_nu, x_mu)
    return np.dot(x_mu, A_sup_mu_max)
