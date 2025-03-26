import numpy as np
import pytest

from colens.coherent import coherent_statistic_adapter
from colens.fstatistic import get_two_f


@pytest.fixture
def M_mu_nu():
    return np.array(
        [
            [0.17, 0.50, 0.67, 0.93],
            [0.43, 0.23, 0.51, 0.42],
            [0.65, 0.71, 0.18, 0.22],
            [0.58, 0.89, 0.65, 0.47],
        ]
    )


@pytest.fixture
def x_mu():
    return np.array([0.54, 0.68, 0.21, 0.87])


def test_get_two_f(M_mu_nu, x_mu):
    M_sup_mu_sup_nu = np.linalg.inv(M_mu_nu)
    expected_two_f = np.sum(
        np.array(
            [
                [x_mu[i] * M_sup_mu_sup_nu[i, j] * x_mu[j] for j in range(4)]
                for i in range(4)
            ]
        )
    )
    np.testing.assert_allclose(
        get_two_f(M_mu_nu, x_mu),
        expected_two_f,
    )
