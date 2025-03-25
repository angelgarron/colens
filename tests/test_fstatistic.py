import numpy as np
import pytest

from colens.coherent import coherent_statistic_adapter
from colens.fstatistic import get_two_f


@pytest.fixture
def M_mu_nu():
    return np.array(
        [
            [5.27449863e08, -4.98476855e08, 0.00000000e00, 0.00000000e00],
            [-4.98476855e08, 7.05533110e09, 0.00000000e00, 0.00000000e00],
            [0.00000000e00, 0.00000000e00, 5.27449863e08, -4.98476855e08],
            [0.00000000e00, 0.00000000e00, -4.98476855e08, 7.05533110e09],
        ]
    )


@pytest.fixture
def x_mu():
    return np.array(
        [101478.72780461, 227021.81579813, 137157.94876497, -306334.07181856]
    )


def test_coherent_snr(M_mu_nu, x_mu):
    expected_coherent_snr = 8.704664414533402
    np.testing.assert_allclose(
        get_two_f(M_mu_nu, x_mu) ** 0.5,
        expected_coherent_snr,
    )


def test_coherent_statistic_adapter(M_mu_nu, x_mu):
    snr_H1_at_trigger_original = 1.5710026025772095 - 6.665505409240723j
    snr_H1_at_trigger_lensed = 0.6309635639190674 - 1.0601197481155396j
    snr_L1_at_trigger_original = 8.013982772827148 - 1.995304822921753j
    snr_L1_at_trigger_lensed = 2.1277353763580322 - 1.9622015953063965j
    sigma = {
        "H1": 68670.49828320602,
        "H1_lensed": 68929.4583128213,
        "L1": 67826.00808293221,
        "L1_lensed": 68568.44196882896,
    }
    instruments = ["H1", "H1_lensed", "L1", "L1_lensed"]
    fp = {
        "H1": 0.18965039616799562,
        "L1": 0.08232781300940539,
        "H1_lensed": 0.17476991152155769,
        "L1_lensed": 0.19649714442030164,
    }
    fc = {
        "H1": -0.8074485293373161,
        "L1": 0.6607537537291653,
        "H1_lensed": 0.46580796967133664,
        "L1_lensed": -0.4474797457859099,
    }
    M_mu_nu_computed, x_mu_computed = coherent_statistic_adapter(
        snr_H1_at_trigger_original,
        snr_L1_at_trigger_original,
        snr_H1_at_trigger_lensed,
        snr_L1_at_trigger_lensed,
        sigma,
        fp,
        fc,
        instruments,
    )
    np.testing.assert_allclose(M_mu_nu_computed, M_mu_nu)
    np.testing.assert_allclose(x_mu_computed, x_mu)
