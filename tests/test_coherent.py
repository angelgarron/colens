import numpy as np

from colens.coherent import coherent_statistic_adapter


def test_coherent_statistic_adapter():
    fp = [1.0, 2.0, 3.0, 4.0]
    fc = [5.0, 6.0, 7.0, 8.0]
    sigma = [9.0, 10.0, 11.0, 12.0]
    snr_at_trigger = [13.0 + 31.0j, 14.0 + 41.0j, 15.0 + 51.0j, 16.0 + 61.0j]

    w_p = [sigma[0] * fp[0], sigma[1] * fp[1], fp[2] * sigma[2], fp[3] * sigma[3]]
    w_c = [sigma[0] * fc[0], sigma[1] * fc[1], fc[2] * sigma[2], fc[3] * sigma[3]]

    A = w_p[0] ** 2 + w_p[1] ** 2 + w_p[2] ** 2 + w_p[3] ** 2
    B = w_c[0] ** 2 + w_c[1] ** 2 + w_c[2] ** 2 + w_c[3] ** 2
    C = w_p[0] * w_c[0] + w_p[1] * w_c[1] + w_p[2] * w_c[2] + w_p[3] * w_c[3]
    expected_M_mu_nu = np.array(
        [
            [A, C, 0, 0],
            [C, B, 0, 0],
            [0, 0, A, C],
            [0, 0, C, B],
        ]
    )

    expected_x_mu = [
        fp[0] * snr_at_trigger[0].real * sigma[0]
        + fp[1] * snr_at_trigger[1].real * sigma[1]
        + fp[2] * snr_at_trigger[2].real * sigma[2]
        + fp[3] * snr_at_trigger[3].real * sigma[3],
        fc[0] * snr_at_trigger[0].real * sigma[0]
        + fc[1] * snr_at_trigger[1].real * sigma[1]
        + fc[2] * snr_at_trigger[2].real * sigma[2]
        + fc[3] * snr_at_trigger[3].real * sigma[3],
        -fp[0] * snr_at_trigger[0].imag * sigma[0]
        - fp[1] * snr_at_trigger[1].imag * sigma[1]
        - fp[2] * snr_at_trigger[2].imag * sigma[2]
        - fp[3] * snr_at_trigger[3].imag * sigma[3],
        -fc[0] * snr_at_trigger[0].imag * sigma[0]
        - fc[1] * snr_at_trigger[1].imag * sigma[1]
        - fc[2] * snr_at_trigger[2].imag * sigma[2]
        - fc[3] * snr_at_trigger[3].imag * sigma[3],
    ]

    instruments = ["H1", "H1_lensed", "L1", "L1_lensed"]
    M_mu_nu, x_mu = coherent_statistic_adapter(
        snr_at_trigger[0],
        snr_at_trigger[2],
        snr_at_trigger[1],
        snr_at_trigger[3],
        {ifo: sigma[i] for i, ifo in enumerate(instruments)},
        {ifo: fp[i] for i, ifo in enumerate(instruments)},
        {ifo: fc[i] for i, ifo in enumerate(instruments)},
        instruments,
    )
    np.testing.assert_equal(M_mu_nu, expected_M_mu_nu)
    np.testing.assert_equal(x_mu, expected_x_mu)
