import numpy as np
from pycbc.events import coherent as coh

from colens.coherent import coherent_snr
from colens.detector import MyDetector
from colens.fstatistic import get_two_f


def test_coincident_snr():
    INSTRUMENTS = ["H1", "H1_lensed", "L1", "L1_lensed"]
    snr_H1_at_trigger_original = 1.5710026025772095 - 6.665505409240723j
    snr_H1_at_trigger_lensed = 0.6309635639190674 - 1.0601197481155396j
    snr_L1_at_trigger_original = 8.013982772827148 - 1.995304822921753j
    snr_L1_at_trigger_lensed = 2.1277353763580322 - 1.9622015953063965j
    snr_dict_at_trigger = {
        "H1": snr_H1_at_trigger_original,
        "H1_lensed": snr_H1_at_trigger_lensed,
        "L1": snr_L1_at_trigger_original,
        "L1_lensed": snr_L1_at_trigger_lensed,
    }
    ra = -1.6609360667509858
    dec = -0.3290311871510663
    original_trigger_time_seconds = 1185389807.2996583
    lensed_trigger_time_seconds = 1185437144.7885509
    fp = {
        ifo: MyDetector(ifo).antenna_pattern(
            ra,
            dec,
            polarization=0,
            t_gps=original_trigger_time_seconds,
        )[0]
        for ifo in ["H1", "L1"]
    }
    fp.update(
        {
            ifo: MyDetector(ifo).antenna_pattern(
                ra,
                dec,
                polarization=0,
                t_gps=lensed_trigger_time_seconds,
            )[0]
            for ifo in ["H1_lensed", "L1_lensed"]
        }
    )
    fc = {
        ifo: MyDetector(ifo).antenna_pattern(
            ra,
            dec,
            polarization=0,
            t_gps=original_trigger_time_seconds,
        )[1]
        for ifo in ["H1", "L1"]
    }
    fc.update(
        {
            ifo: MyDetector(ifo).antenna_pattern(
                ra,
                dec,
                polarization=0,
                t_gps=lensed_trigger_time_seconds,
            )[1]
            for ifo in ["H1_lensed", "L1_lensed"]
        }
    )
    sigma = {
        "H1": 68670.49828320602,
        "H1_lensed": 68929.4583128213,
        "L1": 67826.00808293221,
        "L1_lensed": 68568.44196882896,
    }
    project = coh.get_projection_matrix(fp, fc, sigma, projection="standard")

    w_p = np.array([sigma[ifo] * fp[ifo] for ifo in INSTRUMENTS])
    w_c = np.array([sigma[ifo] * fc[ifo] for ifo in INSTRUMENTS])

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

    x_mu = np.zeros(4)
    x_mu[0] = sum(
        [fp[ifo] * snr_dict_at_trigger[ifo].real * sigma[ifo] for ifo in INSTRUMENTS]
    )
    x_mu[1] = sum(
        [fc[ifo] * snr_dict_at_trigger[ifo].real * sigma[ifo] for ifo in INSTRUMENTS]
    )
    x_mu[2] = -sum(
        [fp[ifo] * snr_dict_at_trigger[ifo].imag * sigma[ifo] for ifo in INSTRUMENTS]
    )
    x_mu[3] = -sum(
        [fc[ifo] * snr_dict_at_trigger[ifo].imag * sigma[ifo] for ifo in INSTRUMENTS]
    )
    expected_coherent_snr = get_two_f(M_mu_nu, x_mu) ** 0.5
    assert (
        coherent_snr(
            snr_H1_at_trigger_original,
            snr_L1_at_trigger_original,
            snr_H1_at_trigger_lensed,
            snr_L1_at_trigger_lensed,
            project,
        )
        == expected_coherent_snr
    )
