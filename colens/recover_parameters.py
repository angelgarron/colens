import numpy as np


def get_A_parameters(sigma, fp, fc, snr_array_at_trigger, instruments):
    w_p = np.array([sigma[ifo] * fp[ifo] for ifo in instruments])
    w_c = np.array([sigma[ifo] * fc[ifo] for ifo in instruments])
    denom = np.dot(w_p, w_p) * np.dot(w_c, w_c) - np.dot(w_p, w_c) ** 2
    Ap = (
        np.dot(w_c, w_c) * np.dot(w_p, snr_array_at_trigger)
        - np.dot(w_p, w_c) * np.dot(w_c, snr_array_at_trigger)
    ) / denom
    Ac = (
        -np.dot(w_p, w_c) * np.dot(w_p, snr_array_at_trigger)
        + np.dot(w_p, w_p) * np.dot(w_c, snr_array_at_trigger)
    ) / denom
    A_1 = Ap.real
    A_2 = Ac.real
    A_3 = -Ap.imag
    A_4 = -Ac.imag
    return A_1, A_2, A_3, A_4


def recover_parameters(A1, A2, A3, A4):
    Ap = (
        np.sqrt((A1 - A4) ** 2 + (A2 + A3) ** 2)
        + np.sqrt((A1 + A4) ** 2 + (A2 - A3) ** 2)
    ) / 2
    Ac = (
        -np.sqrt((A1 - A4) ** 2 + (A2 + A3) ** 2)
        + np.sqrt((A1 + A4) ** 2 + (A2 - A3) ** 2)
    ) / 2

    cos_iota = (Ap - np.sqrt(Ap**2 - Ac**2)) / Ac
    iota = np.arccos(cos_iota)

    phi = -0.5 * np.arctan((Ap * A4 - Ac * A1) / (Ap * A2 + Ac * A3))

    psi = -0.5 * np.arctan((Ap * A4 - Ac * A1) / (-Ac * A2 - Ap * A3))

    distance = cos_iota / Ac

    return {
        "iota": iota,
        "phi": phi,
        "psi": psi,
        "distance": distance,
        "Ap": Ap,
        "Ac": Ac,
    }
