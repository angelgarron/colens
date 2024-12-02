import numpy as np


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
