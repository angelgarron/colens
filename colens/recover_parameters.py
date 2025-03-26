from dataclasses import dataclass

import numpy as np

D0 = 1


@dataclass
class AmplitudeParameters:
    A_1: float
    A_2: float
    A_3: float
    A_4: float
    A_p: float
    A_c: float


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


def physical_parameters_to_a(distance, iota, psi, phi):
    A_p = D0 / distance * (1 + np.cos(iota) ** 2) / 2
    A_c = D0 / distance * np.cos(iota)
    A_1 = A_p * np.cos(2 * phi) * np.cos(2 * psi) - A_c * np.sin(2 * phi) * np.sin(
        2 * psi
    )
    A_2 = A_p * np.cos(2 * phi) * np.sin(2 * psi) + A_c * np.sin(2 * phi) * np.cos(
        2 * psi
    )
    A_3 = -A_p * np.sin(2 * phi) * np.cos(2 * psi) - A_c * np.cos(2 * phi) * np.sin(
        2 * psi
    )
    A_4 = -A_p * np.sin(2 * phi) * np.sin(2 * psi) + A_c * np.cos(2 * phi) * np.cos(
        2 * psi
    )
    return AmplitudeParameters(A_1, A_2, A_3, A_4, A_p, A_c)


def XLALAmplitudeParams2Vect(distance, cosi, psi, phi0) -> tuple[float]:
    aPlus = 0.5 * distance * (1.0 + cosi**2)
    aCross = distance * cosi
    cos2psi = np.cos(2.0 * psi)
    sin2psi = np.sin(2.0 * psi)
    cosphi0 = np.cos(phi0)
    sinphi0 = np.sin(phi0)

    A_1 = aPlus * cos2psi * cosphi0 - aCross * sin2psi * sinphi0
    A_2 = aPlus * sin2psi * cosphi0 + aCross * cos2psi * sinphi0
    A_3 = -aPlus * cos2psi * sinphi0 - aCross * sin2psi * cosphi0
    A_4 = -aPlus * sin2psi * sinphi0 + aCross * cos2psi * cosphi0
    return A_1, A_2, A_3, A_4, aPlus, aCross


def XLALAmplitudeVect2Params(
    A1: float, A2: float, A3: float, A4: float
) -> tuple[float]:
    Asq = A1**2 + A2**2 + A3**2 + A4**2
    Da = A1 * A4 - A2 * A3

    disc = np.sqrt(Asq**2 - 4.0 * Da**2)

    Ap2 = 0.5 * (Asq + disc)
    aPlus = np.sqrt(Ap2)

    Ac2 = 0.5 * (Asq - disc)
    aCross = np.sign(Da) * np.sqrt(Ac2)

    beta = aCross / aPlus

    b1 = A4 - beta * A1
    b2 = A3 + beta * A2
    b3 = -A1 + beta * A4

    psiRet = 0.5 * np.arctan2(b1, b2)
    phi0Ret = np.arctan2(b2, b3)

    A1check = aPlus * np.cos(phi0Ret) * np.cos(2.0 * psiRet) - aCross * np.sin(
        phi0Ret
    ) * np.sin(2 * psiRet)
    if A1check * A1 < 0:
        phi0Ret += np.pi

    while psiRet > np.pi / 4:
        psiRet -= np.pi / 2
        phi0Ret -= np.pi

    while psiRet < -np.pi / 4:
        psiRet += np.pi / 2
        phi0Ret += np.pi

    while phi0Ret < 0:
        phi0Ret += 2 * np.pi

    while phi0Ret > 2 * np.pi:
        phi0Ret -= 2 * np.pi

    return aPlus, aCross, psiRet, phi0Ret


def aplus_across_to_distance_cosi(aplus, across):
    cosi = aplus / across + (aplus**2 / across**2 - 1) ** 0.5
    if np.abs(cosi) > 1:
        cosi = aplus / across - (aplus**2 / across**2 - 1) ** 0.5
    distance = across / cosi
    return distance, cosi
