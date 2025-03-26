import numpy as np
import pytest

from colens.recover_parameters import (
    XLALAmplitudeParams2Vect,
    XLALAmplitudeVect2Params,
    aplus_across_to_distance_cosi,
)

rng = np.random.default_rng(1234)


@pytest.mark.parametrize(
    "distance, cosi, psi, phi0",
    [
        *rng.uniform(
            low=(0, -1, -np.pi / 4, 0),
            high=(10, 1, np.pi / 4, 2 * np.pi),
            size=(10, 4),
        )
    ],
)
def test_identity_recover_parameters(distance, cosi, psi, phi0):
    A1, A2, A3, A4, aPluss, aCross = XLALAmplitudeParams2Vect(distance, cosi, psi, phi0)
    aPlus_recovered, aCross_recovered, psi_recovered, phi0_recovered = (
        XLALAmplitudeVect2Params(A1, A2, A3, A4)
    )
    distance_recovered, cosi_recovered = aplus_across_to_distance_cosi(
        aPlus_recovered, aCross_recovered
    )
    np.testing.assert_allclose(
        [distance_recovered, cosi_recovered, psi_recovered, phi0_recovered],
        [distance, cosi, psi, phi0],
    )
