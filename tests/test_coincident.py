import numpy as np

from colens.coincident import coincident_snr


def test():
    expected = (2**2 + 3**2 + 5**2 + 7**2) ** 0.5
    np.testing.assert_equal(coincident_snr([2 + 3j, 5 + 7j]), expected)
    expected = (
        np.array([2**2 + 3**2 + 5**2 + 7**2, 11**2 + 13**2 + 17**2 + 19**2]) ** 0.5
    )
    np.testing.assert_equal(
        coincident_snr([[2 + 3j, 5 + 7j], [11 + 13j, 17 + 19j]]), expected
    )
