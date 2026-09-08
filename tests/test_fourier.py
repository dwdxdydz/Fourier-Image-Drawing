import numpy as np
import pytest

from main import c_n, get_coeffs


def test_constant_signal_has_zero_first_harmonic():
    x = np.ones(16) * 3
    y = np.zeros(16)
    assert abs(c_n(x, y, 0) - 3) < 1e-9
    assert abs(c_n(x, y, 1)) < 1e-9


def test_coefficients_include_positive_and_negative_frequencies():
    coeffs = get_coeffs([0, 1, 1, 0], [0, 0, 1, 1], 2)
    frequencies = {frequency for _, frequency in coeffs}
    assert frequencies == {-2, -1, 0, 1, 2}


def test_invalid_empty_signal():
    with pytest.raises(ValueError):
        c_n([], [], 0)
