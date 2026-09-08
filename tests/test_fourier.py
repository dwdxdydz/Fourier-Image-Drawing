import numpy as np
import pytest

from main import c_n, get_coeffs, get_coordinates, see_animation


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


def test_invalid_mismatched_coordinate_lengths():
    with pytest.raises(ValueError, match="same length"):
        c_n([0, 1], [0], 0)


def test_get_coordinates_accepts_grayscale_images():
    image = np.full((80, 80), 255, dtype=np.uint8)
    image[20:61, 20:61] = 0

    x, y = get_coordinates(image)

    assert len(x) == len(y)
    assert len(x) > 0
    assert np.isclose(x.mean(), 0)
    assert np.isclose(y.mean(), 0)


def test_animation_requires_gif_output(tmp_path):
    with pytest.raises(ValueError, match=".gif"):
        see_animation([(1 + 0j, 0)], tmp_path / "drawing.mp4", frames=2)
