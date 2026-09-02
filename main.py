#!/usr/bin/python3
# coding: utf-8

from pathlib import Path

import cv2
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp


def read_img(image_path: str = "image.jpg") -> np.ndarray:
    """Load an input image and fail with a useful message when unavailable."""
    path = Path(image_path)
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def c_n(list_x, list_y, n: int) -> complex:
    """Compute the discrete Fourier coefficient for frequency n."""
    x = np.asarray(list_x, dtype=float)
    y = np.asarray(list_y, dtype=float)
    sample_count = len(x)
    if sample_count == 0:
        raise ValueError("At least one contour point is required")

    indices = np.arange(sample_count)
    phase = 2 * np.pi * n * indices / sample_count
    return np.sum((x - 1j * y) * np.exp(-1j * phase)) / sample_count


def get_coeffs(list_x, list_y, n_terms: int):
    """Return Fourier coefficients for frequencies from -n_terms to n_terms."""
    coefficients = [(c_n(list_x, list_y, 0), 0)]
    for frequency in range(1, n_terms + 1):
        coefficients.extend(
            [(c_n(list_x, list_y, frequency), frequency),
             (c_n(list_x, list_y, -frequency), -frequency)]
        )
    return coefficients


def get_coordinates(img: np.ndarray):
    """Extract and center the largest contour from an image."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    _, threshold = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )
    contours, _ = cv2.findContours(
        threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        raise ValueError("No contour found in the input image")

    contour = max(contours, key=cv2.contourArea).reshape(-1, 2)
    x = contour[:, 0].astype(float)
    y = contour[:, 1].astype(float)
    return x - x.mean(), y - y.mean()


def get_circle_coords(center, radius, points: int = 80):
    theta = np.linspace(0, 2 * np.pi, points)
    return (
        center[0] + radius * np.cos(theta),
        center[1] + radius * np.sin(theta),
    )


def get_next_pos(coefficient: complex, frequency: int, time: float, drawing_time: float = 1):
    angle = frequency * 2 * np.pi * time / drawing_time
    return coefficient * np.exp(1j * angle)


def print_eqn(coefficients):
    """Print a symbolic Fourier-series representation."""
    t = sp.symbols("t", real=True)
    series = sum(c * sp.exp(sp.I * 2 * sp.pi * n * t) for c, n in coefficients)
    print("Equation:")
    print(sp.latex(sp.expand_complex(series)))


def see_animation(coefficients, output_path: str = "file.gif", terms: int = 300):
    fig, ax = plt.subplots(figsize=(10, 10))
    vectors = [ax.plot([], [])[0] for _ in coefficients]
    circles = [ax.plot([], [])[0] for _ in coefficients]
    drawing, = ax.plot([], [], linewidth=2)

    ax.set_xlim(-500, 500)
    ax.set_ylim(-500, 500)
    ax.set_axis_off()
    ax.set_aspect("equal")

    draw_x, draw_y = [], []
    time = np.linspace(0, 1, num=terms)

    def animate(frame):
        current_time = time[frame]
        center = 0j
        for index, (coefficient, frequency) in enumerate(coefficients):
            vector = get_next_pos(coefficient, frequency, current_time)
            next_center = center + vector
            radius = abs(vector)
            circle_x, circle_y = get_circle_coords((center.real, center.imag), radius)
            circles[index].set_data(circle_x, circle_y)
            vectors[index].set_data(
                [center.real, next_center.real], [center.imag, next_center.imag]
            )
            center = next_center

        draw_x.append(center.real)
        draw_y.append(center.imag)
        drawing.set_data(draw_x, draw_y)
        return [*vectors, *circles, drawing]

    anim = animation.FuncAnimation(
        fig, animate, frames=len(time), interval=5, blit=False
    )
    anim.save(output_path, fps=15)
    plt.close(fig)


if __name__ == "__main__":
    image = read_img()
    x, y = get_coordinates(image)
    coefficients = get_coeffs(x, y, n_terms=300)
    print_eqn(coefficients)
    see_animation(coefficients)
