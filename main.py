#!/usr/bin/env python3
"""Reconstruct an image contour with a Fourier epicycle animation."""

import argparse
from pathlib import Path

import cv2
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


def read_img(image_path: str) -> np.ndarray:
    img = cv2.imread(str(Path(image_path)))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return img


def c_n(list_x, list_y, n: int) -> complex:
    x, y = np.asarray(list_x, dtype=float), np.asarray(list_y, dtype=float)
    if len(x) == 0:
        raise ValueError("At least one contour point is required")
    indices = np.arange(len(x))
    phase = 2 * np.pi * n * indices / len(x)
    return np.sum((x - 1j * y) * np.exp(-1j * phase)) / len(x)


def get_coeffs(list_x, list_y, n_terms: int):
    if n_terms < 0:
        raise ValueError("n_terms must be non-negative")
    coefficients = [(c_n(list_x, list_y, 0), 0)]
    for frequency in range(1, n_terms + 1):
        coefficients.extend([
            (c_n(list_x, list_y, frequency), frequency),
            (c_n(list_x, list_y, -frequency), -frequency),
        ])
    return sorted(coefficients, key=lambda item: abs(item[0]), reverse=True)


def get_coordinates(img: np.ndarray):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    _, threshold = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("No contour found in the input image")
    contour = max(contours, key=cv2.contourArea).reshape(-1, 2)
    return contour[:, 0].astype(float) - contour[:, 0].mean(), contour[:, 1].astype(float) - contour[:, 1].mean()


def see_animation(coefficients, output_path: str, frames: int = 300, fps: int = 20):
    if frames < 2 or fps <= 0:
        raise ValueError("frames must be >= 2 and fps must be positive")
    fig, ax = plt.subplots(figsize=(9, 9))
    vectors = [ax.plot([], [])[0] for _ in coefficients]
    circles = [ax.plot([], [])[0] for _ in coefficients]
    drawing, = ax.plot([], [], linewidth=2)

    radius = max(1.0, sum(abs(c) for c, _ in coefficients))
    margin = radius * 0.08
    ax.set_xlim(-radius - margin, radius + margin)
    ax.set_ylim(-radius - margin, radius + margin)
    ax.set_axis_off()
    ax.set_aspect("equal")

    draw_x, draw_y = [], []
    times = np.linspace(0, 1, frames)

    def animate(frame):
        center = 0j
        current_time = times[frame]
        for index, (coefficient, frequency) in enumerate(coefficients):
            vector = coefficient * np.exp(1j * frequency * 2 * np.pi * current_time)
            next_center = center + vector
            theta = np.linspace(0, 2 * np.pi, 60)
            circles[index].set_data(
                center.real + abs(vector) * np.cos(theta),
                center.imag + abs(vector) * np.sin(theta),
            )
            vectors[index].set_data(
                [center.real, next_center.real], [center.imag, next_center.imag]
            )
            center = next_center
        draw_x.append(center.real)
        draw_y.append(center.imag)
        drawing.set_data(draw_x, draw_y)
        return [*vectors, *circles, drawing]

    anim = animation.FuncAnimation(fig, animate, frames=frames, interval=1000 / fps, blit=False)
    anim.save(output_path, fps=fps)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Draw an image contour using Fourier epicycles.")
    parser.add_argument("--input", default="image.jpg", help="Input image path")
    parser.add_argument("--output", default="file.gif", help="Output GIF path")
    parser.add_argument("--terms", type=int, default=100, help="Positive/negative Fourier frequencies")
    parser.add_argument("--frames", type=int, default=300, help="Animation frames")
    parser.add_argument("--fps", type=int, default=20, help="Animation frame rate")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    x, y = get_coordinates(read_img(args.input))
    coefficients = get_coeffs(x, y, args.terms)
    print(f"Using {len(coefficients)} coefficients")
    see_animation(coefficients, args.output, args.frames, args.fps)
    print(f"Saved animation to {args.output}")
