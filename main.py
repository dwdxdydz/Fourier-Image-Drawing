#!/usr/bin/env python3
"""Reconstruct an image contour with a Fourier epicycle animation."""

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter


def read_img(image_path: str | Path) -> np.ndarray:
    """Read an image or raise an actionable error when it cannot be opened."""
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Could not read image: {image_path}")
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def c_n(list_x: Sequence[float], list_y: Sequence[float], n: int) -> complex:
    """Return the discrete Fourier coefficient at frequency ``n``."""
    x, y = np.asarray(list_x, dtype=float), np.asarray(list_y, dtype=float)
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("Contour coordinates must be one-dimensional sequences")
    if len(x) == 0:
        raise ValueError("At least one contour point is required")
    if len(x) != len(y):
        raise ValueError("x and y coordinate sequences must have the same length")
    indices = np.arange(len(x))
    phase = 2 * np.pi * n * indices / len(x)
    return np.sum((x - 1j * y) * np.exp(-1j * phase)) / len(x)


def get_coeffs(
    list_x: Sequence[float], list_y: Sequence[float], n_terms: int
) -> list[tuple[complex, int]]:
    """Build and magnitude-sort the zero, positive, and negative frequencies."""
    if n_terms < 0:
        raise ValueError("n_terms must be non-negative")
    coefficients = [(c_n(list_x, list_y, 0), 0)]
    for frequency in range(1, n_terms + 1):
        coefficients.extend([
            (c_n(list_x, list_y, frequency), frequency),
            (c_n(list_x, list_y, -frequency), -frequency),
        ])
    return sorted(coefficients, key=lambda item: abs(item[0]), reverse=True)


def _largest_component(mask: np.ndarray) -> np.ndarray | None:
    """Return the largest 8-connected component in a boolean image."""
    visited = np.zeros(mask.shape, dtype=bool)
    largest: list[tuple[int, int]] = []
    height, width = mask.shape
    for start_y, start_x in np.argwhere(mask):
        if visited[start_y, start_x]:
            continue
        stack = [(start_y, start_x)]
        visited[start_y, start_x] = True
        component: list[tuple[int, int]] = []
        while stack:
            y, x = stack.pop()
            component.append((y, x))
            for next_y in range(max(0, y - 1), min(height, y + 2)):
                for next_x in range(max(0, x - 1), min(width, x + 2)):
                    if mask[next_y, next_x] and not visited[next_y, next_x]:
                        visited[next_y, next_x] = True
                        stack.append((next_y, next_x))
        if len(component) > len(largest):
            largest = component
    if not largest:
        return None
    return np.asarray(largest, dtype=int)


def _otsu_threshold(gray: np.ndarray) -> int:
    """Calculate an Otsu threshold for an unsigned 8-bit grayscale image."""
    histogram = np.bincount(gray.ravel(), minlength=256).astype(float)
    total = gray.size
    cumulative_weight = np.cumsum(histogram)
    cumulative_mean = np.cumsum(histogram * np.arange(256))
    mean = cumulative_mean[-1]
    denominator = cumulative_weight * (total - cumulative_weight)
    variance = np.divide(
        (mean * cumulative_weight - cumulative_mean) ** 2,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    return int(np.argmax(variance))


def get_coordinates(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract a centered boundary for the largest foreground shape in an image."""
    if img is None or img.size == 0:
        raise ValueError("Input image must not be empty")
    if img.ndim == 2:
        gray = img.astype(np.uint8, copy=False)
    elif img.ndim == 3 and img.shape[2] == 3:
        gray = np.dot(img[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
    elif img.ndim == 3 and img.shape[2] == 4:
        gray = np.dot(img[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
    else:
        raise ValueError("Input image must be grayscale, RGB, or RGBA")
    blurred = np.asarray(Image.fromarray(gray).filter(ImageFilter.GaussianBlur(radius=1)))
    threshold = _otsu_threshold(blurred)
    components = [_largest_component(blurred <= threshold), _largest_component(blurred > threshold)]
    non_border_components = [
        component
        for component in components
        if component is not None
        and not np.any(
            (component[:, 0] == 0)
            | (component[:, 0] == gray.shape[0] - 1)
            | (component[:, 1] == 0)
            | (component[:, 1] == gray.shape[1] - 1)
        )
    ]
    candidates = non_border_components or [component for component in components if component is not None]
    if not candidates:
        raise ValueError("No contour found in the input image")
    component = max(candidates, key=len)
    component_mask = np.zeros(gray.shape, dtype=bool)
    component_mask[component[:, 0], component[:, 1]] = True
    interior = component_mask.copy()
    interior[1:-1, 1:-1] &= (
        component_mask[:-2, 1:-1]
        & component_mask[2:, 1:-1]
        & component_mask[1:-1, :-2]
        & component_mask[1:-1, 2:]
    )
    boundary = np.argwhere(component_mask & ~interior)
    center = boundary.mean(axis=0)
    order = np.argsort(np.arctan2(boundary[:, 0] - center[0], boundary[:, 1] - center[1]))
    boundary = boundary[order]
    x, y = boundary[:, 1].astype(float), boundary[:, 0].astype(float)
    return x - x.mean(), y - y.mean()


def see_animation(
    coefficients: Sequence[tuple[complex, int]],
    output_path: str | Path,
    frames: int = 300,
    fps: int = 20,
) -> None:
    """Save a GIF of Fourier epicycles tracing the supplied coefficients."""
    if frames < 2 or fps <= 0:
        raise ValueError("frames must be >= 2 and fps must be positive")
    if not coefficients:
        raise ValueError("At least one Fourier coefficient is required")
    output = Path(output_path)
    if output.suffix.lower() != ".gif":
        raise ValueError("Output path must use the .gif extension")
    output.parent.mkdir(parents=True, exist_ok=True)
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
    # Do not duplicate the starting point in the final frame.
    times = np.linspace(0, 1, frames, endpoint=False)

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
    try:
        anim.save(output, fps=fps, writer="pillow")
    finally:
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
