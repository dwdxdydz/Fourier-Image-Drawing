# 🌀 Fourier Image Drawing

## What is this project?

This project takes an image, finds its main outline, and redraws that outline using a collection of rotating circles.

The interesting part is that the circles are not chosen manually. Mathematics is used to calculate how they should rotate so that their combined movement recreates the original outline.

This is based on **Fourier analysis**, a mathematical technique for representing a complicated signal using simpler repeating components.

## Simple idea

Imagine tracing the outline of a shape with your finger.

The program records points along that outline and treats the points as a signal. Fourier analysis then breaks that signal into components with different frequencies and sizes.

Those components are visualised as rotating circles, called **epicycles**.

```text
Input image
    ↓
Find the outline
    ↓
Record outline points
    ↓
Represent points as numbers
    ↓
Calculate Fourier components
    ↓
Create rotating circles
    ↓
Combine their movement
    ↓
Reconstruct the outline
    ↓
Create animation / GIF
```

## What does the result mean?

A small number of Fourier components gives a rough version of the shape.

More components preserve more detail:

```text
Few terms       → rough shape
More terms      → better shape
Many terms      → more detailed shape
```

There is therefore a trade-off between **detail and rendering work**.

## Main features

- Reads an input image.
- Finds the largest useful contour.
- Converts 2D points into complex numbers.
- Calculates discrete Fourier coefficients.
- Uses positive and negative frequency components.
- Reconstructs the outline using epicycles.
- Creates an animation.
- Exports the result as a GIF.
- Supports command-line options.
- Validates input before processing.
- Includes automated tests and CI support.

## Run it

Install the requirements:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python main.py --input image.jpg --output file.gif --terms 100 --frames 300 --fps 20
```

### What do these options mean?

- `--input` — image you want to redraw.
- `--output` — GIF file that will be created.
- `--terms` — number of Fourier components used. More terms usually preserve more detail.
- `--frames` — number of animation frames.
- `--fps` — number of frames shown per second.

## Project structure

```text
main.py       → Main application and reconstruction pipeline
image.jpg     → Example input image
file.gif      → Example generated output
draw.py       → Earlier drawing experiment
working.py    → Development experiment
tests/        → Automated tests
```

## Main technologies

- **Python** — application logic
- **NumPy** — numerical calculations and arrays
- **Matplotlib** — drawing and animation
- **Pillow** — image processing and contour preparation
- **Pytest** — automated tests
- **GitHub Actions** — automated CI checks

## Technical terms explained

**Fourier analysis** — A mathematical method for breaking a complicated signal into simpler repeating components. Here, the image outline is treated as a signal.

**Fourier series** — A method of representing a repeating signal as a combination of simpler waves or equivalent rotating components.

**Fourier coefficient** — A value that describes how strongly a particular frequency contributes to the final reconstruction.

**Frequency** — How quickly a component repeats or rotates. Different frequencies help represent different levels of detail in the outline.

**Complex number** — A number with a real part and an imaginary part. In this project, a complex number is a convenient way to store an `(x, y)` point as one value.

**Contour** — The boundary or outline of an object in an image.

**Epicycle** — A rotating circle whose centre can itself move. Multiple epicycles can be connected together to create complex paths.

**Reconstruction** — Creating an approximation of the original outline from the Fourier components.

**Signal** — A sequence of values that carries information. Here, the changing x/y coordinates of the outline are treated as a signal.

**NumPy** — A Python library designed for fast numerical calculations and array operations.

**Matplotlib** — A Python library used to create plots, drawings and animations.

**CLI (Command-Line Interface)** — A way of controlling a program by typing commands in a terminal.

**GIF** — An image format that can contain a sequence of frames to create a simple animation.

**CI (Continuous Integration)** — Automatic checks, such as tests, that run when code changes are pushed to GitHub.

## What does this project demonstrate?

This project connects mathematics, image processing and visualisation:

**Image → contour → numerical signal → Fourier analysis → rotating components → reconstructed drawing**

It demonstrates practical **Python, NumPy, image processing, Fourier analysis, mathematical modelling, visualisation, animation, CLI design and testing** skills.

## Future improvements

- Use FFT-based coefficient calculation for faster processing.
- Improve contour cleaning and resampling.
- Add reconstruction performance benchmarks.
- Add interactive controls for Fourier terms.
- Add visual explanations of the mathematics.
