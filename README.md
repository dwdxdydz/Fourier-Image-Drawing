# 🌀 Fourier Image Drawing

## What is this project?

This project takes an image, finds its main outline, and then **redraws that outline using rotating circles**.

The result looks like a set of circles rotating around one another until their combined movement draws the original shape.

This is based on **Fourier analysis**, a mathematical technique for representing a complicated signal using simpler waves.

## What does it look like conceptually?

```text
Input image
    ↓
Find the outline
    ↓
Turn outline points into numbers
    ↓
Calculate Fourier components
    ↓
Use rotating circles
    ↓
Reconstruct the outline
    ↓
Create an animation / GIF
```

## Example idea

If the input image contains a simple shape, the application records points along its boundary.

It then represents those points as a signal. The signal can be approximated using multiple rotating components.

```text
Few components     → rough drawing
More components    → closer drawing
Many components    → detailed drawing
```

## Main features

- Reads an input image
- Finds the largest useful contour
- Converts 2D points into complex numbers
- Calculates discrete Fourier coefficients
- Uses positive and negative frequency components
- Reconstructs the image using epicycles
- Creates an animated drawing
- Exports a GIF
- Supports command-line options
- Validates input before processing
- Includes automated tests and CI support

## Run it

Install the requirements:

```bash
pip install -r requirements.txt
```

Then run:

```bash
python main.py --input image.jpg --output file.gif --terms 100 --frames 300 --fps 20
```

### What do these options mean?

- `--input` — image you want to redraw
- `--output` — GIF file to create
- `--terms` — number of Fourier components used for reconstruction
- `--frames` — number of animation frames
- `--fps` — animation speed

Using more terms usually gives a more detailed reconstruction, but requires more work to render.

## Project structure

```text
main.py       → Main application and animation
image.jpg     → Example input image
file.gif      → Example generated output
draw.py       → Earlier drawing experiment
working.py    → Development experiment
tests/        → Automated tests
```

## Technical terms explained

**Fourier analysis** — A mathematical method for breaking a complicated signal into simpler repeating waves. Here, the outline of an image is treated as a signal.

**Fourier series** — A way of representing a repeating signal as a combination of sine/cosine waves or equivalent rotating components.

**Fourier coefficient** — A number that describes how much a particular frequency contributes to the reconstructed signal.

**Frequency** — How quickly a component repeats. Different frequencies capture different levels of detail in the image outline.

**Complex number** — A number containing a real part and an imaginary part. Here it is a convenient way to represent an `(x, y)` point as one value.

**Contour** — The boundary or outline of an object in an image.

**Epicycle** — A circle whose centre moves around another point or circle. In this project, several rotating circles are combined to draw the shape.

**Reconstruction** — Building an approximation of the original image outline from the calculated Fourier components.

**NumPy** — A Python library for working efficiently with numerical data and arrays.

**Matplotlib** — A Python library used here to draw and animate the reconstruction.

**CLI (Command-Line Interface)** — Running and controlling the application by typing commands in a terminal.

**CI (Continuous Integration)** — Automated checks, such as tests, that run when code changes are pushed to GitHub.

## What does this project demonstrate?

This project combines mathematics and programming:

**Image → contour → numerical signal → Fourier analysis → rotating components → reconstructed drawing**

It demonstrates practical **Python, NumPy, mathematical modelling, Fourier analysis, visualization, animation, CLI design and testing** skills.

## Future improvements

- Faster FFT-based coefficient calculation
- Better contour cleaning and resampling
- Performance benchmarks
- Interactive controls for the number of terms
- More mathematical visual explanations
