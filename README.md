# 🌀 Fourier Image Drawing

## What is this project?

This project takes an image and **redraws it using mathematics**.

The interesting part is that the application does not simply copy the pixels. It looks at the outline of the object, turns that outline into mathematical information, and then rebuilds the shape using rotating circles.

The final result looks like a set of circles rotating around each other until they draw the original image.

## Simple example

```text
Input image
    ↓
Find the outline
    ↓
Turn the outline into coordinates
    ↓
Convert coordinates into a mathematical signal
    ↓
Break the signal into frequencies
    ↓
Rebuild the shape with rotating circles
    ↓
Animated drawing
```

## What are the rotating circles?

Each rotating circle represents one part of the mathematical description of the image.

A small number of circles gives a rough drawing:

```text
Few circles → rough shape
```

More circles give a closer drawing:

```text
Many circles → more accurate shape
```

This is based on **Fourier analysis**, a mathematical technique used to break a complicated signal into simpler repeating parts.

## What does the application do?

1. Reads an image.
2. Finds the largest outline in the image.
3. Cleans and centers the outline.
4. Represents the outline using complex numbers.
5. Calculates Fourier coefficients.
6. Selects the required number of frequency components.
7. Uses rotating circles to reconstruct the outline.
8. Creates an animation of the drawing.
9. Can save the animation as a GIF.

## Run it

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python main.py --input image.jpg --output file.gif --terms 100 --frames 300 --fps 20
```

### What do the options mean?

- `--input` → image to draw
- `--output` → GIF file to create
- `--terms` → number of mathematical components used for the drawing
- `--frames` → number of animation frames
- `--fps` → animation playback speed

For example, increasing `--terms` generally makes the drawing more accurate, but also makes the animation more expensive to calculate.

## Project structure

```text
main.py       → Main application and Fourier drawing pipeline
draw.py       → Original drawing experiment
working.py    → Development experiment
image.jpg     → Example input image
file.gif      → Example generated animation
tests/        → Automated tests
```

## Main technologies

- **Python** — application logic
- **NumPy** — mathematical calculations
- **Pillow** — image handling
- **Matplotlib** — animation and visualization
- **Fourier analysis** — mathematical reconstruction

## Why this project is interesting

A normal image contains thousands or millions of pixels. This project takes a completely different approach: it describes the **shape of the image using mathematical waves** and then rebuilds it.

It is a practical demonstration of how mathematics, programming and visualization can work together.

## What I learned

This project helped demonstrate:

- Image processing
- Coordinate and contour handling
- Complex numbers
- Fourier analysis
- Data transformation
- Animation
- Command-line application design
- Automated testing

## Current limitation

The application mainly works with the main/large contour of an image. It is intended as a mathematical visualization and learning project rather than a complete image editor.

## Future improvements

- Faster Fourier calculations using FFT.
- Better handling of multiple contours.
- Better image preprocessing.
- Performance benchmarks.
- Interactive controls for the number of circles and animation speed.
- More examples showing how the mathematics changes the drawing.
