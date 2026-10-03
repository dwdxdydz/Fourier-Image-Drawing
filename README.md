# Fourier Image Drawing

## What is this?

This project takes an image, finds its outline, and redraws that outline using rotating circles.

The circles are calculated using **Fourier analysis**.

In simple words:

```
Image
  ↓
Find outline
  ↓
Convert outline points into numbers
  ↓
Fourier calculation
  ↓
Create rotating circles
  ↓
Rebuild the outline
  ↓
Create GIF
```

## Why is this interesting?

A complicated shape can be represented using many simple rotating components.

- Fewer components → rough shape
- More components → more detail
- More components → more computation

The project is a practical example of mathematics being used in image processing.

## How to run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run an example:

```bash
python main.py --input image.jpg --output file.gif --terms 100 --frames 300 --fps 20
```

### Options

| Option | Meaning |
|---|---|
| `--input` | Image to process |
| `--output` | GIF to create |
| `--terms` | Number of Fourier components |
| `--frames` | Number of animation frames |
| `--fps` | Animation speed |

Run tests:

```bash
pytest -q
```

## Files

- `main.py` — main Fourier drawing program
- `draw.py` — drawing experiment
- `working.py` — development experiment
- `image.jpg` — sample input
- `file.gif` — sample output
- `tests/` — automated tests

## Technologies

- Python
- NumPy
- Pillow
- Matplotlib
- Pytest

## Important terms

**Contour:** the outline of an object in an image.

**Fourier analysis:** a mathematical method for representing a complex signal using simpler components.

**Fourier coefficient:** a number describing how much a particular component contributes to the final result.

**Epicycle:** a rotating circle used to help draw the final shape.

## What I learned

This project demonstrates:

- Python
- NumPy
- Image processing
- Fourier mathematics
- Data transformation
- Animation
- Command-line programs
- Automated testing
