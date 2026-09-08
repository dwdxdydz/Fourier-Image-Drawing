# Fourier Image Drawing

Reconstructs the largest contour in an image using a discrete Fourier series and visualizes the reconstruction as an epicycle animation.

## What it demonstrates

- OpenCV contour extraction
- Discrete Fourier coefficients
- Complex-number representation of 2D coordinates
- Fourier approximation with positive/negative frequencies
- Matplotlib animation and GIF export

## Run

```bash
pip install -r requirements.txt
python main.py --input image.jpg --output file.gif --terms 100 --frames 300 --fps 20
```

Increase `--terms` for a closer reconstruction at the cost of more animation objects. `--frames` controls smoothness and `--fps` controls playback speed.

## Project structure

- `main.py` — production CLI and reconstruction pipeline
- `draw.py` — original drawing experiment
- `image.jpg` — sample input
- `file.gif` — sample output

## Notes

The implementation centers the contour, normalizes the animation bounds from the selected coefficients, and validates CLI inputs instead of relying on hardcoded plotting limits.
