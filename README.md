# 🌀 Fourier Image Drawing

Reconstructs an image contour using a **discrete Fourier series** and visualizes the reconstruction as an epicycle animation.

## What it demonstrates

- Image contour extraction
- Complex-number representation of 2D coordinates
- Discrete Fourier coefficients
- Positive and negative frequency components
- Fourier approximation and reconstruction
- Matplotlib animation and GIF export
- Command-line configuration and input validation

## How it works

```text
Image
  ↓
Largest contour
  ↓
Center + normalize coordinates
  ↓
Complex signal
  ↓
Fourier coefficients
  ↓
Select frequency terms
  ↓
Epicycle reconstruction
  ↓
Animated drawing / GIF
```

## Run

```bash
pip install -r requirements.txt
python main.py --input image.jpg --output file.gif --terms 100 --frames 300 --fps 20
```

Increase `--terms` for a closer reconstruction at the cost of more animation objects. `--frames` controls animation smoothness and `--fps` controls playback speed.

## Project structure

- `main.py` — production CLI and reconstruction pipeline
- `draw.py` — original drawing experiment
- `working.py` — development experiment
- `image.jpg` — sample input
- `file.gif` — sample output
- `tests/` — automated checks

## Engineering highlights

The production renderer validates inputs, centers the contour, handles positive/negative frequencies, dynamically scales the animation bounds, and supports headless GIF rendering for CI environments.

## Portfolio value

Demonstrates **Python, NumPy, computer vision basics, mathematical modelling, Fourier analysis, visualization, CLI design, testing, and performance-aware reconstruction**.

## Future improvements

- FFT-based coefficient calculation
- Better contour preprocessing and resampling
- Reconstruction performance benchmarks
- Interactive parameter controls
- Mathematical explanation with visual examples
