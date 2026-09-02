# Fourier Image Drawing

A computer-vision experiment that converts image contours into complex Fourier coefficients and reconstructs the shape with rotating vectors (epicycles).

## What it demonstrates

1. Extract an image contour using OpenCV.
2. Center the contour coordinates.
3. Compute discrete Fourier coefficients for positive and negative frequencies.
4. Reconstruct the path by rotating coefficient vectors over time.
5. Export the reconstruction as an animated GIF.

## Run

Install the Python dependencies and provide an input image path in `main.py`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

## Tech Stack

**Python · OpenCV · NumPy · Matplotlib · SymPy · Fourier Analysis**

## Resume Description

**Fourier Image Drawing | Python, OpenCV, NumPy, Fourier Analysis**

Built a computer-vision visualization pipeline that extracts contours from images, transforms 2D coordinates into complex Fourier coefficients, and reconstructs the original shape using rotating-vector epicycles. Added animation and symbolic Fourier-series generation to make the mathematical reconstruction observable.
