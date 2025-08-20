# vampires_calibration

This Python library provides tools for fitting and visualizing Mueller matrix models for **SCExAO VAMPIRES** and the **SCExAO CHARIS spectropolarimetric mode**.  

You can fit either:
- A **physically motivated model** 
- A **direct fit** of retardances and diattenuations.

## Features
- Create Mueller matrix models with `pyMuellerMat` physical_models branch
- Fit using:
  - `emcee`
  - `scipy.optimize.minimize`
  - `scipy.optimize.least_squares`
- Calibration data handling via CSVs
- Save fitting results as `.txt` files
- Tutorial Jupyter notebooks for both VAMPIRES and CHARIS (see `VAMPIRES/` and `CHARIS/` folders)

---