![logo](logo.png)
This Python library provides tools for fitting and visualizing Mueller matrix models for **SCExAO VAMPIRES** and the **SCExAO CHARIS spectropolarimetric mode**. In the future it will be updated for compatibility with other instruments.

You can fit either:
- A **physically motivated model** 
- A **direct fit** of retardances and diattenuations.

---

## Features
- Create Mueller matrix models with `pyMuellerMat` `physical_models` branch
- Fit using:
  - `emcee`
  - `scipy.optimize.minimize`
  - `scipy.optimize.least_squares`
- Calibration data handling via CSVs
- Save fitting results as `.txt` files
- Tutorial Jupyter notebooks for both VAMPIRES and CHARIS (see `VAMPIRES/` and `CHARIS/` folders)

---

## Requirements
- Python ≥ 3.11

---

## Installation
1. Create and activate a new virtual Python or Conda environment.

2. Clone the repository into your desired directory:
```bash
git clone https://github.com/UCSB-Exoplanet-Polarimetry-Lab/pyPolCal.git
cd pyPolCal
```
3. Install the package and its dependencies. Editable mode (`-e`) is recommended, as it allows you to modify the source code — this is often necessary for this package:
```bash
pip install -e .
```

---

## Repository Structure

```
pyPolCal/
│── pyPolCal/       # Package root for pyPolCal
   │── constants.py   # Hard coded constants
   │── csv_tools.py   # Tools for writing and reading CSV files with FITS info
   │── fitting.py     # Tools for fitting the Mueller matrix model
   │── instruments.py     # Wrapper functions for minimize_system_mueller_matrix to fit with one function
   │── instruments_jax.py     # Jax-compatible functions for MCMC
   │── mcmc_helper_funcs_jax.py     # More MCMC helpers
   │── on_sky.py     # CSV/fitting tools for unpolarized or polarized standard star calibration
   │── plotting.py     # Plotting functions
   │── utils.py     # Helper functions
   │── CHARIS/      # Tutorial notebooks for CHARIS fitting and source code for McIntosh+ 2025 SPIE polcal proceeding
   │── VAMPIRES/    # Source code for Zhang+ 2024 SPIE polcal proceeding
   

  


  
    

│── pyproject.toml    # Installation file
│── README.md
```
