# SAXS Demo

This repository contains a demonstration Jupyter notebook for analyzing 2D SAXS and WAXS data from Xenocs Xeuss 3 .edf files, and provides some useful tools.

It includes:
1. Importing 2D .edf data from Xenocs Xeuss 3.0 machine, including metadata
2. Create 2D images and 1D plots with correct parameters inferred from file metadata
3. create q-range and azimuthal angle  (theta) masks,
4. 1D integration
5. 1D fitting for an arbitrary _n_ gauss peaks.


## Contents
- `WAXS_Fitting_ExampleGH.ipynb`: Jupyter notebook showcasing the workflow.
- `saxs_utils_GH.py`: Utility functions for SAXS and WAXS data processing.
- `data/WAXS_0_64670.edf`: Example 2D SAXS dataset.

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/lsbesley/SAXSUtils.git
   




