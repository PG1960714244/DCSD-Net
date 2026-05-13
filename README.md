# DCSD-Net

This repository provides the PyTorch implementations of the core components introduced in the paper:

**DCSD-Net: A feature interaction and reconstruction collaborative network for steel surface defect detection**

The released code is intended to facilitate understanding, reproduction, and integration of the proposed modules into existing detection frameworks.

## Overview

DCSD-Net is designed for steel surface defect detection, with a focus on weak-texture defects, small-scale defects, irregular defect morphologies, and efficient feature reconstruction.

The main components released in this repository include:

- **CDFI**: Cross-dimensional feature interaction module
- **SRF-Conv**: Split-reconstruction fusion convolution
- **DySample**: Dynamic content-aware upsampling module
- **Visualization figures** used in the manuscript
- **Analysis tools** for confusion matrix and heatmap visualization

The overall detection framework is built upon the DEIM-D-FINE detection framework. For the baseline framework, please refer to the official DEIM project page:

https://www.shihuahuang.cn/DEIM/

## Repository Structure

```text
DCSD-Net/
├── baseline/
│   └── README.md
├── figures/
│   ├── Fig1.png
│   ├── Fig2.png
│   ├── Fig3.png
│   └── Fig4.png
├── modules/
│   ├── CDFI/
│   ├── DySample/
│   └── SRF-Conv/
├── tools/
│   ├── confusion matrix.py
│   └── heatmap.py
└── README.md
