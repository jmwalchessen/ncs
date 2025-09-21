# Neural Conditional Simulation for Complex Spatial Processes

Source code for:
J. Walchessen, A. Zammit-Mangion, R. Huser, M. Kuusela (2024). Neural Conditional Simulation for Complex Spatial Processes. arXiv preprint arXiv:2505.24556.

## Getting Started
This repository contains the code and training details for each case study in "Neural Conditional Simulation for Complex Spatial Processes." This document information about the training details and the contents of the repository. To quickly understand what a folder pertains to, please scroll down to find the information. **The package environment for this project is contained in requirements.txt**

## Training Details Common to Both  Case Studies

From experimentation, there are few hyperparameters that are sensitive to how well the U-Net approximates the true conditional score function. As such, the reason the hyperparameters vary between the U-Net and spatial process types is due to computational constraints, not training issues. The only training sensitivity we encountered pertains to amortization---the amortized variable ought to be sampled from a continuous, positive distribution.

