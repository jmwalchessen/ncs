# Neural Conditional Simulation for Complex Spatial Processes

Source code for:
J. Walchessen, A. Zammit-Mangion, R. Huser, M. Kuusela (2024). Neural Conditional Simulation for Complex Spatial Processes. arXiv preprint arXiv:2505.24556.

## Getting Started
This repository contains the code and training details for each case study in "Neural Conditional Simulation for Complex Spatial Processes." **The package environment for this project is contained in requirements.txt**

## Training Details Common to Both  Case Studies

From experimentation, there are few hyperparameters that are sensitive to how well the U-Net approximates the true conditional score function. As such, the reason the hyperparameters vary between the U-Net and spatial process types is due to computational constraints, not training issues. The only training sensitivity we encountered pertains to amortization---the amortized variable ought to be sampled from a continuous, positive distribution.

## Hyperparameters

### Gaussian Process

#### Parameter U-Net
There are $10$ epochs per data draw and $10$ total draws. The batch size is $512$.

#### Proportion U-Net
There are $10$ epochs per data draw and $40$ total draws. The batch size is $2048$.

### Brown--Resnick Process
As in the Gaussian process case study, we did not discover any training sensitivities to any hyperparameters except with respect to amortization---the amortized variable ought to be sampled from a continuous, positive distribution. As such, we used the same hyperparameters as in the Gaussian process case study with some noted exceptions due to computational efficiency.

#### Parameter U-Net
The batch size, number of data draws, and epochs per data draw are the same as for the parameter U-Net in the Gaussian process case study.

#### Proportion U-Net
 The batch size, number of data draws, and epochs per data draw are also the same as for the proportion U-Net in the Gaussian process case study.

#### Small Conditioning Set U-Net
There are $10$ epochs per data draw and $40$ total data draws. The batch size is $2048$.

