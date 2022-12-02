# ada_ml_project3
A simple project on population growth comparing numerical, NN and stochastic NN solutions of the ODE or SDE respectively.

# Requirements
- download ada_ml_project2 and add repo to PYTHONPATH

# Dependencies
- numpy, autograd, torch, torchsde, matplotlib

# Scripts
- ode_nn.py : test script for solving the ODE both numerically and using a simple single layer NN
- ode_setup.py: to set up the ODE problem from the NN
- sde_solver.py: solving the SDE version of the ODE problem both using implementation and torchsde (for comparison)
