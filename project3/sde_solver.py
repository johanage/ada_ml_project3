# implementation of the euler maruyama sheme
import torch
import torchsde
import numpy as np
import matplotlib.pyplot as plt
from torchsde import BrownianInterval

"""

Following the example given by torchsde to test 
the torchsde-implemented sde solver.

"""


batch_size, state_size, brownian_size = 5, 1, 5#100
t_size = 100
ts = torch.linspace(0, 10, t_size)
y0 = torch.full(size=(batch_size, state_size), fill_value=0.1)
bm = BrownianInterval(t0=ts[0], 
                      t1=ts[-1], 
                      size=(batch_size, brownian_size))#,
                      #device='')
class SDE(torch.nn.Module):
    def __init__(self, state_size, batch_size, brownian_size, 
                 r, alpha, y0):
        super().__init__()
        self.state_size = state_size
        self.batch_size = batch_size
        self.brownian_size = brownian_size
        # defines the first term on the lhs of the Itô formula
        self.mu = torch.nn.Linear(self.state_size, 
                                  self.state_size)
        # defines the second term on the lhs of the Itô formula
        self.sigma = torch.nn.Linear(self.state_size,
                                     self.state_size * self.brownian_size)
        self.noise_type = "diagonal"
        self.sde_type = "ito"
        self.alpha = torch.nn.Parameter( torch.tensor(alpha), requires_grad = True)
        self.r = torch.nn.Parameter( torch.tensor(r), requires_grad = True)
    # Drift
    def f(self, t, y):
        return self.r*y  # shape (batch_size, state_size)

    # Diffusion
    def g(self, t, y):
        return self.alpha*y
    
    def solve(self,t, bm):
        self.sol = self.y0*torch.exp( (self.r - .5*self.alpha*self.alpha)*t + self.alpha*self.bm )
# define the sde solver object
sde = SDE(state_size, batch_size, brownian_size, 0.5, 0.1, y0)

with torch.no_grad():
    ys = torchsde.sdeint(sde, y0, ts, bm = bm, method='euler')  # (t_size, batch_size, state_size) = (100, 3, 1).

for i in range(ys.shape[1]):
    for j in range(ys.shape[2]):
        plt.plot(ts, ys[:, i, j])#, xlabel='$t$', ylabel='$Y_t$')

plt.show()
