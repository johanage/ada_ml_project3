# testing the NN class using pytorch
from project3.simple_pde_neural_network import NN_torch
import torch
n_input = 10
t = torch.linspace(0,1,n_input).reshape(-1,1)
x = torch.linspace(0,1,n_input).reshape(-1,1)
X = torch.cat((torch.ones(x.shape), x, t), 1)

num_hidden_neurons = [20, 10, 10]
nntorch = NN_torch(n_input, num_hidden_neurons)
nntorch.forward(X)

