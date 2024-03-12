import torch
import torch.nn as nn
# from torchdiffeq import odeint_adjoint as odeint
from torchdyn.core import NeuralODE
# - docs https://torchdyn.readthedocs.io/en/latest/source/torchdyn.core.html?highlight=NeuralODE#torchdyn.core.NeuralDE

from example_dynamics import *

# ********************************************************* Class Definitions ********************************************************** #

# Define the neural network architecture
class NeuralNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NeuralNet, self).__init__()

        layers = [

            nn.Linear(input_dim, 17),
            nn.SiLU(),
            nn.Linear(17, 23),
            nn.SiLU(),
            nn.Linear(23, 17),
            nn.SiLU(),
            nn.Linear(17, output_dim)
        ]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# Define the ODE function to be used with the neural network
class ODEFunc(nn.Module):
    def __init__(self, neural_net):
        super(ODEFunc, self).__init__()
        self.neural_net = neural_net

    def forward(self, t, x):
        return self.neural_net(x)
    
class Mapping(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Mapping, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.linear(x)

# ********************************************************* Model ****************************************************************** #

# Hyperparameters
T = 1001        # time stamps rounded to the nearest 1ms
L = 3           # number of latents assumed by the model
L_true = 3      # true number of latents
D = 150         # number of observations
N = 343         # number of trials
dt = 0.001      # 1-ms bin
κ = 1.          # scale parameter for prior initial value distribution covariance
tspan = (0.0, 40.0)
trange = torch.linspace(0.0, 40.0, steps=T)  # 40 seconds long, 1001 time points
u0 = torch.FloatTensor([1.0, 1.0, 1.0])

f = NeuralNet(input_dim=L, output_dim=L)
# odefunc = ODEFunc(f)

_nn_ode = NeuralODE(f)

solution = #odeint(odefunc, u0, trange, rtol=1e-6, atol=1e-6)# , method='dopri5', rtol=1e-6, atol=1e-6)
print("soln : ", solution)

# Mappings from latent space to observations
_logλ_true = Mapping(input_dim=L_true, output_dim=D)
_logλ = Mapping(input_dim=L, output_dim=D)

# TODO are these correct equivalencies
nn_ode = lambda u : _nn_ode(u, trange)
logλ = lambda u : _logλ(u)
