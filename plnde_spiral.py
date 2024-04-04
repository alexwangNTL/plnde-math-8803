import numpy as np
import torch
from torchdiffeq import odeint_adjoint as odeint
from scipy.optimize import minimize
import scipy.stats as stats
import bson
import random

from scipy.integrate import solve_ivp

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)

from example_dynamics import *
from meshgrid import *

# ********************************************************* Class Definitions ********************************************************** #

import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint

class FeedforwardNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(FeedforwardNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 17),
            nn.SiLU(),  # Swish activation function is called SiLU in PyTorch
            nn.Linear(17, 23),
            nn.SiLU(),
            nn.Linear(23, 17),
            nn.SiLU(),
            nn.Linear(17, output_size),
        )
        
    def forward(self, x):
        return self.network(x)

class NeuralODE(nn.Module):
    def __init__(self, func):
        super(NeuralODE, self).__init__()
        self.func = func
    
    def forward(self, t, u0):
        return odeint(self.func, u0, t, method='dopri5')
    
# ********************************************************* Helper Functions ****************************************************************** #

def train(model, optimizer, t_span, loss_fn, epochs=300, learning_rate=0.01):
    for epoch in range(epochs):
        optimizer.zero_grad()
        p = odeint(model, y0, t_span)
        loss = loss_fn(p)
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item()}")

# ********************************************************* Model ****************************************************************** #

# Hyperparameters
T = 1001        # time stamps rounded to the nearest 1ms
L = 3           # number of latents assumed by the model
L_true = 3      # true number of latents
D = 150         # number of observations
N = 343         # number of trials
dt = 0.001      # 1-ms bin
k = 1.          # scale parameter for prior initial value distribution covariance
tspan = (0.0, 40.0)  # 40 seconds long
trange = np.linspace(tspan[0], tspan[1], T)
u0 = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)

L = 3
D = 150
t = torch.tensor(trange, dtype=torch.float32)

f = FeedforwardNN(L, L)
nn_ode = NeuralODE(f)

# For the mapping from latent space to observations
_logλ_true = nn.Linear(L_true, D)
_logλ = nn.Linear(L, D)

# ***************************************************** Generate data ************************************************************** #

# Generate loading matrix with properties similar to those described
C = (np.random.rand(D, L_true) + 2) * np.sign(np.random.randn(D, L_true))
# For an alternative high mean firing rates version, you could uncomment the following line
# C = (np.random.rand(D, L_true) + 8) * np.sign(np.random.randn(D, L_true))
params_λ_true = np.hstack([C.flatten(), np.zeros(D)])

# Convert C to a PyTorch tensor and ensure it's the correct data type
C_tensor = torch.tensor(C, dtype=torch.float32)

# The biases are being reset to zero, so we create a tensor of zeros with length D
zeros_bias = torch.zeros(D, dtype=torch.float32)

with torch.no_grad():  # Disable gradient tracking for this operation
    _logλ_true.weight.copy_(C_tensor)
    _logλ_true.bias.copy_(zeros_bias)

# Generate initial values for training dataset
x = np.linspace(-0.5, 0.5, 7); y = np.linspace(-0.5, 0.5, 7); z = np.linspace(-0.5, 0.5, 7)
X, Y, Z = meshgrid_3d(x, y, z)
u0s = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T

prob_true = solve_ivp(spiral, tspan, u0)
z_true_mat = np.empty((len(trange), 3, N))  # Placeholder for the solutions
for i in range(N):
    u0 = u0s[i, :]
    sol = solve_ivp(spiral, tspan, u0, method='RK45', t_eval=trange)
    z_true_mat[:, :, i] = sol.y.T

# Generate training dataset spikes from discretized intensity
## Initialize the spike times array
spike_times_disc = np.zeros((D, T, N))

## Generate spikes
for i in range(N):
    _temp_tensor = torch.tensor(z_true_mat[:, :, i], dtype=torch.float32)
    rate = dt * np.exp(_logλ_true(_temp_tensor).detach())
    spike_times_disc[:, :, i] = (np.random.poisson(rate) > 0).T

## Convert to boolean if not already
spike_times_disc = spike_times_disc.astype(bool)

## Rearrange dimensions to get spikes in the format [neurons x trials x timebins]
spikes = np.transpose(spike_times_disc, (0, 2, 1))

# Generate initial values for test dataset
u0s_test = np.random.rand(L_true, N) - 0.5

z_true_test = np.empty((L_true, len(trange), N))  # Preallocate array

for i in range(N):
    sol = solve_ivp(spiral, tspan, u0s_test[:, i], t_eval=trange)
    z_true_test[:, :, i] = sol.y

spike_times_test = np.zeros((D, T, N), dtype=bool)

for i in range(N):
    _temp_tensor = torch.tensor(z_true_test[:, :, i].T, dtype=torch.float32)
    rate = dt * np.exp(_logλ_true(_temp_tensor).detach())
    spike_times_test[:, :, i] = (np.random.poisson(rate) > 0).T

spikes_test = np.transpose(spike_times_test, axes=(0, 2, 1))

# ********************************************************** Loss ****************************************************************** #

# Initialize model parameters
# θ = torch.cat([
#     torch.nn.init.xavier_uniform_(torch.empty(length_nn_ode_p, dtype=torch.float32)),
#     torch.randn(D*(L_true+1), dtype=torch.float32),
#     torch.randn(L_true*N, dtype=torch.float32),
#     -10.0 * torch.ones(L_true*N, dtype=torch.float32)
# ])

def loss_nn_ode(p):
    # Reshape and apply transformations to parameters as needed
    u0_m = p[-L*N-L*N:-L*N].reshape(L, N)
    u0_s = torch.clamp(p[-L*N:].reshape(L, N), -1e8, 0)
    u0s = u0_m + torch.exp(u0_s) * torch.randn_like(u0_s)
    # Assuming nn_ode and logλ are callable and return tensors
    z_hat = nn_ode(u0s, p)  # You need to adapt this call to match your neural ODE implementation
    λ_hat = torch.exp(_logλ(z_hat.view(-1, L), p)).view(D, -1, N)
    Nlogλ = spikes * torch.log(dt * λ_hat + torch.sqrt(torch.finfo(torch.float32).eps))
    kld = 0.5 * (N * L * torch.log(torch.tensor(k)) - torch.sum(2.0 * u0_s) - N * L + torch.sum(torch.exp(2.0 * u0_s)) / k + torch.sum(u0_m**2) / k)
    loss = (torch.sum(dt * λ_hat - Nlogλ) + kld) / N
    return loss


# def cb(p, l):
#     print(l.item())  # Assuming l is a tensor
#     return False

# ******************************************************** Training **************************************************************** #

# ******************************************************** Testing ***************************************************************** #
