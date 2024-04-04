import numpy as np

def FHN(du, u, t, p):
    du = np.zeros_like(u)
    du[0] = 30*u[0] - 10*u[0]**3 - 30*u[1]
    du[1] = 7.5*u[0]
    return du

def spiral(t, u):
    true_A = np.array([[-0.1, -2.0, 0], 
                       [2.0, -0.1, 0], 
                       [0, 0, -0.3]])
    du = np.dot(true_A, u**3 + u)
    return du

def NDM(u, t, p):
    du = np.zeros_like(u)
    du[0] = 10 * (-u[0] + 1 / (1 + np.exp(16 * (u[1] - 0.5))))
    du[1] = 10 * (-u[1] + 1 / (1 + np.exp(16 * (u[0] - 0.5))))
    return du
