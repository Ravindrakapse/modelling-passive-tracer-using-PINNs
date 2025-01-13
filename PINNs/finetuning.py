import deepxde as dde
import numpy as np
import torch
import tensorflow as tf
from tensorflow.keras import Sequential
import os

dde.backend.set_default_backend("tensorflow")

if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

if dde.backend.backend_name == "pytorch":
    sin = dde.backend.pytorch.sin
    cos = dde.backend.pytorch.cos

# Domain parameters
Lx, Lz = 2*np.pi , 2*np.pi
D = 0.001

# domain and geometry 
geom = dde.geometry.Rectangle([0, 0], [Lx, Lz])
timedomain = dde.geometry.TimeDomain(0, 5)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)


#  PDE system
def pde(x, y):
    """
    x: (x, z, t) -> input tensor containing spatial and temporal coordinates.
    y: (s) -> output tensor containing solution field:
        - `s`: Passive tracer concentration.
    """
    
   
    s_t = dde.grad.jacobian(y, x, i=0, j=2)
    s_x = dde.grad.jacobian(y, x, i=0, j=0)
    s_z = dde.grad.jacobian(y, x, i=0, j=1)
    s_xx = dde.grad.hessian(y, x, i=0, j=0)
    s_zz = dde.grad.hessian(y, x, i=1, j=1)


    # Advection-diffusion equation for tracer
    tracer_eq = s_t + (sin(x[:, 1:2])) * s_x + (sin(x[:, 0:1])) * s_z - D * (s_xx + s_zz)

    return  tracer_eq

# Boundary conditions
def boundary_x_l(x, on_boundary):
    return on_boundary and np.isclose(x[0], 0)

def boundary_x_r(x, on_boundary):
    return on_boundary and np.isclose(x[0], Lx)

def boundary_y_b(x, on_boundary):
    return on_boundary and np.isclose(x[1], 0)

def boundary_y_u(x, on_boundary):
    return on_boundary and np.isclose(x[1], Lz)

def initial(x):
    return (np.sin(2*np.pi*x[:, 0:1])) * (np.sin(2*np.pi*x[:, 1:2]))


# periodic boundary conditions
bc_x_l = dde.PeriodicBC(geomtime, 0, boundary_x_l)
bc_x_r = dde.PeriodicBC(geomtime, 0, boundary_x_r)
bc_y_b = dde.PeriodicBC(geomtime, 1, boundary_y_b)
bc_y_u = dde.PeriodicBC(geomtime, 1, boundary_y_u)


ic = dde.IC(geomtime, initial, lambda _, on_initial: on_initial)

data = dde.data.TimePDE(
    geomtime,
    pde,
    [ic,bc_x_l,bc_x_r,bc_y_b,bc_y_u],
    num_domain= 30000,
    num_boundary=6000,
    num_initial= 6000,
    #train_distribution="uniform",
    num_test= None
)

# neural network model
net = dde.maps.FNN([3] + [100] * 10 + [1], "sin", "Glorot normal")

# Compile the model
model = dde.Model(data, net)

# Load the PyTorch checkpoint ( pretrained model)
checkpoint = torch.load('../trained_model')

# Ensure compatibility
net.load_state_dict(checkpoint['model_state_dict'])


model.compile("L-BFGS-B")

losshistory, train_state = model.train()

dde.saveplot(losshistory, train_state, issave=True, isplot=True)