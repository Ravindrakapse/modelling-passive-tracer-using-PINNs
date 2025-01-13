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
timedomain = dde.geometry.TimeDomain(0, 30)
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
    return (np.sin(x[:, 0:1])) * (np.sin(x[:, 1:2]))




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
    num_domain=100000,
    num_boundary=20000,
    num_initial=9000,
    #train_distribution="uniform",
    num_test= None
)


os.makedirs("trained_model", exist_ok=True)

# neural network model
net = dde.maps.FNN([3] + [100] * 10 + [1], "sin", "Glorot normal")

# Compile the model
model = dde.Model(data, net)
model.compile('adam', lr = 0.001)
losshistory, train_state = model.train(epochs=15000)
model.compile("L-BFGS-B")
checkpointer = dde.callbacks.ModelCheckpoint("trained_model/model.ckpt", verbose=1, save_better_only=True)
# Train the model
losshistory, train_state = model.train(callbacks=[checkpointer])

dde.saveplot(losshistory, train_state, issave=True, isplot=True)



# Plot and saving the data

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import deepxde as dde
from scipy.interpolate import griddata
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import axes3d
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import h5py


nelx = 511
nely = 511
timesteps = 200
x = np.linspace(0,Lx,nelx+1)
y = np.linspace(0,Lz,nely+1)
t = np.linspace(0,20,timesteps)
delta_t = t[1] - t[0]
xx,yy = np.meshgrid(x,y)


x_ = np.zeros(shape = ((nelx+1) * (nely+1),))
y_ = np.zeros(shape = ((nelx+1) * (nely+1),))
for c1,ycor in enumerate(y):
        for c2,xcor in enumerate(x):
            x_[c1*(nelx+1) + c2] = xcor
            y_[c1*(nelx+1) + c2] = ycor
Ts = []

with h5py.File('tracer_pinns.h5', 'w') as f:
    # Create a dataset for tracer data
    max_shape = (None, nelx+1, nely+1)  # None indicates that the dataset can grow indefinitely
    dset = f.create_dataset('tracer', shape=(0, nelx+1, nely+1), maxshape=max_shape)

    for time in t:
        t_ = np.ones((nelx+1) * (nely+1),) * (time)
        X = np.column_stack((x_, y_))
        X = np.column_stack((X, t_))
        T = model.predict(X)
        T = T.reshape(T.shape[0],)
        T = T.reshape(nelx+1, nely+1)
        Ts.append(T)
        
        # Append new data to the dataset
        dset.resize(dset.shape[0] + 1, axis=0)
        dset[-1] = T

for time in t:
        t_ = np.ones((nelx+1) * (nely+1),) * (time)
        X = np.column_stack((x_,y_))
        X = np.column_stack((X,t_))
        T = model.predict(X)
        T = T.reshape(T.shape[0],)
        T = T.reshape(nelx+1,nely+1)
        Ts.append(T)


def plotheatmap(T,time):
      # Clear the current plot figure
        plt.clf()
        plt.title(f"Tracer at t = {time*delta_t}")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.pcolor(xx, yy, T,cmap = 'RdBu_r')
        plt.colorbar()
        return plt

def animate(k):
        plotheatmap(Ts[k], k)

anim = animation.FuncAnimation(plt.figure(), animate, interval=1, frames=len(t), repeat=False)

anim.save("tracer_pinns.gif")

fig,ax0=plt.subplots(1,1)
c=ax0.pcolor(xx,yy,Ts[200],cmap='RdBu_r')
fig.colorbar(c,ax=ax0)
plt.show()