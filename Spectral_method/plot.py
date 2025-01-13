import numpy as np
import matplotlib.pyplot as plt
import h5py
from matplotlib.animation import FuncAnimation

# Setup
nelx, nely = 512, 512
timesteps = 200
Lx, Lz = 2*np.pi, 2*np.pi  # Assuming domain sizes
x = np.linspace(0, Lx, nelx+1)
y = np.linspace(0, Lz, nely+1)
xx, yy = np.meshgrid(x, y)

# Load tracer data
file_path_pinns = '../snapshots/snapshots_s1.h5'
with h5py.File(file_path_pinns, 'r') as f:
    data_set_spectral = f['tracer'][:]

# Create figure for plotting
fig, ax = plt.subplots()
cax = ax.pcolor(xx, yy, data_set_spectral[0], cmap='RdBu_r')
fig.colorbar(cax)

# Function to update the frame in the animation
def update(frame):
    cax.set_array(data_set_spectral[frame].flatten())
    ax.set_title(f'Time step: {frame}')
    return cax,

# Create animation
anim = FuncAnimation(fig, update, frames=timesteps, repeat=False)

# Save animation as GIF
anim.save('tracer_animation_spectral.gif')

plt.show()