from dedalus import public as d3
from dedalus.core.operators import lap
import numpy as np

dealias = 3/2
stop_sim_time = 3.9
timestepper = d3.RK222  # Use RK443 if higher accuracy is desired
max_timestep = 1e-2
dtype = np.float64

Nx, Ny = 512, 512  # Number of grid points
Lx, Ly = 2 * np.pi, 2 * np.pi  # Domain lengths

# Define the coordinate system
coords = d3.CartesianCoordinates('x', 'y')
dist = d3.Distributor(coords, dtype=dtype)

x_basis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=dealias)
y_basis = d3.RealFourier(coords['y'], size=Ny, bounds=(0, Ly), dealias=dealias)

# Create fields for s, sinx, and siny
s = dist.Field(name='s', bases=(x_basis, y_basis))
sinx = dist.Field(name='sinx', bases=(x_basis, y_basis))
siny = dist.Field(name='siny', bases=(x_basis, y_basis))

x, y = dist.local_grids(x_basis, y_basis)

sinx['g'] = np.sin(x)  # Dedalus field for sin(x)
siny['g'] = np.sin(y)  # Dedalus field for sin(y)

# Define velocity field u
u = dist.VectorField(coords, name='u', bases=(x_basis, y_basis))
u['g'][0] = siny['g']  # u_x = sin(y)
u['g'][1] = sinx['g']  # u_y = sin(x)

# Define the problem
D_val = 0.001  # Diffusion coefficient
ex, ey = coords.unit_vector_fields(dist)

problem = d3.IVP([s], namespace=locals())
problem.add_equation("dt(s) + u@grad(s) - 0.001*lap(s) = 0")

# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

# Initial conditions
s['g'] = np.sin(x) * np.sin(y)

# Snapshots
snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=0.1, max_writes=201)
snapshots.add_task(s, name='tracer')

# CFL
CFL = d3.CFL(solver, initial_dt=max_timestep, cadence=10, safety=0.1, threshold=0.1,
             max_change=1.2, min_change=0.8, max_dt=max_timestep)
CFL.add_velocity(u)

# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=10)
flow.add_property((s)**2, name='s1')

# Main loop
import logging
logger = logging.getLogger(__name__)
try:
    logger.info('Starting main loop')
    while solver.proceed:
        timestep = CFL.compute_timestep()
        solver.step(timestep)
        if (solver.iteration-1) % 10 == 0:
            max_s = np.sqrt(flow.max('s1'))
            logger.info('Iteration=%i, Time=%e, dt=%e, max(s)=%f' % (solver.iteration, solver.sim_time, timestep, max_s))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()