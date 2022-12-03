import numpy as np
import jax.random as jrand
import jax.numpy as jnp
"""parameters for initializing mcTangent"""

# setup
c = 0.9
u = 1.0

t_max = 2.0
nt = 1000
dt = t_max/nt

x_max = 2.0
dx = u*dt/c
nx = np.ceil(x_max/dx)
dx = x_max/float(nx)
nx = int(nx)

mc_alpha = 1e6
noise_level = 0.02

num_epochs = int(100)
learning_rate = 1e-4
batch_size = 10
ns = 1
layers = 1
n_units = 5000
net_key = jrand.PRNGKey(0)

# initial condition
train_key = jrand.PRNGKey(1)
test_key = jrand.PRNGKey(2)
noise_key = jrand.PRNGKey(3)

# sample set size
num_train = 10
num_test = 10