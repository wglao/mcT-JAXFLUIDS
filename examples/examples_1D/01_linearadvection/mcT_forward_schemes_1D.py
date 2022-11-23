import jax.numpy as jnp
import numpy as np

def FTCS(un, velo, dt, dx):
    u = un - velo * dt / dx * (jnp.roll(un, -1) - jnp.roll(un, 1)) / 2
    return u

def MacCormack(un, velo, dt, dx):
    u = un - velo*dt/dx * (jnp.roll(un, -1) - jnp.roll(un, 1) / 2) \
           + (velo*dt/dx)**2 * (jnp.roll(un, -1) - 2*un + jnp.roll(un, 1) / 2)
    return u