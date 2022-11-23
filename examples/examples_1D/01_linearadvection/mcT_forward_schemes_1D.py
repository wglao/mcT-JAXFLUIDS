import jax.numpy as jnp
import numpy as np

def FTCS(un, velo, dt, dx):
    u = un - velo * dt / dx * (jnp.roll(un, -1) - jnp.roll(un, 1)) / 2
    return u

def MacCormack(un, velo, dt, dx):
    u1 = un - velo * dt / dx * (jnp.roll(un, 1) - un)
    u2 = un - velo * dt / dx * (u1 - jnp.roll(u1, -1))
    u = (u1 + u2) / 2
    return u