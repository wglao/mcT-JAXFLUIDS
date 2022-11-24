import jax.numpy as jnp
import numpy as np

def FTCS(un, velo, dt, dx):
    u = un - velo*dt/dx * (jnp.roll(un, -1) - jnp.roll(un, 1)) / 2
    return u

def Upwind(un, velo, dt, dx):
    u = un - velo*dt/dx * (un - jnp.roll(un, 1))
    return u

def MacCormack(un, velo, dt, dx):
    u1 = un - velo*dt/dx * (jnp.roll(un, -1) - un)
    u = (un + u1)/2 - velo*dt/dx/2 * (u1 - jnp.roll(u1, 1))
    return u