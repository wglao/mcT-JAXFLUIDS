"""
Forward numerical schemes
currently only implemented for 1D advection
"""

import jax.numpy as jnp
import numpy as np

def FTCS(un: jnp.ndarray, velo: float, dt: float, dx: float):
    """
    Forward in Time, Center in Space
    Unconditionally unstable for hyperbolic PDE

    ----- inputs -----\n
    :param un: starting state
    :param velo: advecting velocity
    :param dt: time step size
    :param dx: space step size

    ----- returns -----\n
    :return u: predicted state after 1 dt
    """
    u = un - velo*dt/dx * (jnp.roll(un, -1) - jnp.roll(un, 1)) / 2
    return u

def Upwind(un: jnp.ndarray, velo: float, dt: float, dx: float):
    """
    Upwind 1st Order (Forward in Time, Backward in Space)
    1st order accurate

    ----- inputs -----\n
    :param un: starting state
    :param velo: advecting velocity
    :param dt: time step size
    :param dx: space step size

    ----- returns -----\n
    :return u: predicted state after 1 dt
    """
    u = un - velo*dt/dx * (un - jnp.roll(un, 1))
    return u

def MacCormack(un: jnp.ndarray, velo: float, dt: float, dx: float):
    """
    MacCormack (modified 2nd Order Lax Wendroff)
    2nd order accurate

    ----- inputs -----\n
    :param un: starting state
    :param velo: advecting velocity
    :param dt: time step size
    :param dx: space step size

    ----- returns -----\n
    :return u: predicted state after 1 dt
    """
    u1 = un - velo*dt/dx * (jnp.roll(un, -1) - un)
    u = (un + u1)/2 - velo*dt/dx/2 * (u1 - jnp.roll(u1, 1))
    return u