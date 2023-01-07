"""
*------------------------------------------------------------------------------*
* JAX-FLUIDS -                                                                 *
*                                                                              *
* A fully-differentiable CFD solver for compressible two-phase flows.          *
* Copyright (C) 2022  Deniz A. Bezgin, Aaron B. Buhendwa, Nikolaus A. Adams    *
*                                                                              *
* This program is free software: you can redistribute it and/or modify         *
* it under the terms of the GNU General Public License as published by         *
* the Free Software Foundation, either version 3 of the License, or            *
* (at your option) any later version.                                          *
*                                                                              *
* This program is distributed in the hope that it will be useful,              *
* but WITHOUT ANY WARRANTY; without even the implied warranty of               *
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the                *
* GNU General Public License for more details.                                 *
*                                                                              *
* You should have received a copy of the GNU General Public License            *
* along with this program.  If not, see <https://www.gnu.org/licenses/>.       *
*                                                                              *
*------------------------------------------------------------------------------*
*                                                                              *
* CONTACT                                                                      *
*                                                                              *
* deniz.bezgin@tum.de // aaron.buhendwa@tum.de // nikolaus.adams@tum.de        *
*                                                                              *
*------------------------------------------------------------------------------*
*                                                                              *
* Munich, April 15th, 2022                                                     *
*                                                                              *
*------------------------------------------------------------------------------*
"""

from typing import Callable, Dict
from functools import partial

import jax
import jax.numpy as jnp
from jax import jit

from jaxfluids.materials.material_manager import MaterialManager
from jaxfluids.solvers.riemann_solvers.riemann_solver import RiemannSolver
from jaxfluids.utilities import get_fluxes_xi

import haiku as hk

class mcTangentNN(RiemannSolver):
    """mcTangentNN Riemann Solver"""

    def __init__(self, material_manager: MaterialManager, signal_speed: Callable) -> None:
        super().__init__(material_manager, signal_speed)

    # def _reality_check(self, net_out: jnp.ndarray) -> jnp.ndarray:
    #     """
    #     Replace NaN or inf values from net outputs
    #     """
    #     checked = net_out
    #     return checked

    @partial(jit, static_argnums=(0, 7))
    def solve_riemann_problem_xi(self, primes_L: jnp.DeviceArray, primes_R: jnp.DeviceArray, 
        cons_L: jnp.DeviceArray, cons_R: jnp.DeviceArray, axis: int, 
        ml_parameters_dict: dict, ml_networks_dict: dict, **kwargs) -> jnp.DeviceArray:        
        params = ml_parameters_dict['riemann_solver']
        net    = ml_networks_dict['riemann_solver']

        assert type(net) == hk.Transformed, "Network architecture must be constructed using the Haiku Transform"

        # EVALUATE NEURAL NETWORK FOR TANGENT MANIFOLD
        net_out_L = jax.device_put(net.apply(params, primes_L, cons_L))
        net_out_R = jax.device_put(net.apply(params, primes_R, cons_R))

        # net_out_L = self._reality_check(net_out_L)
        # net_out_R = self._reality_check(net_out_R)

        fluxes_xi = 0.5*(jnp.reshape(net_out_L,cons_L.shape) + jnp.reshape(net_out_R,cons_R.shape))
        
        return fluxes_xi