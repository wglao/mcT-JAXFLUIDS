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

import jax
import jax.numpy as jnp

from jaxfluids.materials.material_manager import MaterialManager
from jaxfluids.solvers.riemann_solvers.riemann_solver import RiemannSolver
from jaxfluids.utilities import get_fluxes_xi

import haiku as hk

class mcTangentNN(RiemannSolver):
    """mcTangentNN Riemann Solver"""

    def __init__(self, material_manager: MaterialManager, signal_speed: Callable) -> None:
        super().__init__(material_manager, signal_speed)

    def solve_riemann_problem_xi(self, primes_L: jnp.DeviceArray, primes_R: jnp.DeviceArray, 
        cons_L: jnp.DeviceArray, cons_R: jnp.DeviceArray, axis: int, 
        ml_parameters_dict: dict, ml_networks_dict: dict, **kwargs) -> jnp.DeviceArray:        
        params = ml_parameters_dict['riemann_solver']
        net    = ml_networks_dict['riemann_solver']

        assert type(net) == hk.Transformed, "Network architecture must be constructed using the Haiku Transform"

        # EVALUATE NEURAL NETWORK FOR TANGENT MANIFOLD
        try:
            primes = jnp.mean(jnp.array([primes_L, primes_R]),axis=0)
            fluxes_xi = net.apply(params, primes)
            return fluxes_xi
        except Exception as e:
            print(e)
