#*------------------------------------------------------------------------------*
#* JAX-FLUIDS -                                                                 *
#*                                                                              *
#* A fully-differentiable CFD solver for compressible two-phase flows.          *
#* Copyright (C) 2022  Deniz A. Bezgin, Aaron B. Buhendwa, Nikolaus A. Adams    *
#*                                                                              *
#* This program is free software: you can redistribute it and/or modify         *
#* it under the terms of the GNU General Public License as published by         *
#* the Free Software Foundation, either version 3 of the License, or            *
#* (at your option) any later version.                                          *
#*                                                                              *
#* This program is distributed in the hope that it will be useful,              *
#* but WITHOUT ANY WARRANTY; without even the implied warranty of               *
#* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the                *
#* GNU General Public License for more details.                                 *
#*                                                                              *
#* You should have received a copy of the GNU General Public License            *
#* along with this program.  If not, see <https://www.gnu.org/licenses/>.       *
#*                                                                              *
#*------------------------------------------------------------------------------*
#*                                                                              *
#* CONTACT                                                                      *
#*                                                                              *
#* deniz.bezgin@tum.de // aaron.buhendwa@tum.de // nikolaus.adams@tum.de        *
#*                                                                              *
#*------------------------------------------------------------------------------*
#*                                                                              *
#* Munich, April 15th, 2022                                                     *
#*                                                                              *
#*------------------------------------------------------------------------------*

from typing import List

import jax.numpy as jnp

from jaxfluids.time_integration.time_integrator import TimeIntegrator

class RungeKutta2(TimeIntegrator):
    """2nd-order TVD RK2 scheme
    """
    def __init__(self, nh: int, inactive_axis: List) -> None:
        super(RungeKutta2, self).__init__(nh, inactive_axis)
        self.no_stages = 2
        self.timestep_multiplier = (1.0, 0.5)
        self.timestep_increment_factor = (1.0, 1.0) 

    def prepare_buffer_for_integration(self, cons: jnp.DeviceArray, init: jnp.DeviceArray, stage: int) -> jnp.DeviceArray:
        """ u_cons = 0.5 u^n + 0.5 u^* """
        return 0.5*cons + 0.5*init

    def integrate(self, cons: jnp.DeviceArray, rhs: jnp.DeviceArray, timestep: float, stage: int) -> jnp.DeviceArray:
        timestep = timestep * self.timestep_multiplier[stage]
        cons = self.integrate_conservatives(cons, rhs, timestep)
        return cons

