from typing import Tuple, NamedTuple, Iterable, Union
import shutil, os, json, re, h5py
import matplotlib.pyplot as plt
import jax
from jax import vmap, pmap, lax, jit
from functools import partial
from jaxfluids import InputReader, Initializer, SimulationManager
from jaxfluids.post_process import load_data, create_lineplot
import jax.numpy as jnp
import numpy as np

import mcT_sod_setup as setup


class Sim(NamedTuple):
    domain: str
    case: dict
    numerical: dict

    def load(
        self,
        quantities: Iterable[str] = ["density", "velocityX", "pressure", "temperature"],
        dtype: Union[str, type] = "ARRAY",
    ):
        out = load_data(self.domain, quantities)
        if isinstance(dtype, type):
            dtype = "DICT" if dtype == dict else "ARRAY"
        if dtype == "DICT":
            return out
        if dtype == "ARRAY":
            out = (
                out[0],
                out[1],
                out[2],
                np.array([out[3][quant] for quant in out[3].keys()]),
            )
            return out


class Data:
    """
    Manages data for training and testing
    """

    def __init__(self, save_path):
        self.save_path = save_path
        self.check_sims()

    def check_sims(self) -> list:
        sim_list = os.listdir(self.save_path)
        sim_files = [os.path.join(self.save_path, sim) for sim in sim_list]
        sim_files.sort(key=lambda x: os.path.getmtime(x), reverse=False)
        sim_list = [os.path.split(sf)[1] for sf in sim_files]
        self.sims = sim_list

    def size(self) -> int:
        return len(self.sims)

    def next_sim(self) -> Sim:
        # takes sim out of queue and puts it at the back
        sim_path = self.sims.pop(0)
        # cycle sim to other end of list
        self.sims.append(sim_path)

        sim_name = (
            re.split(r"-\d+", sim_path)[0] if re.split(r"-\d+", sim_path) else sim_path
        )
        domain = os.path.join(self.save_path, sim_path, "domain")

        f = open(os.path.join(self.save_path, sim_path, sim_name + ".json"), "r")
        case_setup = json.load(f)
        f.close()

        f = open(os.path.join(self.save_path, sim_path, "numerical_setup.json"), "r")
        num_setup = json.load(f)
        f.close()

        return Sim(domain, case_setup, num_setup)

    def generate(self, reset=False):
        if reset:
            shutil.rmtree(self.save_path)

        self.check_sims()
        if self.size() < setup.cases.size():
            for _ in range(self.size()):
                _ = setup.cases.next()
            n_to_gen = setup.cases.size()

            for _ in range(n_to_gen):
                case_setup = setup.cases.next()
                num_setup = setup.numerical

                self.check_sims()
                if self.size() >= setup.num_train:
                    case_setup["general"]["end_time"] = setup.t_max * setup.test_ratio

                input_reader = InputReader(case_setup, num_setup)
                initializer = Initializer(input_reader)
                sim_manager = SimulationManager(input_reader)

                buffer_dictionary = initializer.initialization()
                sim_manager.simulate(buffer_dictionary)
        del setup.cases

    def calc_analytical(
        self, fx0: jnp.ndarray, xs: jnp.ndarray, t: jnp.ndarray
    ) -> jnp.ndarray:
        """Generate analytical solution for a shock tube.

        The solver assumes the same ideal gas on both sides of the discontinuity, with
        the ratio of specifc heats `gamma` = 1.4.

        Args:
            fx0: 3x2 array containing the initial left and right states
                (1D shock tube only uses density, x velocity, and pressure)
            xs: the x locations at which to find solutions
            t: the current timestep, with t=0 being the time of the initial condition.
        """
        gamma = 1.4

        rho4 = fx0[0, 0]
        rho1 = fx0[0, 1]
        u4 = fx0[1, 0]
        # u1 = fx0[1, 1]
        p4 = fx0[2, 0]
        p1 = fx0[2, 1]

        a4 = jnp.sqrt(gamma * p4 / rho4)
        a1 = jnp.sqrt(gamma * p1 / rho1)

        p2 = 0.5 * (p4 + p1)

        def p2_fn(p2):
            p_ratio = (
                p2
                / p1
                * (
                    1
                    - (gamma - 1)
                    * (a1 / a4)
                    * (p2 / p1 - 1)
                    / jnp.sqrt(2 * gamma * (2 * gamma + (gamma + 1) * (p2 / p1 - 1)))
                )
                ** (-2 * gamma / (gamma - 1))
            )
            return p_ratio - p4 / p1

        def newton_body(val):
            it, err, p2 = val
            f, fprime = jax.value_and_grad(p2_fn)(p2)
            err = p2
            p2 -= f / fprime
            err = jnp.abs(p2 - err) / err
            it += 1
            return it, err, p2

        def newton_cond(val):
            it, err, p2 = val
            maxit = 1000
            tol = 1e-8
            while_cond = jnp.where(
                jnp.sum(jnp.where((it < maxit) + (err > tol), 0, 1)) > 0, False, True
            )
            return while_cond

        it = 0
        err = 1

        _, _, p2 = lax.while_loop(newton_cond, newton_body, (it, err, p2))

        up = 2 * a4 / (gamma - 1) * (1 - (p2 / p4) ** ((gamma - 1) / 2 / gamma))
        shock_speed = a1 * jnp.sqrt((gamma + 1) / 2 / gamma * (p2 / p1 - 1) + 1)

        shock_x = 0.5 + shock_speed * t
        fan_head_x = 0.5 - a4 * t
        fan_tail_x = 0.5 - (a4 - up) * t
        contact_x = 0.5 + up * t

        def adiabatic_expansion(u, a0):
            adiabatic_ratio = 1 - (gamma - 1) / 2 * u / a0
            rho = rho4 * adiabatic_ratio ** (2 / (gamma - 1))
            p = p4 * adiabatic_ratio ** (2 * gamma / (gamma - 1))
            return jnp.array([rho, u, p])

        def expansion_wave(x, t):
            u = jnp.where(
                t > 0, 2 / (1 - gamma) * (x / jnp.where(t > 0, t, 0.1) + a4), u4
            )
            return adiabatic_expansion(u, a4)

        post_shock = jnp.array(
            [
                rho1
                * (1 + (gamma + 1) / (gamma - 1) * p2 / p1)
                / ((gamma + 1) / (gamma - 1) + p2 / p1),
                up,
                p2,
            ]
        )

        def colocate(x):
            x_prime = fx0[:, 0]
            x_prime = x_prime.at[:].set(
                jnp.where(
                    x <= fan_head_x,
                    x_prime,
                    jnp.where(
                        x <= fan_tail_x,
                        jnp.squeeze(
                            expansion_wave(jnp.where(x <= fan_tail_x, x, fan_tail_x), t)
                        ),
                        jnp.where(
                            x <= contact_x,
                            adiabatic_expansion(up, a4),
                            jnp.where(x <= shock_x, post_shock, fx0[:, 1]),
                        ),
                    ),
                )
            )
            return x_prime

        state = jnp.array(vmap(colocate)(xs))
        return state

    @partial(jit, static_argnums=0)
    def jitted_calc(self, fx0, xs, ts):
        return vmap(self.calc_analytical, in_axes=(None, None, 0))(fx0, xs, ts)

    def gen_analytical(self, reset=False):
        if reset:
            shutil.rmtree(self.save_path)

        self.check_sims()
        if self.size() < setup.cases.size():
            for _ in range(self.size()):
                _ = setup.cases.next()
            n_to_gen = setup.cases.size()

            for _ in range(n_to_gen):
                case_setup = setup.cases.next()
                # num_setup = setup.numerical

                self.check_sims()
                if self.size() >= setup.num_train:
                    case_setup["general"]["end_time"] = setup.t_max * setup.test_ratio

                x0 = jnp.array(case_setup["domain"]["x"]["range"])
                xs = jnp.expand_dims(
                    jnp.linspace(*x0, case_setup["domain"]["x"]["cells"]), 1
                )
                fx0 = jnp.zeros((5, 2))
                for i, var0 in enumerate(case_setup["initial_condition"].values()):
                    if type(var0) == str:
                        fx0 = fx0.at[i].set(eval(var0)(x0))
                    else:
                        fx0 = fx0.at[i].set(jnp.full_like(x0, var0))
                fx0 = fx0[[0, 1, 4], ...]

                tf = case_setup["general"]["end_time"]
                dt = case_setup["general"]["save_dt"]
                ts = jnp.expand_dims(jnp.linspace(0, tf, int(tf / dt) + 1), 1)

                i = 0
                while os.path.exists(
                    os.path.join(self.save_path, setup.case_name + str(i) + ".h5")
                ):
                    i += 1
                path = os.path.join(self.save_path, setup.case_name + str(i) + ".h5")

                trajectory = jnp.array(self.jitted_calc(fx0, xs, ts))
                trajectory = jnp.moveaxis(trajectory, -1, 0)
                trajectory = jnp.expand_dims(trajectory, (3, 4))

                hf = h5py.File(path, "w")
                hf.create_dataset("data", data=trajectory)
                hf.close()
                
                print("created {}".format(os.path.splitext(path)[0]))

    def _load(self, sim: Sim):
        return sim.load()[3]

    def load_all(self):
        # data_train = np.zeros((setup.num_train,5,setup.nt+1,setup.nx_fine,setup.ny_fine,setup.nz_fine))
        # for ii in range(setup.num_train):
        #     data_train[ii,...] = self._load(self.next_sim())
        data_load = np.zeros(
            (setup.num_test, 4, int(setup.nt * 2) + 1, setup.nx, setup.ny, setup.nz)
        )
        for ii in range(setup.num_test):
            data_load[ii, ...] = self._load(self.next_sim())
        data_test = data_load  # for one sample
        data_train = data_load[
            :, :, : setup.nt + 1, ...
        ]  # delete later, for one sample only
        self.check_sims()
        return data_train, data_test


data = Data(setup.save_path)

if __name__ == "__main__":
    # data.generate()
    data.gen_analytical()
