from typing import Union, Tuple, NamedTuple, Optional, Iterable
import os, functools, sys, copy
import json
import pickle
import wandb
import numpy as np
import haiku as hk
import optax
import jax
import jax.random as jrand
import jax.numpy as jnp
from jax import jit
from jax.config import config
from jaxfluids.time_integration import DICT_TIME_INTEGRATION

import mcTangent as mct

"""debugging and config"""
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
# os.environ["XLA_FLAGS"] = "--xla_dump_to=/tmp/foo"
config.update("jax_debug_nans", False)
config.update("jax_disable_jit", False)
config.update("jax_enable_x64", True)

"""parameters for initializing mcTangent"""
os.environ["PROJ"] = '/home/wglao/Documents/PHO-ICES/mcT-JAXFLUIDS/examples/examples_1D/02_sod'
proj = functools.partial(os.path.join,os.environ["PROJ"])
save_path = proj('data')
parallel_flag = False

# data only = False, False
mc_flag = False
noise_flag = False

# use warm params
load_warm = False
load_last = False
# keep track of epochs if interrupted
last_epoch = 21 if load_last else 0

small_batch = True

vis_flag = False

case_name = 'mcT_sod'

u = 1.0

t_max = 0.2
nt = 100
dt = t_max/nt

x_max = 1.0
nx = 100
dx = x_max/float(nx)

ny = 1
nz = 1
nx_fine = 4*nx
ny_fine = ny
nz_fine = nz
cfl = u*dt/dx
integrator = "RK3"

mc_alpha = 1e5 if mc_flag else 0
noise_level = 0.02 if noise_flag else 0
ns = 1
nr = 1

num_epochs = int(300)
learning_rate = 1e-3
batch_size = 1
layers = 1
hidden_size = (5*nx)**2
activation = "relu"

# sample set size
num_train = 1
num_test = 1
test_ratio = 2

# define batch by number of sequences trained on, instead of samples
num_batches = int(np.ceil(num_train/batch_size))

# edit case setup
f = open(case_name+'.json','r')
case_base = json.load(f)
f.close()

case_base['general']['save_path'] = save_path
case_base['general']['end_time'] = t_max
case_base['general']['save_dt'] = dt

case_base['domain']['x']['cells'] = 4*nx
case_base['domain']['x']['range'][1] = x_max

seeds_to_gen = np.arange(num_test+num_train+1)
# case_arr = np.empty(len(seeds_to_gen), dtype=object)

# edit numerical setup
f = open('numerical_setup.json','r')
numerical = json.load(f)
f.close()

numerical['conservatives']['time_integration']['fixed_timestep'] = 0.1*dt
numerical['conservatives']['convective_fluxes']['riemann_solver'] = "HLLC"

class Cases():
    """
    Holds a stack of case setup dicts that can only be popped
    """
    def __init__(self,cases: list):
        self.cases = cases
    
    def next(self) -> dict:
        """
        pops last case
        """
        try:
            return self.cases.pop()
        except Exception as e:
            print(e)
    
    def size(self):
        return len(self.cases)

# random coefficients for initial conditions
def pos_coefs(seed: int, out_shape: Union[tuple,int]) -> jnp.ndarray:
    """
    provides an array of strictly positive coefficients
    """
    key = jrand.PRNGKey(seed)
    if type(out_shape) == int:
        return np.abs(jnp.sqrt(0.5)*jrand.normal(key,(out_shape,)) + jnp.sqrt(0.5)) + jnp.finfo(jnp.float32).eps #strictly positive
    return jnp.abs(jnp.sqrt(0.5)*jrand.normal(key,out_shape) + jnp.sqrt(0.5)) + jnp.finfo(jnp.float32).eps #strictly positive

def get_cases() -> Cases:
    case_list = []
    for seed in seeds_to_gen:
        case_new = copy.deepcopy(case_base)
        coefs = pos_coefs(seed,3)
        rho0 = f"lambda x: 1.0*(x <= {coefs[2]}) + {coefs[0]}*(x > {coefs[2]})",
        p0 = f"lambda x: 1.0*(x <= {coefs[2]}) + {coefs[1]}*(x > {coefs[2]})",
        case_new['initial_condition']['rho'] = rho0
        case_new['initial_condition']['p'] = p0

        case_list.append(case_new)

    return Cases(case_list)

cases = get_cases()

@jit
def _mse(array: jnp.ndarray):
    return jnp.mean(jnp.square(array))

@jit
def mse(pred: jnp.ndarray, true: Optional[jnp.ndarray] = None) -> float:
    """
    calculates the mean squared error between a prediction and the ground truth
    if only one argument is provided, it is taken to be the error array (pred-true)

    ----- inputs -----\n
    :param pred: predicted state
    :param true: true state

    ----- returns -----\n
    :return mse: mean squared error between pred and true
    """
    if true is None:
        return _mse(pred)
    return _mse(pred - true)

# dense network, layer count variable not yet implemented
# def mcT_fn(u_i: jnp.ndarray) -> jnp.ndarray:
#     """Dense network with 1 layer of ReLU units"""
#     mcT = hk.Sequential([
#         hk.Linear(nx + 1), jax.nn.relu,
#         # hk.Linear(32), jax.nn.relu,  # try second layer
#         hk.Linear(nx + 1)
#     ])
#     tangent_i = mcT(jnp.ravel(u_i))
#     return tangent_i
# net = hk.without_apply_rng(hk.transform(mcT_fn))

net = hk.without_apply_rng(mct.nn.create('dense',layers,hidden_size,activation,nx))
# optimizer = optax.adam(learning_rate)
# optimizer = mct.nn.eve()
optimizer = optax.eve()

def save_params(params: hk.Params, path: str, filename: Optional[str] = None) -> None:
    # params = jax.device_get(params)
    if filename:
        path = os.path.join(path,filename)
    os.makedirs(os.path.dirname(path),exist_ok=True)
    with open(path, 'wb') as fp:
        pickle.dump(params, fp)
        fp.close()

def load_params(path: str, filename: Optional[str] = None):
    if filename:
        path = os.path.join(path,filename)
    assert os.path.exists(path), "Specified parameter file does not exist"
    with open(path, 'rb') as fp:
        params = pickle.load(fp)
        fp.close()
    return params

def compare_params(params: hk.Params, shapes: Union[Iterable[Iterable[int]],hk.Params]) -> bool:
    """
    Compares two sets of network parameters or a parameter dict with a list of prescribed shapes
    Returns True if the params dict has all correct shapes

    ----- inputs -----\n
    :param params: list of network params to be checked
    :param shapes: baseline list of shapes or another parameter dict to compare to

    ----- returns -----\n
    :return match: True if shapes match
    """
    for ii in range(len(params)):
        for jj, layer in enumerate(params[ii]):
            for kk, wb in enumerate(params[ii][layer]):
                if jnp.isnan(params[ii][layer][wb]).any():
                    return False
                if type(shapes) == list:
                    if params[ii][layer][wb].shape != shapes[2*jj+kk]:
                        return False
                else:
                    for a,b in zip(params[ii][layer][wb].shape,shapes[ii][layer][wb].shape):
                        if a != b:
                            return False
    return True

if __name__ == "__main__":
    # uploading wandb
    wandb.init(project="mcT-JAXFLUIDS",name="Warm Start")
    wandb.config.problem = case_name
    wandb.config.mc_alpha = mc_alpha
    wandb.config.learning_rate = learning_rate
    wandb.config.num_epochs = num_epochs
    wandb.config.batch_size = batch_size
    wandb.config.ns = ns
    wandb.config.layer = layers
    wandb.config.method = 'Dense_net'

    from jaxfluids.utilities import get_fluxes_xi, get_conservatives_from_primitives
    from jaxfluids.solvers.riemann_solvers.HLLC import HLLC
    from jaxfluids.solvers.riemann_solvers.Rusanov import Rusanov
    from jaxfluids.solvers.riemann_solvers.signal_speeds import signal_speed_Einfeldt
    from jaxfluids.post_process import load_data
    from jaxfluids import InputReader, Initializer, SimulationManager

    import mcT_data as dat
    from mcT import get_coarse
    
    cache_path = '.test_cache'
    # optimizer = optax.adam(1e-3)

    print('\n'+'-'*5+'Warm Start'+'-'*5+'\n')
    
    def warm_loss(params,cons_L,cons_R,truth):
        net_tangent = jnp.zeros_like(cons_L)
        for i in range(5):
            net_L = jax.jit(jax.vmap(net.apply, in_axes=(None,0)))(params[i],cons_L[:,i])
            net_R = jax.jit(jax.vmap(net.apply, in_axes=(None,0)))(params[i],cons_R[:,i])
            tangent_i = jnp.reshape(0.5*(net_L+net_R), net_tangent[:,i,...].shape)
            net_tangent = net_tangent.at[:,i,...].set(tangent_i)
        loss = mse(net_tangent*dx,truth)
        return loss

    def warm_load(sim: dat.Sim, sim_manager: SimulationManager, epoch: int):
        primes = jax.device_put(sim.load()[3])
        primes = jax.vmap(get_coarse, in_axes=(0))(primes)
        primes = jnp.swapaxes(primes,0,1)
        nh = numerical['conservatives']['halo_cells']
        pad_dims = ((0,0),(0,0),(nh,nh),(0,0),(0,0))
        primes = jnp.pad(primes,pad_dims,constant_values=1)
        primes_L = jnp.array(jax.vmap(
            sim_manager.space_solver.flux_computer.flux_computer.reconstruction_stencil.reconstruct_xi,
            in_axes=(0,None,None,None))(primes, 0, 0, dx))
        primes_R = jnp.array(jax.vmap(
            sim_manager.space_solver.flux_computer.flux_computer.reconstruction_stencil.reconstruct_xi,
            in_axes=(0,None,None,None))(primes, 0, 1, dx))
        cons_L = jnp.array(jax.vmap(get_conservatives_from_primitives, in_axes=(0,None))(primes_L,sim_manager.material_manager))
        cons_R = jnp.array(jax.vmap(get_conservatives_from_primitives, in_axes=(0,None))(primes_R,sim_manager.material_manager))

        return primes_L,primes_R,cons_L,cons_R

    def warm_start(epochs):
        # init net params
        u_init = jnp.zeros((1,nx+1,ny,nz))
        params = net.init(jrand.PRNGKey(epochs), u_init)
        opt_state = [optimizer.init(params) for _ in range(5)]
        params = [params for _ in range(5)]
        del u_init

        min_err = sys.float_info.max
        epoch_min = -1
        for epoch in range(epochs):
            dat.data.check_sims()
            sim = dat.data.next_sim()
            for _ in range(epoch%dat.data.size()):  # vary training data
                sim = dat.data.next_sim()
            case_dict = sim.case
            numerical = sim.numerical

            case_dict['general']['save_path'] = cache_path
            case_dict['domain']['x']['cells'] = nx
            numerical['conservatives']['time_integration']['fixed_timestep'] = dt

            input_reader = InputReader(case_dict,numerical)
            sim_manager = SimulationManager(input_reader)

            # load data
            primes_L,primes_R,cons_L,cons_R = warm_load(sim,sim_manager,epoch)

            # learn
            model = HLLC(sim_manager.material_manager,signal_speed_Einfeldt)
            # model = Rusanov(sim_manager.material_manager,signal_speed_Einfeldt)

            warm_true = jax.vmap(model.solve_riemann_problem_xi, in_axes=(0,0,0,0,None))
            truth_array = jnp.array(warm_true(primes_L,primes_R,cons_L,cons_R,0))
            loss, grads = jax.value_and_grad(jit(warm_loss),argnums=(0))(params,cons_L,cons_R,truth_array)
            for i in range(5):
                updates_i, opt_state[i] = jit(optimizer.update)(grads[i], opt_state[i])
                params[i] = jit(optax.apply_updates)(params[i], updates_i)
            
            # test
            sim = dat.data.next_sim()
            primes_L,primes_R,cons_L,cons_R = warm_load(sim,sim_manager,epoch)
            test_truth = warm_true(primes_L,primes_R,cons_L,cons_R,0)
            test_err = warm_loss(params,cons_L,cons_R,test_truth)

            if test_err < min_err:
                epoch_min = epoch
                min_err = test_err

            if epoch % 500 == 0 or epoch == epochs-1:
                print("Loss {:.2e} TE {:.2e}  TE_min {:.2e} EPmin {:d} EP {} ".format(loss, test_err, min_err, epoch_min, epoch))
            
            if epoch % 200 == 0 and epoch > 0:
                jax.clear_backends()
            wandb.log({"Train loss": float(loss), "Test Error": float(test_err), 'Test Min': float(min_err), 'Epoch' : float(epoch)})
            # clean up
            os.system('rm -rf %s' %(cache_path))
        
        save_params(params,os.path.join(proj("network/parameters"),"warm.pkl"))

    warm_epochs = 501
    warm_start(warm_epochs)
else:
    # uploading wandb
    wandb.init(project="mcT-JAXFLUIDS",name=case_name)
    wandb.config.problem = case_name
    wandb.config.mc_alpha = mc_alpha
    wandb.config.learning_rate = learning_rate
    wandb.config.num_epochs = num_epochs
    wandb.config.batch_size = batch_size
    wandb.config.ns = ns
    wandb.config.layer = layers
    wandb.config.method = 'Dense_net'
