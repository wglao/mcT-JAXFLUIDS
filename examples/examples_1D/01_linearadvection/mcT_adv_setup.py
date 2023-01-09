from typing import Union, Tuple, NamedTuple, Optional, Iterable
import os, functools, sys
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

"""debugging and config"""
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# os.environ["XLA_FLAGS"] = "--xla_dump_to=/tmp/foo"
config.update("jax_debug_nans", True)
config.update("jax_disable_jit", False)
config.update("jax_enable_x64", True)

"""parameters for initializing mcTangent"""
proj = functools.partial(os.path.join,os.environ["PROJ"])
save_path = proj('data')
parallel_flag = False

# data only
mc_flag = False
noise_flag = True

case_name = 'mcT_adv'

c = 0.9
u = 1.0

t_max = 2.0
nt = 100
dt = t_max/nt

x_max = 2.0
dx = u*dt/c
nx = np.ceil(x_max/dx)
dx = x_max/float(nx)

nx = int(nx)
ny = 1
nz = 1
nx_fine = 4*nx
ny_fine = ny
nz_fine = nz

mc_alpha = 1e5 if mc_flag else 0
noise_level = 0.02 if noise_flag else 0
ns = 1

num_epochs = int(200)
learning_rate = 1e-6
batch_size = nt-ns-1
layers = 1

# sample set size
num_train = 10
num_test = 10

# define batch by number of sequences trained on, instead of samples
train_seqs = int(nt-ns-1)
num_batches = int(np.ceil(train_seqs/batch_size))

# use warm params
load_warm = True

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
        return jnp.square(jrand.normal(key,(out_shape,))) + jnp.finfo(jnp.float32).eps #strictly positive
    return jnp.square(jrand.normal(key,out_shape)) + jnp.finfo(jnp.float32).eps #strictly positive

def get_cases() -> Cases:
    case_list = []
    for seed in seeds_to_gen:
        case_new = case_base
        coefs = pos_coefs(seed,5)
        rho0 = "lambda x: 1+"+\
            "((x>=0.2) & (x<=0.4)) * ( {0}*(np.exp(-334.477 * (x-0.3-0.005)**2) + np.exp(-334.477 * (x - 0.3 + 0.005)**2) + 4 * np.exp(-334.477 * (x - 0.3)**2))) + "+\
            "((x>=0.6) & (x<=0.8)) * {1} + "+\
            "((x>=1.0) & (x<=1.2)) * ({2} - np.abs(10 * (x - 1.1))) + "+\
            "((x>=1.4) & (x<=1.6)) * ({3}* (np.sqrt(np.maximum( 1 - 100 * (x - 1.5 - 0.005)**2, 0)) + np.sqrt(np.maximum( 1 - 100 * (x - 1.5 + 0.005)**2, 0)) + 4 * np.sqrt(np.maximum( 1 - 100 * (x - 1.5)**2, 0))) ) + "+\
            "~( ((x>=0.2) & (x<=0.4)) | ((x>=0.6) & (x<=0.8)) | ((x>=1.0) & (x<=1.2)) | ((x>=1.4) & (x<=1.6)) ) *{4}"
        case_new['initial_condition']['rho'] = rho0.format(*coefs)

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
def mcT_fn(primes_L: jnp.ndarray, primes_R: jnp.ndarray, cons_L: jnp.ndarray, cons_R: jnp.ndarray) -> jnp.ndarray:
    """Dense network with 1 layer of ReLU units"""
    mcT = hk.Sequential([
        hk.Linear(32), jax.nn.relu,
        hk.Linear(32), jax.nn.relu,  # try second layer
        hk.Linear(5*(nx + 1))
    ])
    state = jnp.concatenate((primes_L,primes_R,cons_L,cons_R),axis=None)
    flux = mcT(state)
    return flux

net = hk.without_apply_rng(hk.transform(mcT_fn))
optimizer = optax.adam(learning_rate)

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
    :param params: network params to be checked
    :param shapes: baseline list of shapes or another parameter dict to compare to

    ----- returns -----\n
    :return match: True if shapes are correct
    """
    for ii, layer in enumerate(params):
        for jj, wb in enumerate(params[layer]):
            if jnp.isnan(params[layer][wb]).any():
                return False
            if type(shapes) == list:
                if params[layer][wb].shape != shapes[2*ii+jj]:
                    return False
            else:
                for i,j in zip(params[layer][wb].shape,shapes[layer][wb].shape):
                    if i != j:
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

    import mcT_adv_data as dat
    from mcT_adv import get_coarse
    
    cache_path = '.test_cache'
    optimizer = optax.adam(1e-3)

    print('\n'+'-'*5+'Warm Start'+'-'*5+'\n')
    
    def warm_loss(params,primes_L,primes_R,cons_L,cons_R,truth):
        net_out = jnp.array(jax.vmap(net.apply, in_axes=(None,0,0,0,0))(params,primes_L,primes_R,cons_L,cons_R))
        net_fluxes = jnp.reshape(net_out,truth.shape)
        loss = mse(net_fluxes,truth)
        return loss

    def warm_load(sim: dat.Sim, sim_manager: SimulationManager, epoch: int):
        primes = jax.device_put(sim.load()[3])
        primes = jax.vmap(get_coarse, in_axes=(0,None))(primes,epoch)
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
        rho_init = jnp.ones((1,nx+1,ny,nz))
        primes_init = jnp.concatenate((rho_init,jnp.ones_like(rho_init),jnp.zeros_like(rho_init),jnp.zeros_like(rho_init),jnp.ones_like(rho_init)))
        cons_init = jnp.concatenate((rho_init,rho_init,jnp.zeros_like(rho_init),jnp.zeros_like(rho_init),1.5*rho_init))
        params = net.init(jrand.PRNGKey(epochs), primes_init, primes_init, cons_init, cons_init)
        opt_state = optimizer.init(params)
        del rho_init, primes_init, cons_init

        min_err = sys.float_info.max
        epoch_min = -1
        for epoch in range(epochs):
            dat.data.check_sims()
            sim = dat.data.next_sim()
            case_dict = sim.case
            numerical = sim.numerical

            case_dict['general']['save_path'] = cache_path
            case_dict['domain']['x']['cells'] = nx
            numerical['conservatives']['time_integration']['fixed_timestep'] = dt

            input_reader = InputReader(case_dict,numerical)
            initializer = Initializer(input_reader)
            sim_manager = SimulationManager(input_reader)

            # load data
            primes_L,primes_R,cons_L,cons_R = warm_load(sim,sim_manager,epoch)

            # learn
            # truth_array = warm_true(primes_L,cons_L)
            model = HLLC(sim_manager.material_manager,signal_speed_Einfeldt)
            # model = Rusanov(sim_manager.material_manager,signal_speed_Einfeldt)
            warm_true = jax.vmap(model.solve_riemann_problem_xi, in_axes=(0,0,0,0,None))
            truth_array = jnp.array(warm_true(primes_L,primes_R,cons_L,cons_R,0))
            loss, grads = jax.value_and_grad(jit(warm_loss),argnums=(0))(params,primes_L,primes_R,cons_L,cons_R,truth_array)
            updates, opt_state = jit(optimizer.update)(grads, opt_state)
            params = jit(optax.apply_updates)(params, updates)
            
            # test
            sim = dat.data.next_sim()
            primes_L,primes_R,cons_L,cons_R = warm_load(sim,sim_manager,epoch)
            test_truth = warm_true(primes_L,primes_R,cons_L,cons_R,0)
            test_err = warm_loss(params,primes_L,primes_R,cons_L,cons_R,test_truth)

            if test_err < min_err:
                epoch_min = epoch
                min_err = test_err

            if epoch % 500 == 0:
                print("Loss {:.2e} TE {:.2e}  TE_min {:.2e} EPmin {:d} EP {} ".format(loss, test_err, min_err, epoch_min, epoch))
            
            if epoch % 4000 == 0 and epoch > 0:
                jax.clear_backends()
            wandb.log({"Train loss": float(loss), "Test Error": float(test_err), 'Test Min': float(min_err), 'Epoch' : float(epoch)})
            # clean up
            os.system('rm -rf %s' %(cache_path))
        
        save_params(params,os.path.join(proj("network/parameters"),"warm.pkl"))

    warm_epochs = 3001
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
