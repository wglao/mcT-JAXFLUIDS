from typing import Union, Tuple, NamedTuple
import os, functools
import json
import wandb
import numpy as np
import jax.random as jrand
import jax.numpy as jnp
"""parameters for initializing mcTangent"""
work = functools.partial(os.path.join,'./')
save_path = work('data')
parallel_flag = False

# data only
mc_flag = True
noise_flag = True

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

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
case_name = 'mcT_adv'

c = 0.9
u = 1.0

t_max = 2.0
nt = 200
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

num_epochs = int(1e4)
learning_rate = 1e-3
batch_size = 5
ns = 1
layers = 1

# sample set size
num_train = 50
num_test = 50
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

seeds_to_gen = np.arange(num_batches*batch_size+num_train+1)
# case_arr = np.empty(len(seeds_to_gen), dtype=object)

# edit numerical setup
f = open('numerical_setup.json','r')
numerical = json.load(f)
f.close()

numerical['conservatives']['time_integration']['fixed_timestep'] = 0.1*dt
numerical['conservatives']['convective_fluxes']['riemann_solver'] = "HLLC"

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