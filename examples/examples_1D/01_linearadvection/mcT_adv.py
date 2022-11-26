from typing import Iterator, NamedTuple
import time, os, wandb
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jrand
from jax import value_and_grad, vmap, jit, lax
import json
import haiku as hk
import optax
import mcTangentNN
from jaxfluids.solvers.riemann_solvers import RusanovNN, RiemannNN
from jaxfluids import InputReader, Initializer, SimulationManager
from jaxfluids.post_process import load_data

"""
Create a case dict from linearadvection.json and modify it to hold an mcTangent network.
Run JAX-Fluids to train mcTangent
Visualize results
"""
# %% setup
# get parameters
from mcT_parameters import *
case = 'linearadvection'

# initial condition coefficients
a_train_arr = jrand.normal(a_train_key, (num_train, 5))
a_test_arr = jrand.normal(a_test_key, (num_test, 5))

# uploading wandb
filename = 'mcT_adv'
wandb.init(project="mcTangent")
wandb.config.problem = filename
wandb.config.mc_alpha = mc_alpha
wandb.config.learning_rate = learning_rate
wandb.config.num_epochs = num_epochs
wandb.config.batch_size = batch_size
wandb.config.n_seq = n_seq
wandb.config.layer = layers
wandb.config.method = 'Dense_net'

# %% get or generate data
test_to_run = num_test
train_to_run = num_train
test_path = os.path.join('data','test')
train_path = os.path.join('data','train')

# make data directory if it does not exist
if not os.path.exists('data'):
    os.makedirs('data')
# if data exists check if the number of samples meets num_test/num_train requirement
else:
    if os.path.exists(test_path):
        if len(os.listdir(test_path)) < num_test:
            test_to_run = num_test - len(os.listdir(test_path))
    if os.path.exists(train_path):
        if len(os.listdir(train_path)) < num_train:
            train_to_run = num_train - len(os.listdir(train_path))

# generate data if needed
def generate(save_path: str, coefs: list) -> None: # writes data to separate h5 file
    """
    creates a random initial condition using the given coefficients
    and runs JAX-Fluids and writes the result to save_path
    
    ----- arguments -----
    save_path: string containing directory JAX-Fluids will save data
    coefs: list of 5 numbers used as coefficients in the initial density profile

    ----- returns ------
    no return, data is saved in an h5 file located at save_path
    """

    # edit case setup
    f = open(case + '.json', 'r')
    case_setup = json.load(f)
    f.close()

    case_setup['general']['save_path'] = save_path
    case_setup['general']['end_time'] = t_max
    case_setup['general']['save_dt'] = dt

    case_setup['domain']['x']['cells'] = nx
    case_setup['domain']['x']['range'] = [0, x_max]

    # random initial condition
    a1, a2, a3, a4, a5 = coefs
    case_setup['initial_condition']['rho'] = \
        "lambda x: " +\
            "((x>=0.2) & (x<=0.4)) * ("+str(a1)+"*(np.exp(-334.477 * (x-0.3-0.005)**2) + np.exp(-334.477 * (x - 0.3 + 0.005)**2) + 4 * np.exp(-334.477 * (x - 0.3)**2))) + " +\
            "((x>=0.6) & (x<=0.8)) * "+str(a2)+" + ((x>=1.0) & (x<=1.2)) * (1 - " +\
            "np.abs("+str(a3)+" * (x - 1.1))) + ((x>=1.4) & (x<=1.6)) * " +\
            "("+str(a4)+" * (np.sqrt(np.maximum( 1 - 100 * (x - 1.5 - 0.005)**2, 0)) + np.sqrt(np.maximum( 1 - 100 * (x - 1.5 + 0.005)**2, 0)) + " +\
            str(a5)+" * np.sqrt(np.maximum( 1 - 100 * (x - 1.5)**2, 0))) ) + " +\
            "~( ((x>=0.2) & (x<=0.4)) | ((x>=0.6) & (x<=0.8)) | ((x>=1.0) & (x<=1.2)) | ((x>=1.4) & (x<=1.6)) ) * 0.01"
    
    # edit numerical setup
    f = open('numerical_setup.json', 'r')
    num_setup = json.load(f)
    f.close()

    # fixed timestep for mcTangent compatability
    num_setup["conservatives"]["time_integration"]['fixed_timestep'] = dt

    # setup sim
    input_reader = InputReader(case_setup, num_setup)
    initializer = Initializer(input_reader)
    sim_manager = SimulationManager(input_reader)

    # run sim
    buffer_dict = initializer.initialization()
    sim_manager.simulate(buffer_dict)

if test_to_run > 0:
    print("-" * 15)
    print('Generating test data...')
    for ii in range(test_to_run):
        generate(test_path, a_test_arr[ii,:])
if train_to_run > 0:
    print("-" * 15)
    print('Generating train data...')
    for ii in range(train_to_run):
        generate(train_path, a_train_arr[ii,:])

# load data
test_data = np.zeros((num_test,nt,nx))
train_data = np.zeros((num_train,nt,nx))
test_setup = []
train_setup = []

test_samples = os.listdir(test_path)
train_samples = os.listdir(train_path)

print("-" * 15)
print('Loading data...')
for ii in range(num_test):
    sample_path = os.path.join(test_path,test_samples[ii])
    _, _, _, data_dict = load_data(os.path.join(sample_path,'domain'))
    test_data[ii,...] = jnp.reshape(data_dict['density'][:,:,0,0],(nt,nx))
    
    # save setup dicts for running JAX-Fluids with mcTangent later
    f = open(os.path.join(sample_path,case+'.json'), 'r')
    case_setup = json.load(f)
    f.close(); f = open(os.path.join(sample_path,'numerical_setup.json'), 'r')
    num_setup = json.load(f)
    test_setup.append({
        'case' : case_setup,
        'numerical' : num_setup
    })


for ii in range(num_train):
    sample_path = os.path.join(test_path,train_samples[ii])
    _, _, _, data_dict = load_data(os.path.join(train_path,train_samples[ii],'domain'))
    train_data[ii,...] = jnp.reshape(data_dict['density'][:,:,0,0],(nt,nx))

    # save setup dicts for running JAX-Fluids with mcTangent later
    f = open(os.path.join(sample_path,case+'.json'), 'r')
    case_setup = json.load(f)
    f.close(); f = open(os.path.join(sample_path,'numerical_setup.json'), 'r')
    num_setup = json.load(f)
    train_setup.append({
        'case' : case_setup,
        'numerical' : num_setup
    })

# %% create mcTangent network
class Batch(NamedTuple):
  data: np.ndarray  # all rho(t,x) in a batch of sequences [batch_size, nt-ns, ns+2, nx]
  

class TrainingState(NamedTuple):
  params: hk.Params
  avg_params: hk.Params
  opt_state: optax.OptState

# dense network, layer count variable not yet implemented
def mcT_fn(state: jnp.ndarray) -> jnp.ndarray:
    """Dense network with 1 layer of ReLU units"""
    nx = state.shape[-1]
    mcT = hk.Sequential([
        hk.Linear(n_units), jax.nn.relu,
        hk.Linear(nx)
    ])
    # Forward Euler in time
    state_out = state + dt*mcT(state)
    return state_out

optimizer = optax.adam(learning_rate)
mcT_net = hk.without_apply_rng(hk.transform(mcT_fn))
initial_params = mcT_net.init(net_key)
initial_opt_state = optimizer.init(initial_params)
net_state = TrainingState(initial_params, initial_params, initial_opt_state)


# train and eval
def run_epoch(opt_state, data):
    loss = 0
    return lax.fori_loop(0, num_batches)

for epoch in range(num_epochs):
    t1 = time.time()
    pass
