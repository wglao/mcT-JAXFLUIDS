from typing import List, Tuple, Union, Dict, NamedTuple
import time, os, wandb
import shutil
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jrand
from jax import value_and_grad, vmap, jit, lax
import json
import pickle
import haiku as hk
import optax
import mcTangentNN
from mcT_forward_schemes_1D import Upwind
from jaxfluids.solvers.riemann_solvers import RusanovNN, RiemannNN
from jaxfluids import InputReader, Initializer, SimulationManager
from jaxfluids.post_process import load_data
import matplotlib.pyplot as plt

"""
Create a case dict from linearadvection.json and modify it to hold an mcTangent network.
Run JAX-Fluids to train mcTangent
Visualize results
"""
# %% setup
# get parameters
from mcT_parameters import *

num_batches = np.ceil(num_train/batch_size)

# data only
mc_flag = False
noise_flag = False

if not mc_flag:
    mc_alpha = 0
if not noise_flag:
    noise_level = 0

case_name = 'linearadvection'

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
wandb.config.ns = ns
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
def generate(case_setup: dict, num_setup: dict) -> None: # writes data to separate h5 file
    """
    runs JAX-Fluids and writes the result to save_path in case_setup
    
    ----- arguments -----\n
    :param case_setup: dict containing details about the simulation for the InputReader
    :param num_setup: dict containing details about the solvers for the InputReader

    ----- returns ------\n
    no return, data is saved in an h5 file located at save_path
    """

    # setup sim
    input_reader = InputReader(case_setup, num_setup)
    initializer = Initializer(input_reader)
    sim_manager = SimulationManager(input_reader)

    # run sim
    buffer_dict = initializer.initialization()
    sim_manager.simulate(buffer_dict)

# edit case setup
f = open(case_name + '.json', 'r')
case_setup = json.load(f)
f.close()

case_setup['general']['end_time'] = t_max
case_setup['general']['save_dt'] = dt
case_setup['domain']['x']['cells'] = nx
case_setup['domain']['x']['range'] = [0, x_max]

# fixed timestep for mcTangent compatability
f = open('numerical_setup.json', 'r')
num_setup = json.load(f)
f.close()

if 'fixed_timestep' in num_setup["conservatives"]["time_integration"].keys():
    if num_setup["conservatives"]["time_integration"]['fixed_timestep'] != dt:
        num_setup["conservatives"]["time_integration"]['fixed_timestep'] = dt
        f = open('numerical_setup.json', 'w')
        json.dump(num_setup,f)
        f.close()
else:
    num_setup["conservatives"]["time_integration"]['fixed_timestep'] = dt
    f = open('numerical_setup.json', 'w')
    json.dump(num_setup,f)
    f.close()

if test_to_run > 0:
    print("-" * 15)
    print('Generating test data...')
    start = num_test - test_to_run
    case_setup['general']['save_path'] = test_path

    for ii in range(test_to_run):
        a1, a2, a3, a4, a5 = a_test_arr[start + ii,:]
        rho0 = "lambda x: " +\
                "((x>=0.2) & (x<=0.4)) * ("+str(a1)+"*(np.exp(-334.477 * (x-0.3-0.005)**2) + np.exp(-334.477 * (x - 0.3 + 0.005)**2) + 4 * np.exp(-334.477 * (x - 0.3)**2))) + " +\
                "((x>=0.6) & (x<=0.8)) * "+str(a2)+" + ((x>=1.0) & (x<=1.2)) * (1 - " +\
                "np.abs("+str(a3)+" * (x - 1.1))) + ((x>=1.4) & (x<=1.6)) * " +\
                "("+str(a4)+" * (np.sqrt(np.maximum( 1 - 100 * (x - 1.5 - 0.005)**2, 0)) + np.sqrt(np.maximum( 1 - 100 * (x - 1.5 + 0.005)**2, 0)) + " +\
                str(a5)+" * np.sqrt(np.maximum( 1 - 100 * (x - 1.5)**2, 0))) ) + " +\
                "~( ((x>=0.2) & (x<=0.4)) | ((x>=0.6) & (x<=0.8)) | ((x>=1.0) & (x<=1.2)) | ((x>=1.4) & (x<=1.6)) ) * 0.01"
        case_setup['initial_condition']['rho'] = rho0   
        generate(case_setup, num_setup)

if train_to_run > 0:
    print("-" * 15)
    print('Generating train data...')
    start = num_train - train_to_run

    case_setup['general']['save_path'] = train_path

    for ii in range(train_to_run):
        a1, a2, a3, a4, a5 = a_train_arr[start + ii,:]
        rho0 = "lambda x: " +\
                "((x>=0.2) & (x<=0.4)) * ("+str(a1)+"*(np.exp(-334.477 * (x-0.3-0.005)**2) + np.exp(-334.477 * (x - 0.3 + 0.005)**2) + 4 * np.exp(-334.477 * (x - 0.3)**2))) + " +\
                "((x>=0.6) & (x<=0.8)) * "+str(a2)+" + ((x>=1.0) & (x<=1.2)) * (1 - " +\
                "np.abs("+str(a3)+" * (x - 1.1))) + ((x>=1.4) & (x<=1.6)) * " +\
                "("+str(a4)+" * (np.sqrt(np.maximum( 1 - 100 * (x - 1.5 - 0.005)**2, 0)) + np.sqrt(np.maximum( 1 - 100 * (x - 1.5 + 0.005)**2, 0)) + " +\
                str(a5)+" * np.sqrt(np.maximum( 1 - 100 * (x - 1.5)**2, 0))) ) + " +\
                "~( ((x>=0.2) & (x<=0.4)) | ((x>=0.6) & (x<=0.8)) | ((x>=1.0) & (x<=1.2)) | ((x>=1.4) & (x<=1.6)) ) * 0.01"
        case_setup['initial_condition']['rho'] = rho0   
        generate(case_setup, num_setup)

# %% load data
test_data = np.zeros((num_test,nt,nx))
train_data = np.zeros((num_train,nt,nx))
test_setup = []
train_setup = []
test_times = np.zeros((num_test,nt))
train_times = np.zeros((num_train,nt))

test_samples = os.listdir(test_path)
train_samples = os.listdir(train_path)

print("-" * 15)
print('Loading data...')
for ii in range(num_test):
    sample_path = os.path.join(test_path,test_samples[ii])
    _, _, times, data_dict = load_data(os.path.join(sample_path,'domain'))
    test_data[ii,...] = jnp.reshape(data_dict['density'][:,:,0,0],(nt,nx))
    test_times[ii,:] = times
    
    # save setup dicts for running JAX-Fluids with mcTangent later
    f = open(os.path.join(sample_path,case_name+'.json'), 'r')
    case_setup = json.load(f)
    f.close()
    test_setup.append(case_setup)


for ii in range(num_train):
    sample_path = os.path.join(test_path,train_samples[ii])
    _, _, times, data_dict = load_data(os.path.join(train_path,train_samples[ii],'domain'))
    train_data[ii,...] = jnp.reshape(data_dict['density'][:,:,0,0],(nt,nx))
    train_times[ii,:] = times

    # save setup dicts for running JAX-Fluids with mcTangent later
    f = open(os.path.join(sample_path,case_name+'.json'), 'r')
    case_setup = json.load(f)
    f.close()
    train_setup.append(case_setup)

# %% create mcTangent network and training functions
class Batch(NamedTuple):
    data: jnp.ndarray  # all rho(t,x) in a batch of sequences [batch_size, nt-ns, ns+2, nx]
    times: jnp.ndarray # all times for all sequences in data [batch_size, nt-ns, ns+2]
    cases: list         # contains dicts for every sample in the batch

class TrainingState(NamedTuple):
    params: hk.Params
    opt_state: optax.OptState
    loss: float

class Setup(NamedTuple):
    test: list  # contains dicts for every test run
    train: list  # contains dicts for every train run
    numerical: dict  # contains numerical setup consistent across all runs
    save_path: str  # path to the directory containing save data from the evaluation step

# dense network, layer count variable not yet implemented
def mcT_fn(state: jnp.ndarray) -> jnp.ndarray:
    """Dense network with 1 layer of ReLU units"""
    nx = state.shape[-1]
    mcT = hk.Sequential([
        hk.Linear(n_units), jax.nn.relu,
        hk.Linear(nx)
    ])
    flux = mcT(state)
    return flux

def save_params(params, path):
    params = jax.device_get(params)
    os.makedirs(path)
    with open(path, 'wb') as fp:
        pickle.dump(params, fp)

def load_params(path):
    assert os.path.exists(path), "Specified parameter save path does not exist"
    with open(path, 'rb') as fp:
        params = pickle.load(fp)
    return jax.device_put(params)

def _mse(pred: jnp.ndarray, true=None) -> float:
    """
    calculates the mean squared error between a prediction and the ground truth
    if only one argument is provided, it is taken to be the error array (pred-true)

    ----- inputs -----\n
    :param pred: predicted state
    :
    """
    if true is None:
        true = jnp.zeros(pred.shape)
    else:
        assert true.shape == pred.shape, "both arguments must have the same shape"
    return jnp.mean(jnp.square(pred - true))

def _body_mc_solve(i, val) -> jnp.ndarray:
    # pass val into target mc solver
    new_val = Upwind(val[i,:], u, dt, dx)
    return jnp.concatenate((val,new_val))

def _step_fwd_one_seq(data: jnp.ndarray, times: jnp.ndarray, case_setup: dict, num_setup: dict, params: hk.Params, nn_fn: hk.Transformed) -> Tuple[jnp.DeviceArray, jnp.DeviceArray, jnp.DeviceArray]:
    """
    creates a simulation manager to feed forward the state by ns+1 steps
    outputs an array of predictions to be used in calculating the loss
    
    ----- inputs -----\n
    :param data: one sequence of a data sample, of shape [ns+2, nx]
    :param times: time of every profile in the sequence, of shape [ns+2]
    :param case_setup: dictionary used to create the InputReader
    :param num_setup: dictionary used to create the InputReader
    :param params: NN parameters
    :param nn_fn: Haiku transformed function containing NN architecture

    ----- returns -----\n
    :return pred_arr: array containing all predicted trajectories, of shape [ns+2, nx]
    :return time_arr: array containing all prediction times, of shape [ns+2]
    :return loss_one_seq: loss calculated from the processed sequence
    """
    # set up JAX-Fluids simulation
    case_setup['intial_condition']['rho'] = data[0,:]
    input_reader = InputReader(case_setup,num_setup)
    sim_manager = SimulationManager(input_reader)
    primes_init = jnp.reshape(sim_manager.domain_information.domain_slices_conservatives,(1,3))

    # step forward
    pred_arr, time_arr = sim_manager.feed_forward(primes_init,None,ns+1,dt,times[0],1,params,nn_fn)
    pred_mc = jnp.array([data[0,:]])
    pred_mc = lax.fori_loop(0,ns,_body_mc_solve,pred_mc)

    # calculate loss for sequence
    loss_ml = _mse(pred_arr[1:,:],data[1:,:])
    loss_mc = mc_alpha*_mse(pred_arr[1:,:],pred_mc[1:,:])
    loss_one_seq = loss_ml + mc_alpha*loss_mc

    return pred_arr, time_arr, loss_one_seq

def _step_fwd_one_sample(data_sample: jnp.ndarray, times_sample: jnp.ndarray, case_setup: dict, num_setup: dict, params: hk.Params, nn_fn: hk.Transformed) -> Tuple[jnp.DeviceArray, jnp.DeviceArray, jnp.DeviceArray]:
    """
    vectorized version of _step_fwd_one_seq
    
    ----- inputs -----\n
    :param data_sample: one sample of batched data, of shape [nt-ns, ns+2, nx]
    :param times_sample: times for all sequences in a sample, of shape [nt-ns, ns+2]
    :param case_setup: dictionary used to create the InputReader
    :param num_setup: dictionary used to create the InputReader
    :param params: NN parameters
    :param nn_fn: Haiku transformed function containing NN architecture

    ----- returns -----\n
    :return pred_arr: array containing all predicted trajectories, of shape [nt-ns, ns+2, nx]
    :return time_arr: array containing all prediction times, of shape [nt-ns, ns+2]
    :return loss_arr: loss over all sequences
    """
    return vmap(_step_fwd_one_seq, in_axes=(0,0,None,None,None,None), out_axes=(0,0,0))(data_sample, times_sample, case_setup, num_setup, params, nn_fn)

def step_fwd_batch(data_batch: jnp.ndarray, times_batch: jnp.ndarray, case_batch: list, num_setup: dict, params: hk.Params, nn_fn: hk.Transformed) -> Tuple[jnp.DeviceArray, jnp.DeviceArray, jnp.DeviceArray]:
    """
    vectorized form of _step_fwd_one_sample
    
    ----- inputs -----\n
    :param data_batch: batched dataset, of shape [batch_size, nt-ns, ns+2, nx]
    :param times_batch: times for all samples in a batch, of shape [batch_size, nt-ns, ns+2]
    :param case_batch: list of case setup dicts for each sample in the batch, of shape [batch_size]
    :param num_setup: dictionary used to create the InputReader
    :param params: NN parameters
    :param nn_fn: Haiku transformed function containing NN architecture

    ----- returns -----\n
    :return pred_arr: array containing all predicted trajectories, of shape [batch_size, nt-ns, ns+2, nx]
    :return time_arr: array containing all prediction times, of shape [batch_size, nt-ns, ns+2]
    :return loss_batch: mean loss over all samples and sequences
    """
    pred_arr, time_arr, loss_arr = vmap(_step_fwd_one_sample, in_axes=(0,0,0,None,None,None), out_axes=(0,0,0))(data_batch, times_batch, case_batch, num_setup, params, nn_fn)
    loss_batch = jnp.mean(loss_arr)
    return pred_arr, time_arr, loss_batch


def _body_make_data_seq(i, args) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    creates sequences and appends them into the batched format
    """
    data_sample, times_sample, data_seq, times_seq = args
    data_seq[i,...] = lax.dynamic_slice_in_dim(data_sample, i, i+ns+1)
    times_seq[i,:] = lax.dynamic_slice_in_dim(times_sample, i, i+ns+1)
    return data_seq, times_seq

def _make_data_seq(data_sample,times_sample,data_seq,times_seq) -> Tuple[jnp.ndarray, jnp.ndarray]:
    data_seq = lax.fori_loop(0,nt-ns,_body_make_data_seq,(data_sample,times_sample,data_seq,times_seq))
    return data_seq, times_seq

def make_data_batch(data_batch, times_batch, data_seq, times_seq) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    vectorized version of _make_data_seq
    """
    return vmap(_make_data_seq, in_axes=(0,0,0,0), out_axes=(0,0))(data_batch,times_batch,data_seq,times_seq)

def _body_batch_cases(i, args) -> list:
    cases, cases_batch, cases_idx = args
    cases_batch.append(cases[cases_idx[i]])
    return cases_batch

def make_batch(i,data,times,cases,num_samples) -> Batch:
    data_seq = np.zeros((batch_size,nt-ns,ns+2,nx))
    times_seq = np.zeros((batch_size,nt-ns,ns+2))
    cases_batch = []

    # batch data
    data_batch = lax.dynamic_slice_in_dim(data, i * batch_size, batch_size)
    times_batch = lax.dynamic_slice_in_dim(times, i * batch_size, batch_size)
    data_seq, times_seq = make_batch(data_batch,times_batch,data_seq,times_seq)

    cases_idx = lax.dynamic_slice_in_dim(jnp.arange(0,num_samples), i * batch_size, batch_size)
    cases_batch = lax.fori_loop(0,batch_size,_body_batch_cases,(cases,cases_batch,cases_idx))
    
    return Batch(data_seq,times_seq,cases_batch)

# train and eval
def update(state: TrainingState, batch: Batch, num_setup) -> TrainingState:
    # step forward in time for ns+1 steps through all batched data
    _, _, loss, grads = value_and_grad(step_fwd_batch)(batch.data, batch.times, batch.cases, num_setup, state.params, mcT_net)

    # update mcTangent
    updates, opt_state = optimizer.update(grads, state.opt_state)
    params = optax.apply_updates(state.params, updates)
    
    return TrainingState(params,opt_state,loss)

def _evaluate_sample(state: TrainingState, test_sample: jnp.ndarray, case_setup: dict, num_setup:dict, save_path: str, epoch: int) -> jnp.ndarray:
    """
    creates a simulation manager to fully simulate the case using the updated mcTangent
    the resulting data is then loaded and used to calculate the mse across all test data

    ----- inputs -----\n
    :param state: holds the params and architecture of the network
    :param test_sample: data to be tested against
    :param case_setup: dictionary used to create the InputReader
    :param num_setup: dictionary used to create the InputReader
    :param epoch: the current epoch

    ----- returns -----\n
    :return sample_err: an array holding the error for every position in space and time for the sample
    """
    # set up JAX-Fluids simulation
    case_setup['general']['save_path'] = save_path
    generate(case_setup, num_setup)

    # get error
    sim_dir = os.listdir(save_path)
    sim_dir.sort()
    _, _, _, data_dict = load_data(os.path.join(save_path,sim_dir[-1],'domain'))
    sample_err = jnp.reshape(data_dict['density'][:,:,0,0],(nt,nx)) - test_sample
    # only keep last epoch
    if epoch != num_epochs-1:
        shutil.rmtree(save_path)
    return sample_err

def evaluate(state: TrainingState, batch: Batch, setup: Setup, epoch: int) -> float:
    """
    calls vectorized form of _evaluate_sample and returns the mean squared error of the predictions

    ----- inputs -----\n
    :param state: holds the params and architecture of the network
    :param batch: holds test data
    :param setup: holds all setup dicts
    :param epoch: the current epoch

    ----- returns -----\n
    :return epoch_err: mean squared error for all test data, the error of the epoch
    """
    err_arr = vmap(_evaluate_sample, in_axes=(None,0,0,None,None,None), out_axes=(0))(state,batch.data,setup.test,setup.numerical,setup.save_path,epoch)
    epoch_err = _mse(err_arr)
    return epoch_err

def _body_epoch(i, args):
    train_data, train_times, state, setup = args
    
    train_batch = make_batch(i,train_data,train_times,setup.train,num_train)
    state = update(state,train_batch,setup.numerical)
    return state

def Train(train_data: jnp.ndarray, test_data:jnp.ndarray, train_times: jnp.ndarray, test_times: jnp.ndarray, num_epochs: int, state: TrainingState, setup: Setup):
    min_err = 100
    epoch_min = 1
    optimal_state = state
    for epoch in range(num_epochs):
        t1 = time.time()
        state = lax.fori_loop(0,num_batches,_body_epoch,(train_data,train_times,state,setup))
        t2 = time.time()

        test_batch = Batch(test_data,test_times,setup.test)
        test_err = evaluate(state,test_batch,setup,epoch)
        
        if test_err <= min_err:
            min_err = test_err
            epoch_min = epoch
            best_state = state

        if epoch % 1000 == 0:  # Print MSE every 1000 epochs
            print("Data_d {:d} ns {:d} batch {:d} time {:.2e}s loss {:.2e} TE {:.2e}  TE_min {:.2e} EPmin {:d} EP {} ".format(
                num_train, ns, batch_size, t2 - t1, state.loss, test_err, min_err, epoch_min, epoch))

        wandb.log({"Train loss": float(state.loss), "Test Error": float(test_err), 'TEST MIN': float(min_err), 'Epoch' : float(epoch)})
    return best_state, state

# %% MAIN
optimizer = optax.adam(learning_rate)
mcT_net = hk.without_apply_rng(hk.transform(mcT_fn))
initial_params = mcT_net.init(net_key, train_data[0,0,:])
initial_opt_state = optimizer.init(initial_params)
net_state = TrainingState(initial_params, initial_opt_state, 0)

# change solver to mcTangent
num_setup['conservatives']['convective_fluxes']['riemann_solver'] = 'MCTANGENT'
num_setup['machine_learning'] = {
    'ml_parameters_dict': net_state.params,
    'ml_network_dict': mcT_net
}
save_path = os.path.join('results','mcTangentNN')
setup = Setup(test_setup,train_setup,num_setup,save_path)

best_state, end_state = Train(train_data,test_data,train_times,test_times,num_epochs,net_state,setup)

# save params
param_path = "network/parameters"
save_params(best_state.params,os.path.join(param_path,"best"))
save_params(end_state.params,os.path.join(param_path,"end"))

# %% visualize end state
sample_to_plot = 75
x = jnp.linspace(0,x_max,nx)
sims = os.listdir(save_path)
sims.sort()
_, _, _, data_dict = load_data(os.path.join(save_path,sims[sample_to_plot],'domain'))

data_true = jnp.reshape(test_data[sample_to_plot,...],(nt,nx))
data_pred = jnp.reshape(data_dict['density'][:,:,0,0],(nt,nx))
times = test_times[sample_to_plot]

n_plot = 3
plot_steps = np.linspace(0,nt,n_plot,dtype=int)

fig = plt.figure(figsize=(32,10))
for nn in range(n_plot):
    ut = jnp.reshape(data_true[plot_steps[nn], :], (nx, 1))
    up = jnp.reshape(data_pred[plot_steps[nn], :], (nx, 1))
    ax = fig.add_subplot(1, n_plot, nn+1)
    l1 = ax.plot(x, ut, '-', linewidth=2, label='True')
    l2 = ax.plot(x, up, '--', linewidth=2, label='Predicted')
    ax.set_aspect('auto', adjustable='box')
    ax.set_title('t = ' + str(test_times[plot_steps[nn]]))

    if nn == 0:
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center')