from typing import Tuple, NamedTuple
import time, os, wandb
import shutil
import numpy as np
import re
import jax
import jax.numpy as jnp
import jax.random as jrand
from jax import value_and_grad, vmap, jit, lax
import json
import pickle
import haiku as hk
import optax
from jaxfluids import InputReader, Initializer, SimulationManager
from jaxfluids.post_process import load_data
import matplotlib.pyplot as plt
"""
Train mcTangent to solve linear advection using JAX-Fluids
Run JAX-Fluids to train mcTangent:
    for sample in batch:
        1) Run fine-meshed case, sample every 10 dt_fine (dt_coarse = 10*dt_fine)
        2) for sequence in sample:
            2a) Coarse-grain result by factor of 4
            2b) for step in ns:
                2b2) Step forward coarse case with mcTangent
                2b1) Step forward coarse case with simpler Riemann Solver
                2b3) mse(2b1, 2b2) to get mc loss
                2b4) mse(1b, 2b1) to get ml loss
                2b5) get total loss = loss_ml + mc_alpha*loss_mc
    3) loss = mean(all losses)
    4) update params
Visualize results
"""
# %% setup
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
case_name = 'mcT_adv'

# get parameters
from mcT_parameters import *

# edit case setup
f = open(case_name+'.json','r')
case_setup = json.load(f)
f.close()

case_setup['general']['save_path'] = 'data/epoch_0'
case_setup['general']['end_time'] = t_max
case_setup['general']['save_dt'] = dt

case_setup['domain']['x']['cells'] = 4*nx
case_setup['domain']['x']['range'][1] = x_max

# edit numerical setup
f = open('numerical_setup.json','r')
num_setup = json.load(f)
f.close()

num_setup['conservatives']['time_integration']['fixed_timestep'] = 0.1*dt
num_setup['conservatives']['convective_fluxes']['riemann_solver'] = "HLLC"

# random coefficients for initial conditions
train_coefs = jnp.square(jrand.normal(train_key,(num_train,5))) + jnp.finfo(jnp.float32).eps #strictly positive
test_coefs = jnp.square(jrand.normal(test_key,(num_test,5))) + jnp.finfo(jnp.float32).eps #strictly positive

# data only
mc_flag = False
noise_flag = False

if not mc_flag:
    mc_alpha = 0
if not noise_flag:
    noise_level = 0

# uploading wandb
wandb.init(project="mcTangent")
wandb.config.problem = case_name
wandb.config.mc_alpha = mc_alpha
wandb.config.learning_rate = learning_rate
wandb.config.num_epochs = num_epochs
wandb.config.batch_size = batch_size
wandb.config.ns = ns
wandb.config.layer = layers
wandb.config.method = 'Dense_net'

# %% create mcTangent network and training functions
class TrainingState(NamedTuple):
    params: hk.Params
    opt_state: optax.OptState
    net: hk.Transformed
    loss: float

class Setup(NamedTuple):
    case: dict         # contains base case setup
    numerical: dict    # contains base numerical setup
    batch_size: int    # number of data samples in each batch
    mc_alpha: float    # model-constrained loss weighting coefficient
    noise_level: float # magnitude of noise used to randomize data
    ns: int            # number of addiitonal steps in a training sequence U_k = [u_0, u_1, ... u_ns+1]
    num_batches: int   # number of batches per epoch
    num_epochs: int    # number of epochs to train

# dense network, layer count variable not yet implemented
def mcT_fn(state: jnp.ndarray) -> jnp.ndarray:
    """Dense network with 1 layer of ReLU units"""
    n_fields = state.shape[0]
    nx = state.shape[-1]
    mcT = hk.Sequential([
        hk.Flatten(),
        hk.Linear(5*nx*n_fields), jax.nn.relu,
        hk.Linear(nx*n_fields)
    ])
    flux = jnp.reshape(mcT(state),(state.shape))
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
    :param true: true state

    ----- returns -----\n
    :return mse: mean squared error between pred and true
    """
    if true is None:
        true = jnp.zeros(pred.shape)
    else:
        assert true.shape == pred.shape, "both arguments must have the same shape"
    return jnp.mean(jnp.square(pred - true))

def _get_loss_sample(state: TrainingState, setup: Setup, coefs: jnp.ndarray) -> float:
    """
    Uses a highly resolved simulation as ground truth to calculate loss over a sample
    
    ----- inputs -----\n
    :param state: holds parameters of the NN and optimizer
    :param setup: holds parameters for the operation of JAX-Fluids
    :param coefs: define the initial condition

    ----- returns -----\n
    :return loss_sample: average loss over all sequences in the sample
    """
    # run fine simulation
    setup.case['general']['case_name'] = 'train_fine'
    setup.case['initial_condition']['rho'] = "lambda x: "+\
        "((x>=0.2) & (x<=0.4)) * ( "+str(coefs[0])+"*(np.exp(-334.477 * (x-0.3-0.005)**2) + np.exp(-334.477 * (x - 0.3 + 0.005)**2) + 4 * np.exp(-334.477 * (x - 0.3)**2))) + "+\
        "((x>=0.6) & (x<=0.8)) * "+str(coefs[1])+" + "+\
        "((x>=1.0) & (x<=1.2)) * ("+str(coefs[2])+" - np.abs(10 * (x - 1.1))) + "+\
        "((x>=1.4) & (x<=1.6)) * ("+str(coefs[3])+"* (np.sqrt(np.maximum( 1 - 100 * (x - 1.5 - 0.005)**2, 0)) + np.sqrt(np.maximum( 1 - 100 * (x - 1.5 + 0.005)**2, 0)) + 4 * np.sqrt(np.maximum( 1 - 100 * (x - 1.5)**2, 0))) ) + "+\
        "~( ((x>=0.2) & (x<=0.4)) | ((x>=0.6) & (x<=0.8)) | ((x>=1.0) & (x<=1.2)) | ((x>=1.4) & (x<=1.6)) ) *"+str(coefs[4])
    input_reader = InputReader(setup.case,setup.numerical)
    initializer = Initializer(input_reader)
    sim_manager = SimulationManager(input_reader)
    buffer_dictionary = initializer.initialization()
    sim_manager.simulate(buffer_dictionary)

    # coarse sampling
    path = sim_manager.output_writer.save_path_domain
    quantities = buffer_dictionary["material_fields"]["primes"]
    _, _, _, data_dict_fine = load_data(path, quantities)
    data_dict_coarse = {}
    for quant in quantities:
        data_fine = data_dict_fine[quant]
        data_dict_coarse[quant] = jnp.mean(jnp.array([data_fine[i::4] for i in range(4)]), axis=0)

    # feed forward with mcTangent ns+1 steps
    coarse_case = setup.case
    coarse_num = setup.numerical
    coarse_case['domain']['x']['cells'] = setup.case['domain']['x']['cells']/4
    coarse_num['conservatives']['time_integration']['fixed_timestep'] = 10*setup.numerical['conservatives']['time_integration']['fixed_timestep']
    coarse_num['conservatives']['convective_fluxes']['riemann_solver'] = "MCTANGENT"

    ml_parameters_dict = {"riemann_solver":state.params}
    ml_networks_dict = hk.data_structures.to_immutable_dict({"riemannsolver": state.net})

    input_reader = InputReader(coarse_case,coarse_num)
    sim_manager = SimulationManager(input_reader)

    nt = int(setup.case['general']['end_time'] / setup.case['general']['save_dt']) + 1
    ml_primes_init = jnp.array([data_dict_coarse[key] for key in data_dict_coarse.keys()])[jnp.s_[:,:nt-setup.ns-1,:]]
    
    # switch 0th axis to time for feed forward mapping
    ml_primes_init = jnp.array([ml_primes_init[jnp.s_[:,t,:]] for t in range(ml_primes_init.shape[1])])
    ml_pred_arr, _ = sim_manager.feed_forward( # [ml_pred_arr] = [nt-ns seq, ns+2 times, primes, nx cells]
        batch_primes_init = ml_primes_init,
        batch_levelset_init = jnp.empty_like(ml_primes_init), # not needed for single-phase, but is a required arg for feed_forward
        n_steps = setup.ns+1,
        timestep_size = coarse_case['general']['save_dt'],
        t_start = 0,
        ml_parameters_dict = ml_parameters_dict,
        ml_networks_dict = ml_networks_dict
        )
    
    # ml loss
    # switch 0th axis to time for mapping
    ml_true = jnp.array([data_dict_coarse[key] for key in data_dict_coarse.keys()])[jnp.s_[:,1:,:]]
    ml_true = jnp.array([ml_true[jnp.s_[:,t,:]] for t in range(ml_true.shape[1])])
    ml_true_arr = jnp.array([ml_true[jnp.s_[seq:seq+setup.ns+1,...]] for seq in range(nt-setup.ns)])
    # [ml_true_arr] = [nt-ns seq, ns+1 times, primes, nx cells]
    ml_loss_arr = vmap(_mse,in_axes=(0,0))(
        # [map over seq, from time 1 to end of seq, all vars, all cells]
        ml_pred_arr[jnp.s_[:,1:,...]],
        ml_true_arr
    )
    ml_loss_sample = jnp.mean(ml_loss_arr)

    # feed forward with numerical solver
    mc_loss_sample = 0
    if mc_flag:
        coarse_num['conservatives']['convective_fluxes']['riemann_solver'] = "HLLC"
        input_reader = InputReader(coarse_case,coarse_num)
        sim_manager = SimulationManager(input_reader)
        
        #  map over times, concatenate all sequences
        mc_primes_init = jnp.concatenate(ml_pred_arr[jnp.s_[:,:-1,...]])
        # [mc_primes_init] = [(nt-ns-1)*(ns+1), primes, nx cells]
        mc_pred_arr, _ = sim_manager.feed_forward( # [mc_pred_arr] = [(nt-ns-1)*(ns+1), 2, primes, nx cells]
            batch_primes_init = mc_primes_init,
            batch_levelset_init = jnp.empty_like(mc_primes_init), # not needed for single-phase, but is a required arg
            n_steps = 1,
            timestep_size = coarse_case['general']['save_dt'],
            t_start = 0
            )
        
        # mc loss
        mc_loss_arr = vmap(_mse,in_axes=(0,0))(
            jnp.concatenate(ml_pred_arr[jnp.s_[:,1:,...]]),
            mc_pred_arr[jnp.s_[:,-1,...]]
        )
        mc_loss_sample = setup.mc_alpha*jnp.mean(mc_loss_arr)
        
    return ml_loss_sample + mc_loss_sample

def get_loss_batch(state: TrainingState, setup: Setup, coefs_batch: jnp.ndarray):
    """
    vectorized version of _get_loss_sample

    ----- inputs -----\n
    :param state: holds parameters of the NN and optimizer
    :param setup: holds parameters for the operation of JAX-Fluids
    :param coefs_batch: defines the initial conditions of each sample

    ----- returns -----\n
    :return loss_batch: average loss over batch
    """
    return jnp.mean(vmap(_get_loss_sample, in_axes=(None,None,0))(state,setup,coefs_batch))

def _evaluate_sample(state: TrainingState, setup: Setup, coefs: jnp.ndarray) -> jnp.ndarray:
    """
    creates a simulation manager to fully simulate the case using the updated mcTangent
    the resulting data is then loaded and used to calculate the mse across all test data

    ----- inputs -----\n
    :param state: holds parameters of the NN and optimizer
    :param setup: holds parameters for the operation of JAX-Fluids
    :param coefs: defines the initial conditions

    ----- returns -----\n
    :return sample_err: an array holding the error for every position in space and time for the sample
    """
    # run fine simulation
    setup.case['general']['case_name'] = 'test_fine'
    setup.case['initial_condition']['rho'] = "lambda x: "+\
        "((x>=0.2) & (x<=0.4)) * ( "+str(coefs[0])+"*(np.exp(-334.477 * (x-0.3-0.005)**2) + np.exp(-334.477 * (x - 0.3 + 0.005)**2) + 4 * np.exp(-334.477 * (x - 0.3)**2))) + "+\
        "((x>=0.6) & (x<=0.8)) * "+str(coefs[1])+" + "+\
        "((x>=1.0) & (x<=1.2)) * ("+str(coefs[2])+" - np.abs(10 * (x - 1.1))) + "+\
        "((x>=1.4) & (x<=1.6)) * ("+str(coefs[3])+"* (np.sqrt(np.maximum( 1 - 100 * (x - 1.5 - 0.005)**2, 0)) + np.sqrt(np.maximum( 1 - 100 * (x - 1.5 + 0.005)**2, 0)) + 4 * np.sqrt(np.maximum( 1 - 100 * (x - 1.5)**2, 0))) ) + "+\
        "~( ((x>=0.2) & (x<=0.4)) | ((x>=0.6) & (x<=0.8)) | ((x>=1.0) & (x<=1.2)) | ((x>=1.4) & (x<=1.6)) ) *"+str(coefs[4])
    input_reader = InputReader(setup.case,setup.numerical)
    initializer = Initializer(input_reader)
    sim_manager = SimulationManager(input_reader)
    buffer_dictionary = initializer.initialization()
    sim_manager.simulate(buffer_dictionary)

    # coarse sampling
    path = sim_manager.output_writer.save_path_domain
    quantities = buffer_dictionary["material_fields"]["primes"]
    _, _, _, data_dict_fine = load_data(path, quantities)
    data_dict_coarse = {}
    for quant in quantities:
        data_fine = data_dict_fine[quant]
        data_dict_coarse[quant] = jnp.mean(jnp.array([data_fine[i::4] for i in range(4)]), axis=0)

    # run mcTangent simulation
    coarse_case = setup.case
    coarse_num = setup.numerical
    coarse_case['general']['case_name'] = 'test_mcT'
    coarse_case['domain']['x']['cells'] = setup.case['domain']['x']['cells']/4
    coarse_num['conservatives']['time_integration']['fixed_timestep'] = 10*setup.numerical['conservatives']['time_integration']['fixed_timestep']
    coarse_num['conservatives']['convective_fluxes']['riemann_solver'] = "MCTANGENT"

    ml_parameters_dict = {"riemann_solver":state.params}
    ml_networks_dict = hk.data_structures.to_immutable_dict({"riemannsolver": state.net})

    input_reader = InputReader(setup.case,setup.numerical)
    initializer = Initializer(input_reader)
    sim_manager = SimulationManager(input_reader)
    buffer_dictionary = initializer.initialization()
    buffer_dictionary['machinelearning_modules'] = {
        'ml_parameters_dict': ml_parameters_dict,
        'ml_networks_dict': ml_networks_dict
    }
    sim_manager.simulate(buffer_dictionary)

    # get error
    path = sim_manager.output_writer.save_path_domain
    _, _, _, data_dict_mcT = load_data(path, quantities)
    sample_err = data_dict_mcT['density'][:,:,0,0] - data_dict_coarse['density'][:,:,0,0]

    # clean up

    return sample_err

def evaluate_epoch(state: TrainingState, setup: Setup, coefs_epoch: jnp.ndarray) -> float:
    """
    vectorized form of _evaluate_sample

    ----- inputs -----\n
    :param state: holds parameters of the NN and optimizer
    :param setup: holds parameters for the operation of JAX-Fluids
    :param coefs_batch: defines the initial conditions of each sample

    ----- returns -----\n
    :return epoch_err: mean squared error for the epoch
    """
    return _mse(vmap(_evaluate_sample, in_axes=(None,None,0))(state,setup,coefs_epoch))

def Train(state: TrainingState, setup: Setup, train_coefs: jnp.ndarray, test_coefs:jnp.ndarray) -> Tuple[TrainingState,TrainingState]:
    """
    Train mcTangent through end-to-end optimization in JAX-Fluids

    ----- inputs -----\n
    :param state: holds parameters of the NN and optimizer
    :param setup: holds parameters for the operation of JAX-Fluids
    :param train_coefs: define initial conditions in training
    :param test_coefs: define initial conditions in testing
    """
    min_err = 100
    epoch_min = 1
    best_state = state
    for epoch in range(setup.num_epochs):
        state.loss = 0
        setup.case['general']['save_path'] = re.sub(r'epoch_\d+', 'epoch_'+str(epoch), setup.case['general']['save_path'])
        t1 = time.time()
        for batch in range(setup.num_batches):
           # get loss and grads
            coefs_batch = lax.dynamic_slice_in_dim(train_coefs,batch*setup.batch_size,batch_size)
            loss_batch, grads = value_and_grad(get_loss_batch)(state,setup,coefs_batch)

            # update mcTangent
            updates, opt_state_new = optimizer.update(grads, state.opt_state)
            params_new = optax.apply_updates(state.params, updates)
            state.params = params_new
            state.opt_state = opt_state_new
            state.loss += loss_batch
        t2 = time.time()

        test_err = evaluate_epoch(state,setup,test_coefs)
        
        if test_err <= min_err:
            min_err = test_err
            epoch_min = epoch
            best_state = state

            # clean up
            del_path = os.listdir('data')
            del_path.sort()
            if len(del_path) > 1:
                shutil.rmtree(del_path[0])
        elif epoch < setup.num_epochs - 1:
            # clean up
            del_path = os.listdir('data')
            del_path.sort()
            shutil.rmtree(del_path[-1])

        if epoch % 10 == 0:  # Print every 10 epochs
            print("Data_d {:d} ns {:d} batch {:d} time {:.2e}s loss {:.2e} TE {:.2e}  TE_min {:.2e} EPmin {:d} EP {} ".format(
                num_train, ns, batch_size, t2 - t1, state.loss, test_err, min_err, epoch_min, epoch))

        wandb.log({"Train loss": float(state.loss), "Test Error": float(test_err), 'TEST MIN': float(min_err), 'Epoch' : float(epoch)})
    return best_state, state

# %% MAIN

# data input will be (primes_L, primes_R, cons_L, cons_R) ([6,nx], [6,nx], [5,nx], [5,nx]) -> [22,nx]
data_init = jnp.empty((22,nx))
optimizer = optax.adam(learning_rate)
mcT_net = hk.without_apply_rng(hk.transform(mcT_fn))
initial_params = mcT_net.init(net_key, data_init)
initial_opt_state = optimizer.init(initial_params)

state = TrainingState(initial_params, initial_opt_state, mcT_net, 0)
setup = Setup(case_setup,num_setup,batch_size,mc_alpha,noise_level)

best_state, end_state = Train(state,setup,train_coefs,test_coefs)

# save params
param_path = "network/parameters"
save_params(best_state.params,os.path.join(param_path,"best"))
save_params(end_state.params,os.path.join(param_path,"end"))

# %% visualize end state
sample_to_plot = 0
x = jnp.linspace(0,x_max,nx)
end_path = os.path.join('data','epoch'+str(num_epochs))
quantities = ['density']
fine_load_path = os.path.join(end_path,'test_fine-'+str(sample_to_plot),'domain') if sample_to_plot else os.path.join(end_path,'test_fine','domain')
mcT_load_path = os.path.join(end_path,'test_mcT-'+str(sample_to_plot),'domain') if sample_to_plot else os.path.join(end_path,'test_mcT','domain')
centers_fine, _, times, data_dict_fine = load_data(fine_load_path, quantities)
centers_mcT, _, _, data_dict_mcT = load_data(mcT_load_path, quantities)

data_true = data_dict_fine['density']
data_pred = data_dict_mcT['density']

n_plot = 3
plot_steps = np.linspace(0,nt,n_plot,dtype=int)
plot_times = times[plot_steps]

fig = plt.figure(figsize=(32,10))
for nn in range(n_plot):
    ut = jnp.reshape(data_true[plot_steps[nn], :], (4*nx, 1))
    up = jnp.reshape(data_pred[plot_steps[nn], :], (4*nx, 1))
    ax = fig.add_subplot(1, n_plot, nn+1)
    l1 = ax.plot(x, ut, '-', linewidth=2, label='True')
    l2 = ax.plot(x, up, '--', linewidth=2, label='Predicted')
    ax.set_aspect('auto', adjustable='box')
    ax.set_title('t = ' + str(plot_times[nn]))

    if nn == 0:
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center')