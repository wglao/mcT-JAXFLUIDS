from typing import Tuple, NamedTuple
import time, os, wandb
import shutil, functools
import numpy as np
import re
import jax
import jax.numpy as jnp
import jax.random as jrand
import jax.image as jim
from jax import value_and_grad, vmap, jit, lax, pmap
import json
import pickle
import haiku as hk
import optax
from jaxfluids import InputReader, Initializer, SimulationManager
from jaxfluids.post_process import load_data
import matplotlib.pyplot as plt

import mcT_adv_setup as setup
import mcT_adv_data as dat

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
Evaluate against validation set to get error
Visualize results
"""

# %% create mcTangent network and training functions
class TrainingState(NamedTuple):
    params: hk.Params
    opt_state: optax.OptState
    loss: float

# dense network, layer count variable not yet implemented
def mcT_fn(state: jnp.ndarray) -> jnp.ndarray:
    """Dense network with 1 layer of ReLU units"""
    n_fields = state.shape[0]
    n_faces = state.shape[1]
    mcT = hk.Sequential([
        hk.Flatten(),
        hk.Linear(5*n_faces*n_fields), jax.nn.relu,
        hk.Linear(n_faces)
    ])
    flux = mcT(state)
    flux = jnp.reshape(flux,(n_fields,n_faces,1,1))
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
    # else:
    #     assert true.shape == pred.shape, "both arguments must have the same shape"
    return jnp.mean(jnp.square(pred - true))

def _get_loss_sample(params: hk.Params, *args) -> float:
    """
    Uses a highly resolved simulation as ground truth to calculate loss over a sample
    
    ----- inputs -----\n
    :param params: holds parameters of the NN
    :param args: allows for mapping input for vmap api

    ----- returns -----\n
    :return loss_sample: average loss over all sequences in the sample
    """
    # load fine simulation
    fine_sim = dat.data.next_sim()
    quantities = ['density','velocityX','temperature']
    _, _, _, data_dict_fine = load_data(fine_sim.domain, quantities)
    # coarse sampling
    data_dict_coarse = {}
    for quant in quantities:
        data_fine = data_dict_fine[quant]
        data_dict_coarse[quant] = jim.resize(data_fine,(setup.nt+1,setup.nx,1,1),"linear")

    # feed forward with mcTangent ns+1 steps
    coarse_case = fine_sim.case
    coarse_num = fine_sim.numerical
    coarse_case['domain']['x']['cells'] = setup.nx
    coarse_num['conservatives']['time_integration']['fixed_timestep'] = setup.dt
    coarse_num['conservatives']['convective_fluxes']['riemann_solver'] = "MCTANGENT"

    ml_parameters_dict = {"riemann_solver":params}
    ml_networks_dict = hk.data_structures.to_immutable_dict({"riemann_solver": mcT_net})

    input_reader = InputReader(coarse_case,coarse_num)
    sim_manager = SimulationManager(input_reader)

    ml_primes_buff = jnp.array([data_dict_coarse[key] for key in data_dict_coarse.keys()])[jnp.s_[:,:setup.nt-setup.ns,...]]
    ml_primes_init = jnp.zeros((5,setup.nt-setup.ns,setup.nx,1,1))
    for ii, prime in enumerate([0,1,4]):
        ml_primes_init = ml_primes_init.at[prime,...].set(jnp.reshape(ml_primes_buff[jnp.s_[ii,...]],(setup.nt-setup.ns,setup.nx,1,1)))
    
    # switch 0th axis to time for feed forward mapping
    ml_primes_init = jnp.swapaxes(ml_primes_init,1,0)
    # ml_primes_init = jnp.array([ml_primes_init[jnp.s_[:,t,:]] for t in range(ml_primes_init.shape[1])])

    feed_forward = functools.partial(sim_manager.feed_forward,ml_parameters_dict=ml_parameters_dict,ml_networks_dict=ml_networks_dict)
    ml_pred_arr, _ = feed_forward(
        ml_primes_init,
        jnp.empty_like(ml_primes_init), # not needed for single-phase, but is a required arg for feed_forward
        setup.ns+1, coarse_case['general']['save_dt'], 0
    )
    
    # ml_pred_arr, _ = sim_manager.feed_forward(
    #     ml_primes_init,
    #     jnp.empty_like(ml_primes_init), # not needed for single-phase, but is a required arg for feed_forward
    #     setup.ns+1, coarse_case['general']['save_dt'], 0, ml_parameters_dict, ml_networks_dict
    # )

    # ml loss
    # switch 0th axis to time for mapping
    ml_true_buff = jnp.array([data_dict_coarse[key] for key in data_dict_coarse.keys()])[jnp.s_[:,1:,:]]
    ml_true = jnp.zeros((5,setup.nt,setup.nx,1,1))
    for ii, prime in enumerate([0,1,4]):
        ml_true = ml_true.at[prime,...].set(jnp.reshape(ml_true_buff[jnp.s_[ii,...]],(setup.nt,setup.nx,1,1)))
    
    ml_true = jnp.array([ml_true[jnp.s_[:,t,:]] for t in range(ml_true.shape[1])])
    ml_true_arr = jnp.array([ml_true[jnp.s_[seq:seq+setup.ns+1,...]] for seq in range(setup.nt-setup.ns)])
    # [ml_true_arr] = [nt-ns seq, primes, ns+1 times, nx cells]
    ml_loss_arr = vmap(_mse,in_axes=(0,0))(
        # [map over seq, all vars, from time 1 to end of seq, all cells]
        ml_pred_arr[jnp.s_[:,1:,...]],
        ml_true_arr
    )
    ml_loss_sample = jnp.mean(ml_loss_arr)

    # feed forward with numerical solver
    mc_loss_sample = 0
    if setup.mc_flag:
        coarse_num['conservatives']['convective_fluxes']['riemann_solver'] = "HLLC"
        input_reader = InputReader(coarse_case,coarse_num)
        sim_manager = SimulationManager(input_reader)
        
        #  map over times, concatenate all sequences
        mc_primes_init = jnp.concatenate(ml_pred_arr[jnp.s_[:,:-1,...]])
        
         # batch for parallel execution
        batch_idx = jnp.linspace(0,mc_primes_init.shape[0],n_dev+1)
        mc_primes_init_par = jnp.array([mc_primes_init[int(ii):int(jj),...] for ii,jj in zip(batch_idx[:-1],batch_idx[1:])])

        mc_pred_arr, _ = pmap(sim_manager.feed_forward, in_axes=(0,0,None,None,None))(
            batch_primes_init = mc_primes_init_par,
            batch_levelset_init = jnp.empty_like(mc_primes_init_par), # not needed for single-phase, but is a required arg
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
    loss_sample = ml_loss_sample + mc_loss_sample
    return loss_sample

def get_loss_batch(params: hk.Params) -> float:
    """
    looped version of _get_loss_sample

    ----- inputs -----\n
    :param params: holds parameters of the NN

    ----- returns -----\n
    :return loss_batch: average loss over batch
    """
    samples = jnp.arange(setup.batch_size)
    return jnp.mean(vmap(functools.partial(_get_loss_sample,params),in_axes=(0,))(samples))

def _evaluate_sample(params: hk.Params, *args) -> jnp.ndarray:
    """
    creates a simulation manager to fully simulate the case using the updated mcTangent
    the resulting data is then loaded and used to calculate the mse across all test data

    ----- inputs -----\n
    :param params: holds parameters of the NN
    :param args: allows for mapping input for vmap api

    ----- returns -----\n
    :return sample_err: mean squared error for the sample
    """
    # load fine simulation
    fine_sim = dat.data.next_sim()
    quantities = ['density']
    _, _, _, data_dict_fine = load_data(fine_sim.domain, quantities)
    # coarse sampling
    data_dict_coarse = {}
    for quant in quantities:
        data_fine = data_dict_fine[quant]
        data_dict_coarse[quant] = jim.resize(data_fine,(setup.nt+1,setup.nx,1,1),"linear")

    # run mcTangent simulation
    coarse_case = fine_sim.case
    coarse_num = fine_sim.numerical
    coarse_case['general']['case_name'] = 'test_mcT'
    coarse_case['general']['save_path'] = 'results'
    coarse_case['domain']['x']['cells'] = setup.nx
    coarse_num['conservatives']['time_integration']['fixed_timestep'] = setup.dt
    coarse_num['conservatives']['convective_fluxes']['riemann_solver'] = "MCTANGENT"

    ml_parameters_dict = {"riemann_solver": params}
    ml_networks_dict = hk.data_structures.to_immutable_dict({"riemann_solver": mcT_net})

    input_reader = InputReader(coarse_case,coarse_num)
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
    sample_err = _mse(data_dict_mcT['density'][:,:,0,0] - data_dict_coarse['density'][:,:,0,0])

    return sample_err

def evaluate_epoch(params: hk.Params) -> float:
    """
    vectorized form of _evaluate_sample

    ----- inputs -----\n
    :param state: holds parameters of the NN

    ----- returns -----\n
    :return epoch_err: mean squared error for the epoch
    """
    samples = jnp.arange(setup.num_test)
    return jnp.mean(vmap(functools.partial(_evaluate_sample, params), in_axes=(0,))(samples))

def Train(state: TrainingState) -> Tuple[TrainingState,TrainingState]:
    """
    Train mcTangent through end-to-end optimization in JAX-Fluids

    ----- inputs -----\n
    :param state: holds parameters of the NN and optimizer
    :param setup: holds parameters for the operation of JAX-Fluids
    :param train_coefs: define initial conditions in training
    :param test_coefs: define initial conditions in testing

    ----- returns -----\n
    :return states: tuple holding the best state by least error and the end state
    """
    min_err = 100
    epoch_min = 1
    best_state = state
    for epoch in range(setup.num_epochs):
        # reset loss
        state = TrainingState(state.params,state.opt_state,0)
        t1 = time.time()
        for batch in range(setup.num_batches):
           # get loss and grads
            loss_batch, grads = value_and_grad(get_loss_batch,allow_int=True)(state.params)

            # update mcTangent
            updates, opt_state_new = optimizer.update(grads, state.opt_state)
            params_new = optax.apply_updates(state.params, updates)
            state = TrainingState(params_new,opt_state_new,state.loss + loss_batch/setup.num_batches)
        t2 = time.time()

        test_err = evaluate_epoch(state.params)
        
        if test_err <= min_err:
            min_err = test_err
            epoch_min = epoch
            best_state = state

        if epoch % 10 == 0:  # Print every 10 epochs
            print("time {:.2e}s loss {:.2e} TE {:.2e}  TE_min {:.2e} EPmin {:d} EP {} ".format(
                    t2 - t1, state.loss, test_err, min_err, epoch_min, epoch))

        dat.data.check_sims()
        wandb.log({"Train loss": float(state.loss), "Test Error": float(test_err), 'TEST MIN': float(min_err), 'Epoch' : float(epoch)})
    return best_state, state

# data input will be mean(primes_L, primes_R) -> [5,(nx+1),1,1]
mcT_net = hk.without_apply_rng(hk.transform(mcT_fn))

data_init = jnp.empty((5,setup.nx+1,1,1))
optimizer = optax.adam(setup.learning_rate)
initial_params = mcT_net.init(jrand.PRNGKey(0), data_init)
initial_opt_state = optimizer.init(initial_params)
state = TrainingState(initial_params, initial_opt_state, 0)

best_state, end_state = Train(state)

# save params
param_path = "network/parameters"
save_params(best_state.params,os.path.join(param_path,"best"))
save_params(end_state.params,os.path.join(param_path,"end"))

# %% visualize best and end state

# fine
fine_sim = dat.data.next_sim()

quantities = ['density']
x_fine, _, times, data_dict_fine = load_data(fine_sim.domain, quantities)

# coarse
coarse_case = fine_sim.case
coarse_num = fine_sim.numerical
coarse_case['domain']['x']['cells'] = setup.dx
coarse_num['conservatives']['time_integration']['fixed_timestep'] = setup.dt
input_reader = InputReader(coarse_case,coarse_num)
initializer = Initializer(input_reader)
sim_manager = SimulationManager(input_reader)
buffer_dictionary = initializer.initialization()
sim_manager.simulate(buffer_dictionary)

path = sim_manager.output_writer.save_path_domain
quantities = ['density']
x_coarse, _, _, data_dict_coarse = load_data(path, quantities)

# best state
coarse_num['conservatives']['convective_fluxes']['riemann_solver'] = "MCTANGENT"

params_best = load_params(os.path.join(param_path,"best"))
params_end = load_params(os.path.join(param_path,"end"))

ml_parameters_dict = {"riemann_solver":params_best}
ml_networks_dict = hk.data_structures.to_immutable_dict({"riemannsolver": mcT_net})

input_reader = InputReader(coarse_case,coarse_num)
initializer = Initializer(input_reader)
sim_manager = SimulationManager(input_reader)
buffer_dictionary = initializer.initialization()
buffer_dictionary['machinelearning_modules'] = {
    'ml_parameters_dict': ml_parameters_dict,
    'ml_networks_dict': ml_networks_dict
}
sim_manager.simulate(buffer_dictionary)

path = sim_manager.output_writer.save_path_domain
_, _, _, data_dict_best = load_data(path, quantities)

# end state
ml_parameters_dict = {"riemann_solver":params_end}
buffer_dictionary['machinelearning_modules']['ml_parameters_dict'] = ml_parameters_dict
sim_manager.simulate(buffer_dictionary)

path = sim_manager.output_writer.save_path_domain
_, _, _, data_dict_end = load_data(path, quantities)

data_true = data_dict_fine['density']
data_coarse = data_dict_coarse['density']
data_best = data_dict_best['density']
data_end = data_dict_end['density']

n_plot = 3
plot_steps = np.linspace(0,setup.nt,n_plot,dtype=int)
plot_times = times[plot_steps]

fig = plt.figure(figsize=(32,10))
for nn in range(n_plot):
    ut = jnp.reshape(data_true[plot_steps[nn], :], (4*setup.nx, 1))
    uc = jnp.reshape(data_coarse[plot_steps[nn], :], (setup.nx, 1))
    ub = jnp.reshape(data_best[plot_steps[nn], :], (setup.nx, 1))
    ue = jnp.reshape(data_end[plot_steps[nn], :], (setup.nx, 1))
    ax = fig.add_subplot(1, n_plot, nn+1)
    l1 = ax.plot(x_fine, ut, '-', linewidth=2, label='True')
    l2 = ax.plot(x_coarse, uc, '--', linewidth=2, label='Coarse')
    l2 = ax.plot(x_coarse, ub, '--', linewidth=2, label='Predicted')
    l2 = ax.plot(x_coarse, ue, '--', linewidth=2, label='Predicted')
    ax.set_aspect('auto', adjustable='box')
    ax.set_title('t = ' + str(plot_times[nn]))

    if nn == 0:
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center')

plt.show()
fig.savefig(os.path.join('figs',setup.case_name+'.png'))