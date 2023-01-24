from typing import Tuple, NamedTuple, Optional, Iterable, Union
import time, os, wandb, sys
import pandas as pd
import shutil
import functools
from functools import partial
import numpy as np
import re
import matplotlib.pyplot as plt
import json
import memray as mra

import jax
import jax.numpy as jnp
import jax.random as jrand
import jax.image as jim
import jax.profiler as jprof
import jax.tree_util as jtr
from jax import value_and_grad, vmap, jit, lax, pmap
import pickle
import haiku as hk
import optax
from jaxfluids import InputReader, Initializer, SimulationManager
from jaxfluids.post_process import load_data

import mcT_adv_setup as setup
import mcT_adv_data as dat
from mcT_adv_data import Sim

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
results_path = setup.proj('results')
test_path = setup.proj('test')
param_path = setup.proj("network/parameters")

# %% create mcTangent network and training functions
class TrainingState(NamedTuple):
    params: Iterable[hk.Params]
    opt_state: Iterable[optax.OptState]
    loss: float

from mcT_adv_setup import save_params, load_params, compare_params, mse, net, optimizer

mse = setup.mse


@jit
def get_coarse(data_fine: jnp.ndarray) -> jnp.ndarray:
    """
    down samples the data by factor of 4 for use in training.

    ----- inputs -----\n
    :param data_fine: data to be downsampled, of shape [times, fxs(, fyz, fzs)]
    :param seed: Optional, defines the rng if noise is added

    ----- returns -----\n
    :return data_coarse: downsampled data array, of shape [times, cxs(, cyz, czs)]
    """
    data_coarse = jim.resize(data_fine,(data_fine.shape[0],setup.nx,1,1),"linear")
    return data_coarse

@partial(jit,static_argnums=[0])
def _add_noise(noise_level: float, arr: jnp.ndarray, seed: int):
    noise_arr = jrand.normal(jrand.PRNGKey(seed),arr.shape)
    noise_arr *= noise_level/jnp.max(noise_arr)
    return arr * (1+noise_arr)

def _partial_add_noise(arr: jnp.ndarray, seed: int):
    return partial(_add_noise,setup.noise_level)(arr,seed)

@jit
def add_noise(data: jnp.ndarray, seed: Optional[int] = 1):
    seed_arr = jrand.randint(jrand.PRNGKey(seed),(5,),1,100)
    data_noisy = vmap(_partial_add_noise,in_axes=(0,0))(data,seed_arr)
    return data_noisy

def get_par_batch(serial):
    n_dev = jax.local_device_count()
    # print('batching for %s devices' %n_dev)
    if isinstance(serial, jnp.ndarray):
        return jnp.array_split(serial, n_dev)
    else:
        return jax.tree_map(lambda x: jnp.array([x] * n_dev), serial)

# def smart_feed_forward():
#     """
#     Rebatches the initial condition
#     """
#     ml_pred_arr, _ = sim_manager.feed_forward(
#         ml_primes_init,
#         setup.ns+1,
#         coarse_case['general']['save_dt'],
#         0, 1,
#         ml_parameters_dict,
#         ml_networks_dict
#     )

def get_rseqs(k_pred: jnp.ndarray) -> jnp.ndarray:
    """
    Resequences kth ML prediction sequence for every Rth prediction for MC loss

    ----- inputs -----\n
    :param k_pred: kth ML prediction sequence, of shape [ns+1+nr, 5, xs(, ys, zs)]

    ----- returns -----\n
    :return rseqs: resequenced prediction, of shape [ns+nr, nr, 5, xs(, ys, zs)]
    """
    n_rseq = setup.ns+2
    rseqs = jnp.zeros((n_rseq,setup.nr,5,setup.nx,1,1))
    for i in range(n_rseq):
        rseq_i = lax.dynamic_slice_in_dim(k_pred,i,setup.nr)
        rseqs = rseqs.at[i].set(rseq_i)
    return rseqs

def get_loss_batch(params: hk.Params, batch:jnp.ndarray, sim: dat.Sim, seed: int) -> float:
    """
    Uses a highly resolved simulation as ground truth to calculate loss over a sample
    
    ----- inputs -----\n
    :param params: holds parameters of the NN
    :param batch: training data for one batch, of shape [sample, sequences, primes, timesteps, xs(, ys, zs)]
    :param sim: structure containing information about the truth simulations
    :param seed: seed number, used as an rng seed for noise

    ----- returns -----\n
    :return loss_batch: average loss over all sequences in the batch
    """
    # feed forward with mcTangent ns+1 steps
    coarse_case = sim.case
    coarse_num = sim.numerical
    coarse_case['domain']['x']['cells'] = setup.nx
    coarse_num['conservatives']['time_integration']['fixed_timestep'] = setup.dt
    coarse_num['conservatives']['time_integration']['time_integrator'] = setup.integrator
    coarse_num['conservatives']['convective_fluxes']['riemann_solver'] = "MCTANGENT"

    ml_parameters_dict = {"MCTANGENT":params}
    ml_networks_dict = hk.data_structures.to_immutable_dict({"MCTANGENT": net})

    input_reader = InputReader(coarse_case,coarse_num)
    sim_manager = SimulationManager(input_reader)
    
    # concatenate all samples as one batch
    sample = jnp.concatenate(batch,axis=0)
    ml_primes_init = sample[:,:,0,...]
    # ml_primes_init = jnp.moveaxis(ml_primes_init,2,1)
    if setup.noise_flag:
        ml_primes_init = vmap(add_noise, in_axes=(0,None))(ml_primes_init,seed)


    # feed_forward_batch = vmap(sim_manager.feed_forward, in_axes=(0,None,None,None,None,None,None,None), out_axes=(0,0))
    ml_pred_arr, _ = sim_manager.feed_forward(
        ml_primes_init,
        None, # not needed for single-phase, but is a required arg for feed_forward
        setup.ns+1,
        coarse_case['general']['save_dt'],
        0.0, 1,
        ml_parameters_dict,
        ml_networks_dict
    )
    ml_pred_arr = jnp.array(ml_pred_arr[1:])
    ml_pred_arr = jnp.moveaxis(ml_pred_arr,0,2)

    # ml loss with density only
    ml_loss_sample = mse(ml_pred_arr[:,0,...], sample[:,0,1:,...])

    if not setup.mc_flag or setup.nr < 1:
        return ml_loss_sample
    
    # mc loss
    # ff R additional steps
    ml_primes_init_R = ml_pred_arr[:,:,-1,...]

    ml_pred_arr_R, _ = sim_manager.feed_forward(
        ml_primes_init_R,
        None, # not needed for single-phase, but is a required arg for feed_forward
        setup.nr,
        coarse_case['general']['save_dt'],
        0.0, 1,
        ml_parameters_dict,
        ml_networks_dict
    )
    ml_pred_arr_R = jnp.array(ml_pred_arr_R[1:])
    ml_pred_arr_R = jnp.moveaxis(ml_pred_arr_R,0,2)
    ml_pred_arr_R = jnp.concatenate((ml_pred_arr,ml_pred_arr_R),2)
    ml_pred_arr_R = jnp.swapaxes(ml_pred_arr_R,1,2)
    
    # resequence for each R seq
    ml_rseqs = jnp.concatenate(vmap(get_rseqs,in_axes=(0,))(ml_pred_arr_R))

    coarse_num['conservatives']['convective_fluxes']['riemann_solver'] = "HLLC"
    input_reader = InputReader(coarse_case,coarse_num)
    sim_manager = SimulationManager(input_reader)
    
    #  map over times, concatenate all sequences
    mc_primes_init = jnp.concatenate(jnp.concatenate(
        (jnp.swapaxes(jnp.reshape(ml_primes_init,(setup.nt-setup.ns-1,5,1,setup.nx,1,1)),1,2),
        jnp.swapaxes(ml_pred_arr,1,2)),axis=1))

    mc_pred_arr, _ = sim_manager.feed_forward(
        mc_primes_init,
        None, # not needed for single-phase, but is a required arg for feed_forward
        setup.nr,
        coarse_case['general']['save_dt'],
        0.0, 1,
        ml_parameters_dict,
        ml_networks_dict
    )
    mc_pred_arr = jnp.array(mc_pred_arr[1:])
    mc_pred_arr = jnp.nan_to_num(mc_pred_arr)
    mc_pred_arr = jnp.moveaxis(mc_pred_arr,0,1)
    
    # mc loss with rho only
    mc_loss_sample = setup.mc_alpha/setup.nr * mse(ml_rseqs[:,0,...],mc_pred_arr[:,0,...])
    loss_sample = ml_loss_sample + mc_loss_sample
    return loss_sample

# @jit
def _evaluate_sample(params: hk.Params, sample: jnp.ndarray, sim: dat.Sim) -> jnp.ndarray:
    """
    creates a simulation manager to fully simulate the case using the updated mcTangent
    the resulting data is then loaded and used to calculate the mse across all test data

    ----- inputs -----\n
    :param params: holds parameters of the NN
    :param sample: sample of data to be used for validation, of shape [primes, sequences, timesteps, xs]
    :param sim: contains simulation data

    ----- returns -----\n
    :return err_sample: mean squared error for the sample
    :return err_hist_sample: mean squared error for every timestep
    :return merr_sample: mean squared error of governing model over the entire sample
    :return merr_hist_sample: mean squared error of governing model for every timestep
    """
    # feed forward with mcTangent ns+1 steps
    coarse_case = sim.case
    coarse_num = sim.numerical
    coarse_case['general']['case_name'] = 'test_mcT'
    coarse_case['general']['save_path'] = test_path
    coarse_case['domain']['x']['cells'] = setup.nx
    coarse_num['conservatives']['time_integration']['fixed_timestep'] = setup.dt
    coarse_num['conservatives']['time_integration']['time_integrator'] = setup.integrator
    coarse_num['conservatives']['convective_fluxes']['riemann_solver'] = "MCTANGENT"

    ml_parameters_dict = {"MCTANGENT": params}
    ml_networks_dict = hk.data_structures.to_immutable_dict({"MCTANGENT": net})

    input_reader = InputReader(coarse_case,coarse_num)
    sim_manager = SimulationManager(input_reader)

    ml_primes_init = sample[:,0,...]
    ml_primes_init = jnp.reshape(ml_primes_init,(1,5,setup.nx,1,1))
    # ml_primes_init = jnp.moveaxis(ml_primes_init,2,1)

    # feed_forward_batch = vmap(sim_manager.feed_forward, in_axes=(0,None,None,None,None,None,None,None), out_axes=(0,0))
    ml_pred_arr, _ = sim_manager.feed_forward(
        ml_primes_init,
        None, # not needed for single-phase, but is a required arg for feed_forward
        int(float(coarse_case['general']['end_time']) / coarse_case['general']['save_dt']),
        coarse_case['general']['save_dt'],
        0.0, 1,
        ml_parameters_dict,
        ml_networks_dict
    )
    ml_pred_arr = jnp.array(ml_pred_arr[1:])
    ml_pred_arr = jnp.nan_to_num(ml_pred_arr)
    ml_pred_arr = jnp.moveaxis(ml_pred_arr,0,2)
    ml_pred_arr = jnp.reshape(ml_pred_arr, sample[:,1:,...].shape)

    # err_sample = mse(ml_pred_arr, sample[:,1:,...])
    err_hist_sample = jnp.array(vmap(mse, in_axes=(1,1))(ml_pred_arr, sample[:,1:,...]))

    coarse_num['conservatives']['convective_fluxes']['riemann_solver'] = "HLLC"

    input_reader = InputReader(coarse_case,coarse_num)
    sim_manager = SimulationManager(input_reader)

    mc_pred_arr, _ = sim_manager.feed_forward(
        ml_primes_init,
        None, # not needed for single-phase, but is a required arg for feed_forward
        int(float(coarse_case['general']['end_time']) / coarse_case['general']['save_dt']),
        coarse_case['general']['save_dt'],
        0.0, 1,
        None,
        None
    )
    mc_pred_arr = jnp.array(mc_pred_arr[1:])
    mc_pred_arr = jnp.nan_to_num(mc_pred_arr)
    mc_pred_arr = jnp.moveaxis(mc_pred_arr,0,2)
    mc_pred_arr = jnp.reshape(mc_pred_arr, sample[:,1:,...].shape)

    merr_hist_sample = jnp.array(vmap(mse, in_axes=(1,1))(mc_pred_arr, sample[:,1:,...]))

    return jnp.mean(err_hist_sample), err_hist_sample, jnp.mean(merr_hist_sample), merr_hist_sample

# @jit
def evaluate(params: hk.Params, data: jnp.ndarray) -> float:
    """
    looped form of _evaluate_sample

    ----- inputs -----\n
    :param state: holds parameters of the NN
    :param data: downsampled testing data, of shape [samples, primes, timesteps, xs(, ys, zs)]


    ----- returns -----\n
    :return err_epoch: mean squared error for the epoch
    :return err_hist: mean squared error for every time step of the predicted trajectories
    :return merr_epoch: mean squared error between model trajectory and truth
    :return merr_hist: mean squared error of governing model for every time step
    """
    err_epoch = 0
    merr_epoch = 0
    err_hist = jnp.zeros(data.shape[2]-1)
    merr_hist = jnp.zeros(data.shape[2]-1)
    sims = [dat.data.next_sim() for _ in range(setup.num_test)]
    for sample, sim in zip(data, sims):
        err_sample, err_hist_sample, merr_sample, merr_hist_sample = _evaluate_sample(params,sample,sim)
        err_epoch += err_sample/setup.num_test
        err_hist += err_hist_sample/setup.num_test
        merr_epoch += merr_sample/setup.num_test
        merr_hist += merr_hist_sample/setup.num_test

    # err_epoch = vmap(_evaluate_sample, in_axes=(None,0))(params, data)
    if not jnp.isnan(err_epoch) and not jnp.isinf(err_epoch):
        pass
    else:
        err_epoch = jnp.array(sys.float_info.max)
    
    return err_epoch, err_hist, merr_epoch, merr_hist 

# @partial(pmap, axis_name='data', in_axes=(0,0,0))
# def update_par(params: hk.Params, opt_state: optax.OptState, data: jnp.ndarray) -> Tuple:
#     """
#     Evaluates network loss and gradients
#     Applies optimizer updates and returns the new parameters, state, and loss

#     ----- inputs -----\n
#     :param params: current network params
#     :param opt_state: current optimizer state
#     :param data: array of sequenced training data, of shape [samples, primes, sequences, timesteps, xs]

#     ----- returns -----\n
#     :return state: tuple of arrays containing updated params, optimizer state, as loss
#     """

#     loss_batch, grads_batch = value_and_grad(get_loss_batch, argnums=0, allow_int=True)(params, data)

#     grads = lax.pmean(grads_batch, axis_name='data')
#     loss = lax.pmean(loss_batch, axis_name='data')

#     updates, opt_state_new = optimizer.update(grads, opt_state)
#     params_new = optax.apply_updates(params, updates)
#     # params_new = jax.tree_map(lambda p, g: optax.apply_updates(p,g), params, updates)

#     return params_new, opt_state_new, loss

def update(params: Iterable[hk.Params], opt_state: Iterable[optax.OptState], data: jnp.ndarray) -> Tuple:
    """
    Evaluates network loss and gradients
    Applies optimizer updates and returns the new parameters, state, and loss

    ----- inputs -----\n
    :param params: current network params
    :param opt_state: current optimizer state
    :param data: array of sequenced training data, of shape [samples, primes, sequences, timesteps, xs(, ys, zs)]

    ----- returns -----\n
    :return state: tuple of arrays containing updated params, optimizer state, as loss
    """
    # loop through data to lower memory cost
    loss = 0
    grads = jax.tree_map(jnp.zeros_like, params)
    sims = [dat.data.next_sim() for _ in range(setup.num_train)]
    for ii in range(setup.num_batches):
        batch = lax.dynamic_slice_in_dim(data,ii*setup.batch_size,setup.batch_size)
        batch = jnp.swapaxes(batch,1,2)
        loss_batch = 0
        grad_batch = jax.tree_map(jnp.zeros_like, params)
        sim_batch_index = lax.dynamic_slice_in_dim(jnp.arange(setup.num_train),ii*setup.batch_size,setup.batch_size)
        sims_batch = sims[sim_batch_index[0]:sim_batch_index[-1]+1]

        # all sims have the same setup, initial condition is taken from the data
        loss_batch, grad_batch = value_and_grad(get_loss_batch, argnums=0)(
            params, batch, sims_batch[0], ii
            )
        grad_batch = jax.tree_map(jnp.nan_to_num, grad_batch)

        # Small Batch
        if setup.small_batch:
            opt_state.hyperparams['f'] = loss_batch
            updates, opt_state = jit(optimizer.update)(grad_batch, opt_state)
            params = jit(optax.apply_updates)(params, updates)
        loss += loss_batch/setup.num_batches
        grads = jax.tree_map(lambda g1, g2: g1+g2/setup.num_batches, grads, grad_batch)
    # Large Batch
    if not setup.small_batch:
        opt_state.hyperparams['f'] = loss
        updates, opt_state = jit(optimizer.update)(grads, opt_state)
        params = jit(optax.apply_updates)(params, updates)

    return params, opt_state, loss

def Train(state: TrainingState, data_test: np.ndarray, data_train: np.ndarray) -> Tuple[TrainingState,TrainingState]:
    """
    Train mcTangent through end-to-end optimization in JAX-Fluids

    ----- inputs -----\n
    :param state: holds parameters of the NN and optimizer
    :param setup: holds parameters for the operation of JAX-Fluids
    :param data_test: data for testing, of shape [samples, primes, times, xs(, ys, zs)]
    :param data_train: data for training, of shape [samples, primes, times, xs(, ys, zs)]

    ----- returns -----\n
    :return states: tuple holding the best state by least error and the end state
    """
    min_err = sys.float_info.max
    epoch_min = -1
    best_state = state
    err_hist_list = []
    err_hist_df = pd.DataFrame(data = {'Time': jnp.linspace(setup.dt,int(setup.t_max*setup.test_ratio),int(setup.nt*setup.test_ratio))})
    for epoch in range(setup.last_epoch,setup.num_epochs):
        # reset each epoch
        state = TrainingState(state.params,state.opt_state,0)
        dat.data.check_sims()
        
        train_coarse = jit(vmap(jit(vmap(get_coarse, in_axes=(0,))),in_axes=(0,)))(data_train)
        test_coarse = jit(vmap(jit(vmap(get_coarse, in_axes=(0,))),in_axes=(0,)))(data_test)

        train_seq = jnp.array([train_coarse[:,:, ii:(ii+setup.ns+2), ...] for ii in range(setup.nt-setup.ns-1)])
        train_seq = jnp.moveaxis(train_seq,0,2)
        del train_coarse

        t1 = time.time()

        params_new, opt_state_new, loss_new = update(state.params, state.opt_state, train_seq)
        state = TrainingState(params_new,opt_state_new,loss_new)
        test_err, err_hist, merr, merr_hist = evaluate(state.params, test_coarse)

        t2 = time.time()

        # save in case job is canceled, can resume
        last_param_path = os.path.join(param_path,'last.pkl')
        if os.path.exists(last_param_path):
            os.remove(last_param_path)
        save_params(state.params,last_param_path)
        
        if test_err <= min_err:
            min_err = test_err
            epoch_min = epoch
            best_state = state

        if epoch % 10 == 0:  # Print every 10 epochs
            print("time {:.2e}s loss {:.2e} TE {:.2e}  TE_min {:.2e} EPmin {:d} EP {:d} ".format(
                    t2 - t1, state.loss, float(test_err), float(min_err), epoch_min, epoch))
        
        if epoch == 0:  # Profile for memory monitoring
            jprof.save_device_memory_profile(f"memory/memory_{epoch}.prof") 
        
        if True:  # Clear every 1 epochs
            jax.clear_backends()

        
        dat.data.check_sims()
        err_hist_df[f"MCT_err_ep{epoch}_{setup.ns}s{setup.nr}r"] =  err_hist
        if epoch == setup.last_epoch:
            err_hist_df[f"HLLC_err"] =  merr_hist
        else:
            del merr_hist
        # pd.concat(axis=1)
        err_hist_table = wandb.Table(data=err_hist_df)
        weight_arr = jax.device_get(state.params[0]['mc_t_net_dense/~_create_net/linear']['w'])
        weight_im = wandb.Image(weight_arr,caption='Linear Density Weights')
        wandb.log({
            "Train loss": float(state.loss),
            "Test Error": float(test_err),
            "Test Min": float(min_err),
            "Model Error": float(merr),
            "Epoch": float(epoch),
            "Error History Table": err_hist_table,
            "Weight Matrix": weight_im
            })
        
    return best_state, state

def run_simulation(case_dict,num_dict,params=None,net=None):
    input_reader = InputReader(case_dict,num_dict)
    initializer = Initializer(input_reader)
    sim_manager = SimulationManager(input_reader)
    buffer_dictionary = initializer.initialization()
    buffer_dictionary['machinelearning_modules'] = {
        'ml_parameters_dict': params,
        'ml_networks_dict': net
    }
    sim_manager.simulate(buffer_dictionary)
    return sim_manager

def visualize():
    import matplotlib.pyplot as plt
    import pandas as pd
    import matplotlib
    import matplotlib.pylab as pylab
    from matplotlib.lines import Line2D
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm  # Colour map
    import matplotlib.animation as animatio
    import matplotlib.font_manager
    params = {'legend.fontsize': 14, 'axes.labelsize': 16, 'axes.titlesize': 20, 'xtick.labelsize': 14, 'ytick.labelsize': 14}
    pylab.rcParams.update(params)
    matplotlib.rcParams['mathtext.fontset'] = 'cm'
    matplotlib.rcParams['mathtext.rm'] = 'serif'
    plt.rcParams['font.family'] = 'TimesNewRoman'
    # date
    now = time.strftime("%d%m%y%H%M")
    # fine
    fine_sim = dat.data.next_sim()

    quantities = ['density']
    x_fine, _, times, data_dict_fine = fine_sim.load(quantities,dict)
    x_fine = x_fine[0]

    # coarse
    coarse_case = fine_sim.case
    coarse_num = fine_sim.numerical
    coarse_case['general']['save_path'] = results_path
    coarse_case['domain']['x']['cells'] = setup.nx
    coarse_num['conservatives']['time_integration']['fixed_timestep'] = setup.dt
    sim_manager = run_simulation(coarse_case,coarse_num)

    path = sim_manager.output_writer.save_path_domain
    quantities = ['density']
    x_coarse, _, _, data_dict_coarse = load_data(path, quantities)
    x_coarse = x_coarse[0]

    # mcTangent
    coarse_num['conservatives']['convective_fluxes']['riemann_solver'] = "MCTANGENT"

    # best state, end state
    params_best = load_params(os.path.join(param_path,"best.pkl"))
    params_end = load_params(os.path.join(param_path,"end.pkl"))

    ml_parameters_dict = {"MCTANGENT":params_best}
    ml_networks_dict = hk.data_structures.to_immutable_dict({"MCTANGENT": net})

    sim_manager = run_simulation(coarse_case,coarse_num,ml_parameters_dict,ml_networks_dict)

    path = sim_manager.output_writer.save_path_domain
    _, _, _, data_dict_best = load_data(path, quantities)

    ml_parameters_dict = {"MCTANGENT":params_end}
    sim_manager = run_simulation(coarse_case,coarse_num,ml_parameters_dict,ml_networks_dict)

    path = sim_manager.output_writer.save_path_domain
    _, _, _, data_dict_end = load_data(path, quantities)

    data_true = data_dict_fine['density']
    data_coarse = data_dict_coarse['density']
    data_best = data_dict_best['density']
    data_end = data_dict_end['density']

    n_plot = 3
    plot_steps = np.linspace(0,data_true.shape[0]-1,n_plot,dtype=int)
    plot_times = times[plot_steps]

    fig = plt.figure(figsize=(32,10))
    for nn in range(n_plot):
        ut = jnp.reshape(data_true[plot_steps[nn]], (setup.nx_fine,))
        uc = jnp.reshape(data_coarse[plot_steps[nn]], (setup.nx,))
        uc = jnp.nan_to_num(uc)
        ub = jnp.reshape(data_best[plot_steps[nn]], (setup.nx,))
        ub = jnp.nan_to_num(ub)
        ue = jnp.reshape(data_end[plot_steps[nn]], (setup.nx,))
        ue = jnp.nan_to_num(ue)
        
        ax = fig.add_subplot(1, n_plot, nn+1)
        l1 = ax.plot(x_fine, ut, '-o', linewidth=2, markevery=0.2, label='True')
        l2 = ax.plot(x_coarse, uc, '--^', linewidth=2, markevery=(0.05,0.2), label='Coarse')
        l2 = ax.plot(x_coarse, ub, '--s', linewidth=2, markevery=(0.10,0.2), label='Best')
        l2 = ax.plot(x_coarse, ue, '--p', linewidth=2, markevery=(0.15,0.2), label='End')

        ax.set_aspect('auto', adjustable='box')
        ax.set_title('t = ' + str(plot_times[nn]))

        if nn == 0:
            handles, labels = ax.get_legend_handles_labels()
            fig.legend(handles, labels)

    # plt.show()
    fig.savefig(os.path.join('figs',setup.case_name+now+'.png'))

    fig = plt.figure()
    coarse_true = get_coarse(data_true)
    err_coarse = vmap(mse,in_axes=(0,0))(coarse_true,data_coarse)
    err_best = vmap(mse,in_axes=(0,0))(coarse_true,data_best)
    err_end = vmap(mse,in_axes=(0,0))(coarse_true,data_end)
    plt.plot(times,err_coarse, '--^', linewidth=2, markevery=0.2, label='Coarse')
    plt.plot(times,err_best, '--s', linewidth=2, markevery=(0.06,0.2), label='Best')
    plt.plot(times,err_end, '--p', linewidth=2, markevery=(0.13,0.2), label='End')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('MSE')
    plt.title('Error Over Time')

    # plt.show()
    fig.savefig(os.path.join('figs',setup.case_name+'_errHist_' + now +'.png'))

# %% main
if __name__ == "__main__":
    # os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

    # data input will be rho(x,t)
    u_init = jnp.zeros((1,setup.nx,setup.ny,setup.nz))
    initial_params = net.init(jrand.PRNGKey(setup.num_epochs), u_init)
    del u_init
    if setup.load_warm or setup.load_last:
        # loads warm params, always uses last.pkl over warm.pkl if available and toggled on
        if setup.load_last and os.path.exists(os.path.join(param_path,'last.pkl')):
            last_params = load_params(param_path,'last.pkl')
            print("\n"+"-"*5+"Using Last Params"+"-"*5+"\n")
            initial_params = last_params
            del last_params
        elif setup.load_warm and os.path.exists(os.path.join(param_path,'warm.pkl')):
            warm_params = load_params(param_path,'warm.pkl')
            print("\n"+"-"*5+"Using Warm-Start Params"+"-"*5+"\n")
            initial_params = warm_params
            del warm_params

    initial_opt_state = jit(optimizer.init)(initial_params)

    state = TrainingState(initial_params, initial_opt_state, 0)

    data_train, data_test = dat.data.load_all()

    # transfer to CPU
    # data_test = jax.device_get(data_test)
    # data_train = jax.device_get(data_train)

    # transfer to GPU
    data_test = jax.device_put(data_test)
    data_train = jax.device_put(data_train)

    best_state, end_state = Train(state, data_test, data_train)

    # save params
    save_params(best_state.params,os.path.join(param_path,"best.pkl"))
    save_params(end_state.params,os.path.join(param_path,"end.pkl"))

    # # %% visualize best and end state
    if setup.vis_flag:
        visualize()
