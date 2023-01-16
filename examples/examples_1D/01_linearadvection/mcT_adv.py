from typing import Tuple, NamedTuple, Optional, Iterable, Union
import time, os, wandb, sys
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
    params: hk.Params
    opt_state: optax.OptState
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

def get_loss_sample(params: hk.Params, sample:jnp.ndarray, sim: dat.Sim, seed: int) -> float:
    """
    Uses a highly resolved simulation as ground truth to calculate loss over a sample
    
    ----- inputs -----\n
    :param params: holds parameters of the NN
    :param sample: training data for one sample, of shape [sequences, primes, timesteps, xs(, ys, zs)]
    :param sim: contains information about the truth simulation
    :param seed: seed number, used as an rng seed for noise

    ----- returns -----\n
    :return loss_sample: average loss over all sequences in the sample
    """
    # feed forward with mcTangent ns+1 steps
    coarse_case = sim.case
    coarse_num = sim.numerical
    coarse_case['domain']['x']['cells'] = setup.nx
    coarse_num['conservatives']['time_integration']['fixed_timestep'] = setup.dt
    coarse_num['conservatives']['convective_fluxes']['riemann_solver'] = "MCTANGENT"

    ml_parameters_dict = {"riemann_solver":params}
    ml_networks_dict = hk.data_structures.to_immutable_dict({"riemann_solver": net})

    input_reader = InputReader(coarse_case,coarse_num)
    sim_manager = SimulationManager(input_reader)

    ml_primes_init = sample[:,:,0,...]
    # ml_primes_init = jnp.moveaxis(ml_primes_init,2,1)
    if setup.noise_flag:
        ml_primes_init = vmap(add_noise, in_axes=(0,None))(ml_primes_init,seed)


    # feed_forward_sample = vmap(sim_manager.feed_forward, in_axes=(0,None,None,None,None,None,None,None), out_axes=(0,0))
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

    # ml loss
    ml_loss_sample = mse(ml_pred_arr, sample[:,:,1:,...])

    if not setup.mc_flag:
        return ml_loss_sample
    
    # mc loss
    # ff R additional steps
    min_rho = jnp.min(sample[0,0,0,...])
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

    # enforce realistic bounds
    # jax.config.update("jax_disable_jit", True)

    mc_primes_init = mc_primes_init.at[:,0,...].set(jnp.where(mc_primes_init[:,0,...]<min_rho,
                                                    jnp.full_like(mc_primes_init[:,0,...],min_rho),
                                                    mc_primes_init[:,0,...]))
    mc_primes_init = mc_primes_init.at[:,1,...].set(jnp.ones_like(mc_primes_init[:,1,...]))
    mc_primes_init = mc_primes_init.at[:,2:3,...].set(jnp.zeros_like(mc_primes_init[:,2:3,...]))
    mc_primes_init = mc_primes_init.at[:,4,...].set(jnp.ones_like(mc_primes_init[:,4,...]))
    # for ii, primes in enumerate(mc_primes_init):
    #     plt.plot(jnp.linspace(*coarse_case['domain']['x']['range'],num=coarse_case['domain']['x']['cells']),jax.device_get(jnp.concatenate(primes[0],axis=None)).primal)
    #     plt.title(f"{ii+1} / {mc_primes_init.shape[0]}")
    #     plt.show()
    # print(jnp.sum(mc_primes_init[:,0,...].primal<=0))
    # print(jnp.sum(mc_primes_init[:,2:3,...].primal!=0))
    # print(jnp.sum(mc_primes_init[:,4,...].primal<=0))
    # jax.config.update("jax_disable_jit", False)


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
    # mc_rseqs = vmap(get_rseqs,in_axes=(0,))(mc_pred_arr)

    mc_loss_sample = setup.mc_alpha/setup.nr * mse(ml_rseqs,mc_pred_arr)
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
    """
    # feed forward with mcTangent ns+1 steps
    coarse_case = sim.case
    coarse_num = sim.numerical
    coarse_case['general']['case_name'] = 'test_mcT'
    coarse_case['general']['save_path'] = test_path
    coarse_case['domain']['x']['cells'] = setup.nx
    coarse_num['conservatives']['time_integration']['fixed_timestep'] = setup.dt
    coarse_num['conservatives']['convective_fluxes']['riemann_solver'] = "MCTANGENT"

    ml_parameters_dict = {"riemann_solver": params}
    ml_networks_dict = hk.data_structures.to_immutable_dict({"riemann_solver": net})

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

    # ml loss
    ml_pred_arr = jnp.reshape(ml_pred_arr, sample[:,1:,...].shape)
    # err_sample = mse(ml_pred_arr, sample[:,1:,...])
    err_hist_sample = jnp.array(vmap(mse, in_axes=(1,1))(ml_pred_arr, sample[:,1:,...]))

    return jnp.mean(err_hist_sample), err_hist_sample

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
    """
    err_epoch = 0
    err_hist = jnp.zeros(data.shape[2]-1)
    sims = [dat.data.next_sim() for _ in range(setup.num_test)]
    for sample, sim in zip(data, sims):
        err_sample, err_hist_sample = _evaluate_sample(params,sample,sim)
        err_epoch += err_sample/setup.num_test
        err_hist += err_hist_sample/setup.num_test

    # err_epoch = vmap(_evaluate_sample, in_axes=(None,0))(params, data)
    if not jnp.isnan(err_epoch) and not jnp.isinf(err_epoch):
        pass
    else:
        err_epoch = jnp.array(sys.float_info.max)
    return err_epoch, err_hist 

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

@jit
def cumulate(loss: float, loss_new: float, grads: dict, grads_new: dict, batch_size: int):
    loss += loss_new/batch_size
    for layer in grads_new:
        if layer not in grads.keys():
            grads[layer] = {}
        for wb in grads_new[layer]:
            if wb not in grads[layer].keys():
                grads[layer][wb] = jnp.zeros_like(grads_new[layer][wb])
            grads[layer][wb] += grads_new[layer][wb]/batch_size
    return loss, grads

def update(params: hk.Params, opt_state: optax.OptState, data: jnp.ndarray) -> Tuple:
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
    grads = {}
    sims = [dat.data.next_sim() for _ in range(setup.num_train)]
    for ii in range(setup.num_batches):
        batch = lax.dynamic_slice_in_dim(data,ii*setup.batch_size,setup.batch_size)
        batch = jnp.swapaxes(batch,1,2)
        loss_batch = 0
        grad_batch = {}
        sim_batch_index = lax.dynamic_slice_in_dim(jnp.arange(setup.num_train),ii*setup.batch_size,setup.batch_size)
        sims_batch = sims[sim_batch_index[0]:sim_batch_index[-1]+1]

        for jj, (sample, sim) in enumerate(zip(batch, sims_batch)):
            loss_sample, grad_sample = value_and_grad(get_loss_sample, argnums=0)(params, sample, sim, 1+jj+ii*setup.batch_size)
            for layer in grad_sample.keys():
                for wb in grad_sample[layer].keys():
                    grad_sample[layer][wb] = jnp.nan_to_num(grad_sample[layer][wb])
            loss_batch, grad_batch = cumulate(loss_batch, loss_sample, grad_batch, grad_sample, setup.batch_size)
        # Small Batch
        if setup.small_batch:
            updates, opt_state = jit(optimizer.update)(grad_batch, opt_state)
            params = jit(optax.apply_updates)(params, updates)
        loss, grads = cumulate(loss, loss_batch, grads, grad_batch, setup.num_batches)
    # Large Batch
    if not setup.small_batch:
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
    for epoch in range(setup.num_epochs):
        # reset each epoch
        state = TrainingState(state.params,state.opt_state,0)
        dat.data.check_sims()
        
        train_coarse = jit(vmap(jit(vmap(get_coarse, in_axes=(0,))),in_axes=(0,)))(data_train)
        test_coarse = jit(vmap(jit(vmap(get_coarse, in_axes=(0,))),in_axes=(0,)))(data_test)
        
        # fig = plt.figure()
        # fig.add_subplot(2,1,1)
        # plt.plot(jnp.linspace(0,2,setup.nx_fine),jnp.concatenate(data_train[0,0,0],axis=None),label='Init Fine')
        # plt.plot(jnp.linspace(0,2,setup.nx_fine),jnp.concatenate(data_train[0,0,int(setup.nt/2)],axis=None),label=f't={setup.nt*setup.dt/2} Fine')
        # plt.plot(jnp.linspace(0,2,setup.nx_fine),jnp.concatenate(data_train[0,0,-1],axis=None),label='Final Fine')

        # fig.add_subplot(2,1,2)
        # plt.plot(jnp.linspace(0,2,setup.nx),jnp.concatenate(train_coarse[0,0,0],axis=None),label='Init Coarse')
        # plt.plot(jnp.linspace(0,2,setup.nx),jnp.concatenate(train_coarse[0,0,int(setup.nt/2)],axis=None),label=f't={setup.nt*setup.dt/2} Coarse')
        # plt.plot(jnp.linspace(0,2,setup.nx),jnp.concatenate(train_coarse[0,0,-1],axis=None),label='Final Coarse')
        # fig.legend()
        # plt.show()
        # sequence data
        train_seq = jnp.array([train_coarse[:,:, ii:(ii+setup.ns+2), ...] for ii in range(setup.nt-setup.ns-1)])
        train_seq = jnp.moveaxis(train_seq,0,2)
        del train_coarse

        t1 = time.time()

        # with mra.Tracker(f"memory/memray_out{epoch}.bin"):
        params_new, opt_state_new, loss_new = update(state.params, state.opt_state, train_seq)
        state = TrainingState(params_new,opt_state_new,loss_new)

        # call test function
        test_err, err_hist = evaluate(state.params, test_coarse)

        t2 = time.time()

        # save in case job is canceled, can resume
        save_params(state.params,os.path.join(param_path,'last.pkl'))
        
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
        wandb_err_data = [[t, err] for t, err in zip(jnp.linspace(setup.dt,setup.t_max,setup.nt),err_hist)]
        err_hist_table = wandb.Table(data=wandb_err_data,columns=['Time','MSE'])
        err_hist_list.append(err_hist)
        err_hist_plot = wandb.plot.line_series(jnp.linspace(setup.dt,setup.t_max,setup.nt),err_hist_list,[f"epoch {i}" for i in range(epoch)],xname="Time after t0")
        wandb.log({
            "Train loss": float(state.loss),
            "Test Error": float(test_err),
            'Test Min': float(min_err),
            'Epoch' : float(epoch),
            "Error History Table": err_hist_table,
            "Error History": err_hist_plot})
        
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

    ml_parameters_dict = {"riemann_solver":params_best}
    ml_networks_dict = hk.data_structures.to_immutable_dict({"riemann_solver": net})

    sim_manager = run_simulation(coarse_case,coarse_num,ml_parameters_dict,ml_networks_dict)

    path = sim_manager.output_writer.save_path_domain
    _, _, _, data_dict_best = load_data(path, quantities)

    ml_parameters_dict = {"riemann_solver":params_end}
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

    # data input will be (primes_L, primes_R, cons_L, cons_R) -> ([5,(nx+1),ny,nz], [5,(nx+1),ny,nz], [5,(nx+1),ny,nz], [5,(nx+1),ny,nz])
    cons_init = jnp.zeros((5,setup.nx+1,1,1))
    initial_params = net.init(jrand.PRNGKey(10), cons_init)
    del cons_init
    if setup.load_warm:
        # loads warm params, always uses last.pkl over warm.pkl if available and toggled on
        if setup.load_last and os.path.exists(os.path.join(param_path,'last.pkl')):
            warm_params = load_params(param_path,'last.pkl')    
            if compare_params(warm_params,initial_params):
                print("\n"+"-"*5+"Using Warm-Start Params"+"-"*5+"\n")
                initial_params = warm_params
            else:
                os.system('rm {}'.format(os.path.join(param_path,'warm.pkl')))
        elif os.path.exists(os.path.join(param_path,'warm.pkl')):
            warm_params = load_params(param_path,'warm.pkl')    
            if compare_params(warm_params,initial_params):
                print("\n"+"-"*5+"Using Warm-Start Params"+"-"*5+"\n")
                initial_params = warm_params
            else:
                os.system('rm {}'.format(os.path.join(param_path,'warm.pkl')))
        del warm_params

    initial_opt_state = optimizer.init(initial_params)

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