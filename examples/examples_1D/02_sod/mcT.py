from mcT_sod_setup import save_params, load_params, mse, net, optimizer, mse
from typing import Tuple, NamedTuple, Optional, Iterable, Union
import time
import os
import wandb
import sys
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
from jaxfluids import InputReader, Initializer, SimulationManager, SimulationManagerMCT
from jaxfluids.post_process import load_data

import mcT_sod_setup as setup
import mcT_data as dat

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
last_param_path = os.path.join(param_path, 'last.pkl')
best_param_path = os.path.join(param_path, 'best.pkl')


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
    data_coarse = jim.resize(
        data_fine, (data_fine.shape[0], setup.nx, 1, 1), "linear")
    return data_coarse


@partial(jit, static_argnums=[0])
def _add_noise(noise_level: float, arr: jnp.ndarray, seed: int):
    noise_arr = jrand.normal(jrand.PRNGKey(seed), arr.shape)
    noise_arr *= noise_level/jnp.max(noise_arr)
    return arr * (1+noise_arr)


def _partial_add_noise(arr: jnp.ndarray, seed: int):
    return partial(_add_noise, setup.noise_level)(arr, seed)


@jit
def add_noise(data: jnp.ndarray, seed: Optional[int] = 1):
    seed_arr = jrand.randint(jrand.PRNGKey(seed), (5,), 1, 100)
    data_noisy = vmap(_partial_add_noise, in_axes=(0, 0))(data, seed_arr)
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
    rseqs = jnp.zeros((n_rseq, setup.nr, 5, setup.nx, 1, 1))
    for i in range(n_rseq):
        rseq_i = lax.dynamic_slice_in_dim(k_pred, i, setup.nr)
        rseqs = rseqs.at[i].set(rseq_i)
    return rseqs


case_dict = setup.case_base
numerical = setup.numerical
case_dict['domain']['x']['cells'] = setup.nx
numerical['conservatives']['time_integration']['fixed_timestep'] = setup.dt
numerical['conservatives']['time_integration']['time_integrator'] = setup.integrator

numerical['mcTangent'] = 'true'
input_reader = InputReader(case_dict, numerical.copy())
sim_manager_mct = SimulationManagerMCT(input_reader)

numerical['mcTangent'] = 'false'
input_reader = InputReader(case_dict, numerical.copy())
sim_manager_def = SimulationManagerMCT(input_reader)


@jit
def evaluate(params: hk.Params, data: jnp.ndarray) -> float:
    """
    ----- inputs -----\n
    :param state: holds parameters of the NN
    :param data: downsampled testing data, of shape [samples, primes, timesteps, xs(, ys, zs)]


    ----- returns -----\n
    :return err_epoch: mean squared error for the epoch
    :return err_hist: mean squared error for every time step of the predicted trajectories
    :return merr_epoch: mean squared error between model trajectory and truth
    :return merr_hist: mean squared error of governing model for every time step
    """
    ml_parameters_dict = {"MCTANGENT": params}
    ml_networks_dict = hk.data_structures.to_immutable_dict({"MCTANGENT": net})

    ml_primes_init = jnp.zeros(
        (setup.num_test, 5, setup.nx, setup.ny, setup.nz))
    ml_primes_init = ml_primes_init.at[:, [0, 4]].set(data[:, [0, 2], 0])

    ml_pred_arr, _ = sim_manager_mct.feed_forward(
        ml_primes_init,
        None,  # not needed for single-phase, but is a required arg for feed_forward
        setup.nt*setup.test_ratio,
        ml_parameters_dict,
        ml_networks_dict
    )

    ml_pred_arr = jnp.array(ml_pred_arr[1:])
    ml_pred_arr = jnp.moveaxis(ml_pred_arr, 0, 2)

    err_hist = jnp.array(vmap(vmap(mse, in_axes=(1, 1)), in_axes=(0, 0))(
        ml_pred_arr[:, [0, 1, 4], ...], data[:, 0:3, 1:, ...]))
    err = jnp.mean(err_hist)

    return err, jnp.mean(err_hist, axis=0), ml_pred_arr[0, [0, 1, 4], -1]


# @jit
def get_loss_batch(params: hk.Params, batch: jnp.ndarray) -> float:
    """
    Uses a highly resolved simulation as ground truth to calculate loss over a sample

    ----- inputs -----\n
    :param params: holds parameters of the NN
    :param batch: training data for one batch, of shape [samples, sequences, primes, timesteps, xs(, ys, zs)]
    :param sim: structure containing information about the truth simulations
    :param seed: seed number, used as an rng seed for noise

    ----- returns -----\n
    :return loss_batch: average loss over all sequences in the batch
    """
    ml_parameters_dict = {"MCTANGENT": params}
    ml_networks_dict = hk.data_structures.to_immutable_dict({"MCTANGENT": net})

    # concatenate all samples as one batch
    sample = jnp.concatenate(batch, axis=0)
    # sod needs different init vars
    ml_primes_init = jnp.zeros(
        (setup.batch_size*(setup.nt-setup.ns-1), 5, setup.nx, setup.ny, setup.nz))
    ml_primes_init = ml_primes_init.at[:, [0, 4]].set(sample[:, [0, 2], 0])

    if setup.noise_flag:
        ml_primes_init = vmap(add_noise, in_axes=(0, None))(ml_primes_init, 1)

    ml_pred_arr, _ = sim_manager_mct.feed_forward(
        ml_primes_init,
        None,  # not needed for single-phase, but is a required arg for feed_forward
        setup.ns+1+setup.nr,  # nr = 0 if ml only
        ml_parameters_dict,
        ml_networks_dict
    )

    ml_pred_arr = jnp.array(ml_pred_arr[1:])
    ml_pred_arr = jnp.moveaxis(ml_pred_arr, 0, 2)
    ml_loss_batch = mse(
        ml_pred_arr[:, [0, 1, 4], :setup.ns+1], sample[:, 0:3, 1:])

    # return ml_loss_batch
    if not setup.mc_flag or setup.nr < 1:
        return ml_loss_batch

    # mc loss

    # resequence for each R seq
    ml_pred_arr = jnp.swapaxes(ml_pred_arr, 1, 2)
    ml_rseqs = jnp.concatenate(vmap(get_rseqs, in_axes=(0,))(ml_pred_arr))

    # concatenate predictions up to S+1 as initial condition for mc loss
    mc_primes_init = jnp.reshape(
        ml_primes_init,
        (setup.batch_size*(setup.nt-setup.ns-1), 1, 5, setup.nx, setup.ny, setup.nz)
    )
    mc_primes_init = jnp.concatenate(
        jnp.concatenate(
            (mc_primes_init, ml_pred_arr[:, :setup.ns+1]),
            axis=1
        )
    )

    mc_pred_arr, _ = sim_manager_def.feed_forward(
        mc_primes_init,
        None,  # not needed for single-phase, but is a required arg for feed_forward
        setup.nr,
        ml_parameters_dict,
        ml_networks_dict
    )
    mc_pred_arr = jnp.array(mc_pred_arr[1:])
    mc_pred_arr = jnp.moveaxis(mc_pred_arr, 0, 1)

    # mc loss with rho only
    mc_loss_batch = setup.mc_alpha/setup.nr * mse(ml_rseqs, mc_pred_arr)
    loss_batch = ml_loss_batch + mc_loss_batch

    return loss_batch


def update_scan(carry, x):
    params, opt_state, data = carry
    batch = lax.dynamic_slice_in_dim(
        data, x*setup.batch_size, setup.batch_size)
    batch = jnp.swapaxes(batch, 1, 2)
    loss, grads = value_and_grad(get_loss_batch, argnums=0)(params, batch)
    opt_state.hyperparams['f'] = loss
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)

    return (params, opt_state, data), loss


def update_for(i, args):
    params, opt_state, data, loss = args
    batch = lax.dynamic_slice_in_dim(
        data, i*setup.batch_size, setup.batch_size)
    batch = jnp.swapaxes(batch, 1, 2)
    loss_new, grads = value_and_grad(get_loss_batch, argnums=0)(params, batch)
    opt_state.hyperparams['f'] = loss_new
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, data, loss + loss_new/setup.num_batches


# @jit
def update(params: Iterable[hk.Params], opt_state: Iterable[optax.OptState], data: jnp.ndarray) -> Tuple:
    """
    Evaluates network loss and gradients
    Applies optimizer updates and returns the new parameters, state, and loss

    ----- inputs -----\n
    :param params: current network params
    :param opt_state: current optimizer state
    :param data: array of sequenced training data, of shape [samples, primes, sequences, timesteps, xs(, ys, zs)]

    ----- returns -----\n
    :return state: tuple of arrays containing updated params, optimizer state, and loss
    """
    (params, opt_state, data), loss_arr = lax.scan(
        update_scan,
        (params, opt_state, data),
        jnp.arange(setup.num_batches)
    )

    return params, opt_state, jnp.mean(loss_arr)


def Train(params, opt_state, data_test: np.ndarray, data_train: np.ndarray) -> hk.Params:
    """
    Train mcTangent through end-to-end optimization in JAX-Fluids

    ----- inputs -----\n
    :param params: holds parameters of the NN
    :param opt_state: holds the state of the optimizer
    :param data_test: data for testing, of shape [samples, primes, times, xs(, ys, zs)]
    :param data_train: data for training, of shape [samples, primes, times, xs(, ys, zs)]

    ----- returns -----\n
    :return params: final trained parameters
    """
    train_coarse = vmap(vmap(get_coarse, in_axes=(0,)),
                        in_axes=(0,))(data_train)
    test_coarse = vmap(vmap(get_coarse, in_axes=(0,)),
                       in_axes=(0,))(data_test)

    train_seq = jnp.array([train_coarse[:, :, ii:(ii+setup.ns+2), ...]
                          for ii in range(setup.nt-setup.ns-1)])
    train_seq = jnp.moveaxis(train_seq, 0, 2)
    del train_coarse

    err_hist_df = pd.DataFrame({
        "Time": jnp.linspace(setup.dt, setup.t_max*setup.test_ratio, setup.nt*setup.test_ratio)
    })
    state_f_df = pd.DataFrame({
        "X": jnp.linspace(0, setup.x_max, setup.nx),
        "True0_rho": jax.device_get(jnp.ravel(test_coarse[0, 0, -1])),
        "True0_u": jax.device_get(jnp.ravel(test_coarse[0, 1, -1])),
        "True0_p": jax.device_get(jnp.ravel(test_coarse[0, 2, -1]))
    })

    min_err = sys.float_info.max
    epoch_min = -1

    for epoch in range(setup.num_epochs):
        t0 = time.time()
        params, opt_state, loss = update(params, opt_state, train_seq)
        t1 = time.time()
        err, err_hist, ml_f = evaluate(params, test_coarse)
        t2 = time.time()

        print("Update time: {:.2e}".format(t1-t0),
              "Eval time: {:.2e}".format(t2-t1))
        print("Loss: {:.2e}".format(loss),
              "Err: {:2e}".format(err))

        if err <= min_err:
            min_err = err
            epoch_min = epoch
            if os.path.exists(best_param_path):
                os.remove(best_param_path)
            save_params(params, best_param_path)

        # Log progress
        if epoch % 100 == 0:
            if os.path.exists(last_param_path):
                os.remove(last_param_path)
            save_params(params, last_param_path)
            print("time {:.2e}s loss {:.2e} err {:.2e}  min_err {:.2e} EPmin {:d} EP {:d} ".format(
                t2 - t0, loss, float(err), float(min_err), epoch_min, epoch))
            err_hist_df = pd.concat((
                err_hist_df, pd.DataFrame({
                    f"MCT_err_ep{epoch}_{setup.ns}s{setup.nr}r":  jax.device_get(err_hist)
                })),
                axis=1
            )
            state_f_df = pd.concat((
                state_f_df, pd.DataFrame({
                    f"MCT0_rho{epoch}": jax.device_get(jnp.ravel(ml_f[0])),
                    f"MCT0_u{epoch}": jax.device_get(jnp.ravel(ml_f[1])),
                    f"MCT0_p{epoch}": jax.device_get(jnp.ravel(ml_f[2])),
                })),
                axis=1
            )
            # weight_arr = jax.device_get(
            #     params['mc_t_2_d/conv2_d']['w']
            # )
            wandb.log({
                "Train loss": float(loss),
                "Test Error": float(err),
                "Epoch": float(epoch),
                "Error History Table": wandb.Table(data=err_hist_df),
                "Final State Table": wandb.Table(data=state_f_df),
                # "Weight Matrix": wandb.Image(weight_arr, caption='Kernel Weights'),
            })

        # if epoch % 50 == 0:  # Clear every x epochs
        #     jax.clear_backends()

    return params


def run_sim(case_dict, num_dict, params=None, net=None):
    input_reader = InputReader(case_dict, num_dict)
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
    params = {'legend.fontsize': 14, 'axes.labelsize': 16,
              'axes.titlesize': 20, 'xtick.labelsize': 14, 'ytick.labelsize': 14}
    pylab.rcParams.update(params)
    matplotlib.rcParams['mathtext.fontset'] = 'cm'
    matplotlib.rcParams['mathtext.rm'] = 'serif'
    plt.rcParams['font.family'] = 'TimesNewRoman'
    # date
    now = time.strftime("%d%m%y%H%M")
    # fine
    fine_sim = dat.data.next_sim()

    quantities = ['density']
    x_fine, _, times, data_dict_fine = fine_sim.load(quantities, dict)
    x_fine = x_fine[0]

    # coarse
    coarse_case = fine_sim.case
    coarse_num = fine_sim.numerical
    coarse_case['general']['save_path'] = results_path
    coarse_case['domain']['x']['cells'] = setup.nx
    coarse_num['conservatives']['time_integration']['fixed_timestep'] = setup.dt
    sim_manager = run_sim(coarse_case, coarse_num)

    path = sim_manager.output_writer.save_path_domain
    quantities = ['density']
    x_coarse, _, _, data_dict_coarse = load_data(path, quantities)
    x_coarse = x_coarse[0]

    # mcTangent
    coarse_num['conservatives']['convective_fluxes']['riemann_solver'] = "MCTANGENT"

    # best state, end state
    params_best = load_params(os.path.join(param_path, "best.pkl"))
    params_end = load_params(os.path.join(param_path, "end.pkl"))

    ml_parameters_dict = {"MCTANGENT": params_best}
    ml_networks_dict = hk.data_structures.to_immutable_dict({"MCTANGENT": net})

    sim_manager = run_sim(
        coarse_case, coarse_num, ml_parameters_dict, ml_networks_dict)

    path = sim_manager.output_writer.save_path_domain
    _, _, _, data_dict_best = load_data(path, quantities)

    ml_parameters_dict = {"MCTANGENT": params_end}
    sim_manager = run_sim(
        coarse_case, coarse_num, ml_parameters_dict, ml_networks_dict)

    path = sim_manager.output_writer.save_path_domain
    _, _, _, data_dict_end = load_data(path, quantities)

    data_true = data_dict_fine['density']
    data_coarse = data_dict_coarse['density']
    data_best = data_dict_best['density']
    data_end = data_dict_end['density']

    n_plot = 3
    plot_steps = np.linspace(0, data_true.shape[0]-1, n_plot, dtype=int)
    plot_times = times[plot_steps]

    fig = plt.figure(figsize=(32, 10))
    for nn in range(n_plot):
        ut = jnp.reshape(data_true[plot_steps[nn]], (setup.nx_fine,))
        uc = jnp.reshape(data_coarse[plot_steps[nn]], (setup.nx,))
        uc = jnp.nan_to_num(uc)
        ub = jnp.reshape(data_best[plot_steps[nn]], (setup.nx,))
        ub = jnp.nan_to_num(ub)
        ue = jnp.reshape(data_end[plot_steps[nn]], (setup.nx,))
        ue = jnp.nan_to_num(ue)

        ax = fig.add_subplot(1, n_plot, nn+1)
        l1 = ax.plot(x_fine, ut, '-o', linewidth=2,
                     markevery=0.2, label='True')
        l2 = ax.plot(x_coarse, uc, '--^', linewidth=2,
                     markevery=(0.05, 0.2), label='Coarse')
        l2 = ax.plot(x_coarse, ub, '--s', linewidth=2,
                     markevery=(0.10, 0.2), label='Best')
        l2 = ax.plot(x_coarse, ue, '--p', linewidth=2,
                     markevery=(0.15, 0.2), label='End')

        ax.set_aspect('auto', adjustable='box')
        ax.set_title('t = ' + str(plot_times[nn]))

        if nn == 0:
            handles, labels = ax.get_legend_handles_labels()
            fig.legend(handles, labels)

    # plt.show()
    fig.savefig(os.path.join('figs', setup.case_name+now+'.png'))

    fig = plt.figure()
    coarse_true = get_coarse(data_true)
    err_coarse = vmap(mse, in_axes=(0, 0))(coarse_true, data_coarse)
    err_best = vmap(mse, in_axes=(0, 0))(coarse_true, data_best)
    err_end = vmap(mse, in_axes=(0, 0))(coarse_true, data_end)
    plt.plot(times, err_coarse, '--^', linewidth=2,
             markevery=0.2, label='Coarse')
    plt.plot(times, err_best, '--s', linewidth=2,
             markevery=(0.06, 0.2), label='Best')
    plt.plot(times, err_end, '--p', linewidth=2,
             markevery=(0.13, 0.2), label='End')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('MSE')
    plt.title('Error Over Time')

    # plt.show()
    fig.savefig(os.path.join(
        'figs', setup.case_name+'_errHist_' + now + '.png'))


# %% main
if __name__ == "__main__":
    # data input will be primes(t)
    u_init = jnp.zeros((5, setup.nx+4, 1))
    initial_params = net.init(jrand.PRNGKey(2), u_init)
    del u_init
    if setup.load_warm or setup.load_last:
        # loads warm params, always uses last.pkl over warm.pkl if available and toggled on
        if setup.load_last and os.path.exists(os.path.join(param_path, 'last.pkl')):
            last_params = load_params(param_path, 'last.pkl')
            print("\n"+"-"*5+"Using Last Params"+"-"*5+"\n")
            initial_params = last_params
            del last_params
        elif setup.load_warm and os.path.exists(os.path.join(param_path, 'warm.pkl')):
            warm_params = load_params(param_path, 'warm.pkl')
            print("\n"+"-"*5+"Using Warm-Start Params"+"-"*5+"\n")
            initial_params = warm_params
            del warm_params
        else:
            print("\n"+"-"*5+"No Loadable Params"+"-"*5+"\n")
    else:
        print("\n"+"-"*5+"Fresh Params"+"-"*5+"\n")

    initial_opt_state = jit(optimizer.init)(initial_params)
    data_train, data_test = dat.data.load_all()

    # transfer to GPU
    data_test = jax.device_put(data_test)
    data_train = jax.device_put(data_train)

    params = Train(initial_params, initial_opt_state, data_test, data_train)
    save_params(params, os.path.join(param_path, "end.pkl"))

    # # %% visualize best and end state
    if setup.vis_flag:
        visualize()
