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
import simMan
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
results_path = setup.work('results')
test_path = setup.work('test')
param_path = setup.work("network/parameters")

# %% create mcTangent network and training functions
class TrainingState(NamedTuple):
    params: hk.Params
    opt_state: optax.OptState
    loss: float

# dense network, layer count variable not yet implemented
def mcT_fn(state: jnp.ndarray) -> jnp.ndarray:
    """Dense network with 1 layer of ReLU units"""
    mcT = hk.Sequential([
        hk.Flatten(),
        hk.Linear(5*(setup.nx + 1)), jax.nn.relu,
        hk.Linear(setup.nx + 1)
    ])
    flux = mcT(state)
    return flux

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
                if jnp.sum(jnp.array([i != j for i,j in zip(params[layer][wb].shape,shapes[layer][wb])])):
                    return False
    return True

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

@partial(jit,static_argnums=[2])
def _add_noise(arr: jnp.ndarray, seed: int, noise_level: float):
    noise_arr = jrand.normal(jrand.PRNGKey(seed),arr.shape)
    noise_arr *= noise_level/jnp.max(noise_arr)
    return arr * (1+noise_arr)

def add_noise(arr: jnp.ndarray, seed: int):
    return partial(_add_noise,noise_level=setup.noise_level)(arr,seed)

@jit
def get_coarse(data_fine, seed: Optional[int] = 1) -> jnp.ndarray:
    """
    down samples the data by factor of 4 for use in training.

    ----- inputs -----\n
    :param data_fine: data to be downsampled, of shape [times, fxs(, fyz, fzs)]

    ----- returns -----\n
    :return data_coarse: downsampled data array, of shape [times, cxs(, cyz, czs)]
    """
    data_coarse = jim.resize(data_fine,(setup.nt+1,setup.nx,1,1),"linear")
    if setup.noise_flag:
        seed_arr = jrand.randint(jrand.PRNGKey(seed),(setup.nt+1,),1,setup.nt+1)
        data_coarse = vmap(add_noise,in_axes=(0,0))(data_coarse,seed_arr)
    
    return data_coarse

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

def get_loss_batch(params: hk.Params, sample:jnp.ndarray) -> float:
    """
    Uses a highly resolved simulation as ground truth to calculate loss over a sample
    
    ----- inputs -----\n
    :param params: holds parameters of the NN
    :param sample: training data for one batch, of shape [sequences, primes, timesteps, xs(, ys, zs)]

    ----- returns -----\n
    :return loss_batch: average loss over all sequences in the sample
    """
    sim = dat.data.next_sim()
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
    # sim_manager = simMan.SimulationManager(input_reader)

    ml_primes_init = sample[:,:,0,...]

    # if setup.parallel_flag:
    #     # feed_forward = pmap(sim_manager.feed_forward,axis_name='data')
    #     # ml_pred_arr, _ = feed_forward(
    #     #     get_par_batch(ml_primes_init),
    #     #     jnp.empty_like(get_par_batch(ml_primes_init)), # not needed for single-phase, but is a required arg for feed_forward
    #     #     get_par_batch(setup.ns+1),
    #     #     get_par_batch(coarse_case['general']['save_dt']),
    #     #     get_par_batch(0), get_par_batch(1),
    #     #     get_par_batch(ml_parameters_dict),
    #     #     get_par_batch(ml_networks_dict)
    #     # )
    #     feed_forward = pmap(sim_manager.feed_forward,axis_name='data')
    #     ml_pred_arr, _ = feed_forward(
    #         get_par_batch(ml_primes_init),
    #         get_par_batch(setup.ns+1),
    #         get_par_batch(coarse_case['general']['save_dt']),
    #         get_par_batch(0), get_par_batch(1),
    #         get_par_batch(ml_parameters_dict),
    #         get_par_batch(ml_networks_dict)
    #     )
    # else:
    # ml_pred_arr, _ = partial(jit(sim_manager.feed_forward),static_argnums=(7))(
    ml_pred_arr, _ = sim_manager.feed_forward(
        ml_primes_init,
        jnp.empty_like(ml_primes_init), # not needed for single-phase, but is a required arg for feed_forward
        setup.ns+1,
        coarse_case['general']['save_dt'],
        0.0, 1,
        ml_parameters_dict,
        ml_networks_dict
    )
    # ml_pred_arr, _ = sim_manager.feed_forward(
    #     ml_primes_init,
    #     setup.ns+1,
    #     coarse_case['general']['save_dt'],
    #     0, 1,
    #     ml_parameters_dict,
    #     ml_networks_dict
    # )
    ml_pred_arr = jnp.swapaxes(ml_pred_arr,1,2)

    # ml loss
    ml_loss_batch = mse(ml_pred_arr, sample[:,:,1:,...])

    return ml_loss_batch
    # if not setup.mc_flag:
    # if jnp.isnan(ml_pred_arr).any() or (ml_pred_arr<0).any():
    #     return (setup.mc_alpha*10)*ml_loss_sample
    
    # mc loss
    # coarse_num['conservatives']['convective_fluxes']['riemann_solver'] = "HLLC"
    # input_reader = InputReader(coarse_case,coarse_num)
    # sim_manager = SimulationManager(input_reader)
    
    # #  map over times, concatenate all sequences
    # mc_primes_init = jnp.concatenate(jnp.swapaxes(ml_pred_arr[:,:,:-1,...],1,2))

    # mc_pred_arr, _ = sim_manager.feed_forward(
    #     mc_primes_init,
    #     jnp.empty_like(mc_primes_init), # not needed for single-phase, but is a required arg
    #     1, coarse_case['general']['save_dt'], 0
    # )
    # mc_pred_arr = jnp.nan_to_num(mc_pred_arr)

    # ml_pred_mcloss = jnp.concatenate(jnp.swapaxes(ml_pred_arr[:,:,1:,...],1,2))
    # mc_loss_sample = setup.mc_alpha * mse(ml_pred_mcloss,mc_pred_arr[jnp.s_[:,-1,...]])
    # loss_sample = ml_loss_sample + mc_loss_sample
    # return loss_sample

# @jit
def _evaluate_sample(params: hk.Params, sample: jnp.ndarray) -> jnp.ndarray:
    """
    creates a simulation manager to fully simulate the case using the updated mcTangent
    the resulting data is then loaded and used to calculate the mse across all test data

    ----- inputs -----\n
    :param params: holds parameters of the NN
    :param sample: allows for mapping input for vmap api, also used as the seed number if noise flag is True

    ----- returns -----\n
    :return err_sample: mean squared error for the sample
    """
    sim = dat.data.next_sim()
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
    initializer = Initializer(input_reader)
    sim_manager = SimulationManager(input_reader)
    # sim_manager = simMan.SimulationManager(input_reader)
    buffer_dictionary = initializer.initialization()
    buffer_dictionary['machinelearning_modules'] = {
        'ml_parameters_dict': ml_parameters_dict,
        'ml_networks_dict': ml_networks_dict
    }
    try:
        sim_manager.simulate(buffer_dictionary)

        # get error
        path = sim_manager.output_writer.save_path_domain
        quantities = ['density','velocityX','velocityY','velocityZ','pressure']
        _, _, _, data_dict_mcT = load_data(path, quantities)
        mcT_pred = jnp.array([data_dict_mcT[quant] for quant in data_dict_mcT.keys()])
        err_sample = mse(mcT_pred, sample)
    except:
        err_sample = jnp.array([sys.float_info.max])

    # clean
    os.system('rm -rf {}/*'.format(test_path))

    return err_sample

# @jit
def evaluate(params: hk.Params, data: jnp.ndarray) -> float:
    """
    looped form of _evaluate_sample

    ----- inputs -----\n
    :param state: holds parameters of the NN
    :param data: downsampled testing data, of shape [samples, primes, sequences, timesteps, xs]


    ----- returns -----\n
    :return err_epoch: mean squared error for the epoch
    """
    err_epoch = 0
    for sample in data:
        err_epoch += _evaluate_sample(params,sample)/setup.num_test
    # err_epoch = vmap(_evaluate_sample, in_axes=(None,0))(params, data)
    return err_epoch

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
    for sample in data:
        sample = jnp.swapaxes(sample,0,1)
        loss_sample = 0
        grad_sample = {}

        for i in range(setup.num_batches):
            seqs = lax.dynamic_slice_in_dim(sample,i*setup.batch_size,setup.batch_size)
            loss_batch, grad_batch = value_and_grad(get_loss_batch, argnums=0)(params, seqs)
            # loss_batch = get_loss_batch(params, seqs)
            # grad_batch = params
            loss_sample, grad_sample = cumulate(loss_sample, loss_batch, grad_sample, grad_batch, setup.num_batches)

        loss, grads = cumulate(loss, loss_sample, grads, grad_sample, data.shape[0])
    updates, opt_state_new = optimizer.update(grads, opt_state)
    params_new = optax.apply_updates(params, updates)

    return params_new, opt_state_new, loss

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
    for epoch in range(setup.num_epochs):
        # os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "default"
        # reset each epoch
        state = TrainingState(state.params,state.opt_state,0)
        
        train_coarse = vmap(vmap(get_coarse, in_axes=(0,None)),in_axes=(0,None))(data_train,epoch)
        test_coarse = vmap(vmap(get_coarse, in_axes=(0,None)),in_axes=(0,None))(data_test,epoch)

        # sequence data
        train_seq = jnp.array([train_coarse[:,:, ii:(ii+setup.ns+2), ...] for ii in range(setup.nt-setup.ns-1)])
        train_seq = jnp.moveaxis(train_seq,0,2)
        del train_coarse

        t1 = time.time()

        # with mra.Tracker(f"memory/memray_out{epoch}.bin"):
        params_new, opt_state_new, loss_new = update(state.params, state.opt_state, train_seq)
        state = TrainingState(params_new,opt_state_new,loss_new)

        # call test function
        test_err = evaluate(state.params, test_coarse)

        t2 = time.time()

        # save in case job is canceled, can resume
        save_params(state.params,os.path.join(param_path,'last.pkl'))
        
        if test_err <= min_err:
            min_err = test_err
            epoch_min = epoch
            best_state = state

        if epoch % 1 == 0:  # Print every 1 epochs
            print("time {:.2e}s loss {:.2e} TE {:.2e}  TE_min {:.2e} EPmin {:d} EP {} ".format(
                    t2 - t1, state.loss, test_err, min_err, epoch_min, epoch))
        
        dat.data.check_sims()
        jprof.save_device_memory_profile(f"memory/memory_{epoch}.prof")
        wandb.log({"Train loss": float(state.loss), "Test Error": float(test_err), 'TEST MIN': float(min_err), 'Epoch' : float(epoch)})
        # os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
        jax.clear_backends()

    return best_state, state

net = hk.without_apply_rng(hk.transform(mcT_fn))
optimizer = optax.adam(setup.learning_rate)

if __name__ == "__main__":
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

    # data input will be mean(primes_L[0], primes_R[0]) -> [(nx+1),1,1]
    data_init = jnp.empty((1,setup.nx+1,setup.ny,setup.nz))
    initial_params = net.init(jrand.PRNGKey(1), data_init)
    if os.path.exists(os.path.join(param_path,'last.pkl')):
        last_params = load_params(param_path,'last.pkl')    
        if compare_params(last_params,initial_params):
            initial_params = last_params
        else:
            del last_params
            os.system('rm {}'.format(os.path.join(param_path,'last.pkl')))
    del data_init

    initial_opt_state = optimizer.init(initial_params)

    state = TrainingState(initial_params, initial_opt_state, 0)

    data_test, data_train = dat.data.load_all()

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

    # %% visualize best and end state

    # fine
    fine_sim = dat.data.next_sim()

    quantities = ['density']
    x_fine, _, times, data_dict_fine = load_data(fine_sim.domain, quantities)

    # coarse
    coarse_case = fine_sim.case
    coarse_num = fine_sim.numerical
    coarse_case['general']['save_path'] = results_path
    coarse_case['domain']['x']['cells'] = setup.dx
    coarse_num['conservatives']['time_integration']['fixed_timestep'] = setup.dt
    input_reader = InputReader(coarse_case,coarse_num)
    initializer = Initializer(input_reader)
    sim_manager = SimulationManager(input_reader)
    # sim_manager = simMan.SimulationManager(input_reader)
    buffer_dictionary = initializer.initialization()
    sim_manager.simulate(buffer_dictionary)

    path = sim_manager.output_writer.save_path_domain
    quantities = ['density']
    x_coarse, _, _, data_dict_coarse = load_data(path, quantities)

    coarse_num['conservatives']['convective_fluxes']['riemann_solver'] = "MCTANGENT"

    # best state
    params_best = load_params(os.path.join(param_path,"best.pkl"))

    ml_parameters_dict = {"riemann_solver":params_best}
    ml_networks_dict = hk.data_structures.to_immutable_dict({"riemannsolver": net})

    input_reader = InputReader(coarse_case,coarse_num)
    initializer = Initializer(input_reader)
    sim_manager = SimulationManager(input_reader)
    # sim_manager = simMan.SimulationManager(input_reader)
    buffer_dictionary = initializer.initialization()
    buffer_dictionary['machinelearning_modules'] = {
        'ml_parameters_dict': ml_parameters_dict,
        'ml_networks_dict': ml_networks_dict
    }
    sim_manager.simulate(buffer_dictionary)

    path = sim_manager.output_writer.save_path_domain
    _, _, _, data_dict_best = load_data(path, quantities)

    # end state
    params_end = load_params(os.path.join(param_path,"end.pkl"))
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

    # plt.show()
    fig.savefig(os.path.join('figs',setup.case_name+'.png'))