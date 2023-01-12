# %% imports
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
import matplotlib.pylab as pylab
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm  # Colour map
import matplotlib.animation as animatio
import matplotlib.font_manager
figparams = {'legend.fontsize': 14, 'axes.labelsize': 16, 'axes.titlesize': 20, 'xtick.labelsize': 14, 'ytick.labelsize': 14}
pylab.rcParams.update(figparams)
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['mathtext.rm'] = 'serif'
plt.rcParams['font.family'] = 'TimesNewRoman'

from typing import Tuple, NamedTuple, Optional, Iterable
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
from mcT_adv import *

# %% init
units_to_test = {
    'Data Tools': False,
    'Params Tools': False,
    'Helper Tools': False,
    'NN': False,
    'Learning Tools': False,
    'Visualizaiton': True
}

cache_path = '.test_cache'
rng = [jrand.PRNGKey(i) for i in range(10)]
os.makedirs(cache_path,exist_ok=True)
data_init = jnp.empty((5,setup.nx+1,1,1))
optimizer = optax.adam(setup.learning_rate)
initial_params = net.init(rng.pop(), data_init)
initial_opt_state = optimizer.init(initial_params)

state = TrainingState(initial_params, initial_opt_state, 0)

print('\nNumber of Devices: %s' %jax.local_device_count())
print('\n','-'*10,'Init Complete','-'*10,'\n')

# %% Data Load
if units_to_test['Data Tools']:
    t1 = time.time()
    test = dat.data._load(dat.data.next_sim())
    t2 = time.time()
    dat.data.check_sims()

    try:
        jnp.reshape(test,(5,setup.nt+1,setup.nx_fine,setup.ny_fine,setup.nz_fine))
    except:
        raise "Data is of incorrect shape. Loaded array with shape %s" %(test.shape)

    print('Data loaded, with shape: {}'.format(test.shape))
    print('Load time: %s s' %(t2-t1))

    print('\n','-'*10,'Single Sample Load Pass','-'*10,'\n')

    t1 = time.time()
    train, test = dat.data.load_all()
    t2 = time.time()
    dat.data.check_sims() 
    print('\n')

    try:
        jnp.reshape(train,(setup.num_train,5,setup.nt+1,setup.nx_fine,setup.ny_fine,setup.nz_fine))
    except:
        raise f"\nTrain data is of incorrect shape. Loaded array with shape {train.shape}"
    try:
        jnp.reshape(test,(setup.num_test,5,100*setup.nt+1,setup.nx_fine,setup.ny_fine,setup.nz_fine))
    except:
        raise f"\nTest data is of incorrect shape. Loaded array with shape {test.shape}"

    print('\nData loaded, with shapes: \n{0}, \n{1}'.format(*[test.shape, train.shape]))
    print('Load time: %s s' %(t2-t1))

    del test
    print('\n','-'*10,'Load All Pass','-'*10,'\n')

# %% save and load params
if units_to_test['Params Tools']:
    t1 = time.time()
    save_params(initial_params,cache_path,'test.pkl')
    t2 = time.time()
    test_params = load_params(cache_path,'test.pkl')
    t3 = time.time()

    print('Save Time: %s s' %(t2-t1))
    print('Load Time: %s s' %(t3-t2))

    del test_params
    print('\n','-'*10,'Param Utils Pass','-'*10,'\n')

# %% MSE
if units_to_test['Helper Tools']:
    dim = 500
    test1 = jrand.normal(rng.pop(),(dim,dim,dim))
    test2 = jrand.normal(rng.pop(),(dim,dim,dim))

    t1 = time.time()
    true_mse = jnp.mean(jnp.square(test1-test2))
    t2 = time.time()
    apply_mse = mse(test1,test2)
    t3 = time.time()

    print('True MSE: %s, time: %s s' %(true_mse,t2-t1))
    print('Applied MSE: %s, time: %s s' %(apply_mse,t3-t2))

    if true_mse != apply_mse:
        raise "MSE function failed"

    del true_mse, apply_mse, test1, test2, t3
    print('\n','-'*10,'MSE Pass','-'*10,'\n')

# %% Noise
    if setup.noise_flag:
        xs = jnp.linspace(0,1,5)
        pure = jnp.sinc(xs*2*jnp.pi)
        pure = jnp.array([pure]*5)

        t1 = time.time()
        noisy = add_noise(pure,1)
        t2 = time.time()

        err = jnp.sqrt(mse(noisy, pure))/jnp.max(pure)
        print('True Sinc(2*pi*x): %s' %pure)
        print('Noisy Sinc(2*pi*x): %s' %noisy)
        print('Mean Relative Error: %s' %err)

        print('Noise Level: %s, time: %s s' %(setup.noise_level, t2-t1))

        if err == 0 or err > setup.noise_level:
            raise "Add Noise failed"


        del xs, pure, noisy
    print('\n','-'*10,'Add Noise Pass','-'*10,'\n')

# %% NN
if units_to_test['NN']:
    state = jrand.normal(rng.pop(),(5,setup.nx+1,1,1))+1
    t1 = time.time()
    tangent = net.apply(initial_params,state)
    t2 = time.time()

    print('Tagent Manifold Shape: {}'.format(tangent.shape),'\ntime: %s s' %(t2-t1))
    print(type(tangent))

    try:
        jnp.reshape(tangent,state.shape)
    except:
        raise "NN Apply failed"

    if type(tangent) == None:
        raise "NN Apply failed"

    del state, tangent
    print('\n','-'*10,'NN Apply Pass','-'*10,'\n')

# %% ML Loss
if units_to_test['Learning Tools']:
    sim = dat.data.next_sim()
    _,_,_, data = sim.load()
    data_coarse = jit(vmap(get_coarse, in_axes=(0,)))(data)
    data_seq = jnp.array([data_coarse[:, ii:(ii+setup.ns+2), ...] for ii in range(setup.nt-setup.ns-1)])
    # data_seq = jnp.moveaxis(data_seq,0,1)
    del data

    setup.mc_flag = False
    t1 = time.time()
    loss_ml, grads_ml = value_and_grad(get_loss_sample, argnums=0, allow_int=True)(initial_params, data_seq, sim, 1)
    t2 = time.time()

    print('\nLoss: %s' %loss_ml, type(loss_ml))
    print('time: %s s' %(t2-t1))

    if loss_ml.dtype != jnp.dtype('float64') or jnp.sum(jnp.isnan(loss_ml._value)) > 0:
        raise "ML Loss function failed"

    for layer in grads_ml.keys():
        for wb in grads_ml[layer].keys():
            arr = grads_ml[layer][wb]
            if arr.dtype != jnp.dtype('float64') or jnp.sum(jnp.isnan(arr._value)) > 0:
                raise "MC Loss function failed: invalid grads"


    print('\n','-'*10,'ML Loss Function Pass','-'*10,'\n')

# %% MC Loss
    setup.mc_flag = True
    setup.mc_alpha = 1e5
    t1 = time.time()
    loss_mc, grads_mc = value_and_grad(get_loss_sample, argnums=0, allow_int=True)(initial_params, data_seq, sim, 1)
    t2 = time.time()

    print('\nLoss: %s' %loss_mc, type(loss_mc))
    print('time: %s s' %(t2-t1))

    if loss_mc.dtype != jnp.dtype('float64') or jnp.sum(jnp.isnan(loss_mc._value)) > 0:
        raise "MC Loss function failed: invalid loss"

    for layer in grads_mc.keys():
        for wb in grads_mc[layer].keys():
            arr = grads_mc[layer][wb]
            if arr.dtype != jnp.dtype('float64') or jnp.sum(jnp.isnan(arr._value)) > 0:
                raise "MC Loss function failed: invalid grads"

    print('\n','-'*10,'MC Loss Function Pass','-'*10,'\n')

# %% Cumulate Grads

    loss = 0
    grads = {}

    t1 = time.time()
    loss, grads = cumulate(loss,loss_mc,grads,grads_mc,1)
    t2 = time.time()

    print('time: %s s' %(t2-t1))

    if loss.dtype != jnp.dtype('float64') or type(grads) != dict:
        raise "Cumulate function failed"

    print('\n','-'*10,'Cumulate Function Pass','-'*10,'\n')

# %% Evaluate
    dat.data.check_sims()
    data_coarse = jnp.reshape(data_coarse, (1,5,setup.nt+1,setup.nx,1,1))
    t1 = time.time()
    err = evaluate(initial_params, data_coarse)
    t2 = time.time()

    print('\nError: %s' %err, type(err))
    print('time: %s s' %(t2-t1))

    if err.dtype != jnp.dtype('float64') or jnp.isnan(err):
        raise "Evaluate function failed"

    print('\n','-'*10,'Evaluate Function Pass','-'*10,'\n')

# %% Update
    train_coarse = jit(vmap(jit(vmap(get_coarse, in_axes=(0,))),in_axes=(0,)))(train)

    # sequence data
    train_seq = jnp.array([train_coarse[:,:, ii:(ii+setup.ns+2), ...] for ii in range(setup.nt-setup.ns-1)])
    train_seq = jnp.moveaxis(train_seq,0,2)
    del train_coarse

    if setup.parallel_flag:
        data_batch = np.array_split(data_seq, setup.num_batches)
        n_dev = jax.local_device_count()
        data_par = np.array_split(data_batch[0], n_dev)
        params_par = jax.tree_map(lambda x: jnp.array([x] * n_dev), initial_params)
        opt_par = jax.tree_map(lambda x: jnp.array([x] * n_dev), initial_opt_state)

        t1 = time.time()
        params_new, opt_state_new, loss_new = update(params_par,opt_par,data_par)
        t2 = time.time()

        print("Types of updated state:")
        print(type(params_new[0]),type(opt_state_new[0]),type(loss_new[0]))
        print('time: %s s' %(t2-t1))

        if type(params_new[0]) != dict:
            raise "Update function failed: params error"

        if type(opt_state_new[0]) != optax.OptState:
            raise "Update function failed: opt state error"

        if loss_new[0].dtype != jnp.dtype('float64') or loss_new[0]._value.max == jnp.nan:
            raise "Update function failed: loss error"

        print('\n','-'*10,'Parallel Update Function Pass','-'*10,'\n')

    else:
        t1 = time.time()
        params_new, opt_state_new, loss_new = update(initial_params,initial_opt_state,train_seq)
        t2 = time.time()

        print("Types of updated state:")
        print(type(params_new),type(opt_state_new),type(loss_new))
        print('time: %s s' %(t2-t1))

        if type(params_new) != dict:
            raise "Update function failed: params error"

        if type(opt_state_new) != tuple:
            raise "Update function failed: opt state error"

        if loss_new.dtype != jnp.dtype('float64') or loss_new._value.max == jnp.nan:
            raise "Update function failed: loss error"

        print('\n','-'*10,'Serial Update Function Pass','-'*10,'\n')

# %% Visualize
if units_to_test['Visualizaiton']:
    dat.data.check_sims()
    # for _ in range(setup.num_test + setup.num_train):
    #     dat.data.next_sim()

    figs_existing = len(os.listdir(setup.proj('figs')))

    t1 = time.time()
    visualize()
    t2 = time.time()

    figs_updated = len(os.listdir(setup.proj('figs')))

    figs_generated = figs_updated-figs_existing

    print("Figs Created:")
    print(figs_generated)
    print(f'time: {t2-t1}s')

    if figs_generated != 2:
        raise "Visualize Failed"

    print('\n','-'*10,'Visualization Pass','-'*10,'\n')

# %% clean up
os.system('rm -rf %s' %(cache_path))