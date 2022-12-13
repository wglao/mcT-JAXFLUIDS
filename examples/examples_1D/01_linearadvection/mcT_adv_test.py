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
params = {'legend.fontsize': 14, 'axes.labelsize': 16, 'axes.titlesize': 20, 'xtick.labelsize': 14, 'ytick.labelsize': 14}
pylab.rcParams.update(params)
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

cache_path = '.test_cache'
rng = [jrand.PRNGKey(i) for i in range(10)]
os.makedirs(cache_path,exist_ok=True)
data_init = jnp.empty((1,setup.nx+1,1,1))
optimizer = optax.adam(setup.learning_rate)
initial_params = net.init(rng.pop(), data_init)
initial_opt_state = optimizer.init(initial_params)

state = TrainingState(initial_params, initial_opt_state, 0)
print('\n','-'*10,'Init Complete','-'*10,'\n')

# %% Data Load
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
test, train = dat.data.load_all()
t2 = time.time()
dat.data.check_sims() 
print('\n')

try:
    jnp.reshape(test,(setup.num_test, 5,setup.nt+1,setup.nx_fine,setup.ny_fine,setup.nz_fine))
except:
    raise "\nTest data is of incorrect shape. Loaded array with shape %s" %(test.shape)
try:
    jnp.reshape(train,(setup.num_train, 5,setup.nt+1,setup.nx_fine,setup.ny_fine,setup.nz_fine))
except:
    raise "\nTrain data is of incorrect shape. Loaded array with shape %s" %(train.shape)

print('\nData loaded, with shapes: \n{0}, \n{1}'.format(*[test.shape, train.shape]))
print('Load time: %s s' %(t2-t1))

del test, train
print('\n','-'*10,'Load All Pass','-'*10,'\n')

# %% save and load params

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

    t1 = time.time()
    noisy = add_noise(pure,1)
    t2 = time.time()

    err = mse(noisy, pure)/mse(pure)
    print('True Sinc(2*pi*x): %s' %pure)
    print('Noisy Sinc(2*pi*x): %s' %noisy)
    print('Average Error: %s' %err)

    print('Noise Level: %s, time: %s s' %(setup.noise_level, t2-t1))

    if err == 0 or err > setup.noise_level:
        raise "Add Noise failed"


    del xs, pure, noisy
print('\n','-'*10,'Add Noise Pass','-'*10,'\n')

# %% NN
state = jrand.normal(rng.pop(),(1,setup.nx+1,1,1))+1
t1 = time.time()
flux = net.apply(initial_params,state)
t2 = time.time()

print('Flux Shape: {}'.format(flux.shape),'\ntime: %s s' %(t2-t1))
print(type(flux))

try:
    jnp.reshape(flux,state.shape)
except:
    raise "NN Apply failed"

if type(flux) == None:
    raise "NN Apply failed"

del state, flux
print('\n','-'*10,'NN Apply Pass','-'*10,'\n')

# %% ML Loss
sim = dat.data.next_sim()
_,_,_, data = sim.load()
data_coarse = get_coarse(data)
data_coarse = jax.device_get(data_coarse)
data_coarse = np.reshape(data_coarse,(1,5,setup.nt+1,setup.nx,1,1))
data_seq = np.array([data_coarse[:,:, ii:(ii+setup.ns+2), ...] for ii in range(setup.nt-setup.ns-1)])
data_seq = np.moveaxis(data_seq,0,2)
del data
dat.data.check_sims()

setup.mc_flag = False
t1 = time.time()
loss = get_loss_batch(initial_params, data_seq)
t2 = time.time()

print('\nLoss: %s' %loss, type(loss))
print('time: %s s' %(t2-t1))

if loss.dtype != jnp.dtype('float64') or loss._value.max == jnp.nan:
    raise "ML Loss function failed"


print('\n','-'*10,'ML Loss Function Pass','-'*10,'\n')

# %% MC Loss
setup.mc_flag = True
t1 = time.time()
loss = get_loss_batch(initial_params, data_seq)
t2 = time.time()

print('\nLoss: %s' %loss, type(loss))
print('time: %s s' %(t2-t1))

if loss.dtype != jnp.dtype('float64') or loss._value.max == jnp.nan:
    raise "MC Loss function failed"

print('\n','-'*10,'MC Loss Function Pass','-'*10,'\n')

# %% Test

dat.data.check_sims()

t1 = time.time()
err = evaluate_epoch(initial_params, data_coarse)
t2 = time.time()

print('\nError: %s' %err, type(err))
print('time: %s s' %(t2-t1))

if type(err) != float or jnp.isnan(err):
    raise "Evaluate function failed"

print('\n','-'*10,'Evaluate Function Pass','-'*10,'\n')

# %% Update
if setup.parallel_flag:
    data_batch = np.array_split(data_seq, setup.num_batches)
    n_dev = jax.local_device_count()
    data_par = np.array_split(data_batch[0], n_dev)
    params_par = jax.tree_map(lambda x: jnp.array([x] * n_dev), initial_params)
    opt_par = jax.tree_map(lambda x: jnp.array([x] * n_dev), initial_opt_state)

    t1 = time.time()
    params_new, opt_state_new, loss_new = update(params_par,opt_par,data_par)
    t2 = time.time()

    print(type(params_new[0]),type(opt_state_new[0]),type(loss_new[0]))

    if type(params_new[0]) != dict:
        raise "Update function failed: params error"

    if type(opt_state_new[0]) != optax.OptState:
        raise "Update function failed: opt state error"

    if loss_new[0].dtype != jnp.dtype('float64') or loss_new[0]._value.max == jnp.nan:
        raise "Update function failed: loss error"

    print('\n','-'*10,'Parallel Update Function Pass','-'*10,'\n')

else:
    t1 = time.time()
    params_new, opt_state_new, loss_new = update(initial_params,initial_opt_state,jax.device_put(data_seq))
    t2 = time.time()

    print("Types of updated state:")
    print(type(params_new),type(opt_state_new),type(loss_new))

    if type(params_new) != dict:
        raise "Update function failed: params error"

    if type(opt_state_new) != tuple:
        raise "Update function failed: opt state error"

    if loss_new.dtype != jnp.dtype('float64') or loss_new._value.max == jnp.nan:
        raise "Update function failed: loss error"

    print('\n','-'*10,'Serial Update Function Pass','-'*10,'\n')

# %% clean up
os.system('rm -rf %s' %(cache_path))