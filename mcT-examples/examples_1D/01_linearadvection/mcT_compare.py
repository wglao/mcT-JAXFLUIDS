import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import h5py
import pickle
import jax
from jax import vmap, jit
import jax.numpy as jnp
from jax.example_libraries import stax, optimizers

# get current parameters
import mcT_parameters as pars

N = pars.N
x = np.linspace(0, 1, N)

# gound truth solution
truth = np.zeros((pars.num_test_samples, pars.nt_test_data+1, pars.N))
test_path = 'data/test'
test_runs = os.listdir(test_path)

print('=' * 20 + ' >>')
print('Loading test data ...')
for ii, run in enumerate(test_runs):
    data_path = os.path.join(test_path, run, 'domain')
    saves = os.listdir(data_path)
    Test_times = np.zeros(pars.nt_test_data+1)
    
    for jj, save in enumerate(saves[0:pars.nt_test_data+1]):
        file = os.path.join(data_path, save)
        f = h5py.File(file, 'r')
        
        data = f['primes']['density'][()]
        truth[ii,jj,:] = data
        
        save_time = f['time'][()]
        Test_times[jj] = save_time

print(truth.shape)

# randomized initial condition
input_noise = False
if input_noise:
    ns, nt, nx = truth.shape
    nosie_vec = jax.random.normal(pars.key_data_noise, truth.shape)
    noise_level = 0.02
    truth_noise = np.zeros(truth.shape)

    for i in range(ns):
        for j in range(nt):
                truth_noise[i,j,:] = truth[i,j,:] + noise_level * nosie_vec[i,j,:] * np.max(truth[i,j,:])
    
    plt.plot(x, truth[0,0,:])
    plt.plot(x, truth_noise[0,0,:])
    plt.show()

_, _, opt_get_params = optimizers.adam(pars.learning_rate)
def unpickle_params(filepath):
    ret = pickle.load(open(filepath, 'rb'))
    ret = optimizers.pack_optimizer_state(ret)
    return opt_get_params(ret)

problem = 'linearadvection'

# data only
d_only_params = unpickle_params('Network/Best_' + problem + '_seq_n_mc_' + str(pars.n_seq_mc) +'_forward_mc_train_d' + str(pars.num_train) + '_alpha_0_lr_' + str(pars.learning_rate) + '_batch_' + str(pars.batch_size) + '_nseq_' + str(pars.n_seq) + '_layer_' + str(pars.layers) + 'neurons' + str(pars.units) + '_epochs_' + str(pars.num_epochs))

# model-constrained
# mc_params = unpickle_params('Network/Best_wave1d_dt_train-test_' + str(dt) + '-' + str(dt_test) + '_seq_n_mc_' + str(n_seq_mc) +'_forward_mc_train_d' + str(num_train) + '_alpha_' + str(1e5) + '_lr_' + str(learning_rate) + '_batch_' + str(batch_size) + '_nseq_' + str(n_seq) + '_layer_' + str(layers) + 'neurons' + str(units) + '_epochs_' + str(num_epochs))

# with noise
# noisy_params = unpickle_params('Network/Best_wave1d_noise_0.02_dt_train-test_' + str(dt) + '-' + str(dt_test) + '_seq_n_mc_' + str(n_seq_mc) +'_forward_mc_train_d' + str(num_train) + '_alpha_0_lr_' + str(learning_rate) + '_batch_' + str(batch_size) + '_nseq_' + str(n_seq) + '_layer_' + str(layers) + 'neurons' + str(units) + '_epochs_' + str(num_epochs))

# model-constrained with noise
# mcn_params = unpickle_params('Network/Best_wave1d_noise_0.02dt_train-test_' + str(dt) + '-' + str(dt_test) + '_seq_n_mc_' + str(n_seq_mc) +'_forward_mc_train_d' + str(num_train) + '_alpha_' + str(1e5) + '_lr_' + str(learning_rate) + '_batch_' + str(batch_size) + '_nseq_' + str(n_seq) + '_layer_' + str(layers) + 'neurons' + str(units) + '_epochs_' + '50000')


# from network_dense_wave import *
dt = pars.dt
def ReLU(x):
    """ Rectified Linear Unit (ReLU) activation function """
    return jnp.maximum(0, x)

def Dense(inputs, W, b):
    return jnp.dot(inputs, W) + b

def forward_pass(params, u):
    W1, W2, b1, b2 = params
    u = Dense(ReLU(Dense(u, W1, b1)), W2, b2)
    return u

def single_forward_pass(params, un):
    u = un - pars.facdt * dt * forward_pass(params, un)
    return u.flatten()

@jit
def neural_solver(params, U_test):

    u = U_test[0, :]

    U = jnp.zeros((pars.nt_test_data + 1, N))
    U = U.at[0, :].set(u)

    for i in range(1, pars.nt_test_data + 1):
        u = single_forward_pass(params, u)
        U = U.at[i, :].set(u)

    return U

neural_solver_batch = vmap(neural_solver, in_axes=(None, 0))

plot_sample = 75
U_true = truth[plot_sample, :, :]
if input_noise:
    U_d_only = neural_solver_batch(d_only_params, truth_noise)[plot_sample, :, :]
    # U_mc = neural_solver_batch(mc_params, truth_noise)[plot_sample, :, :]
    # U_noisy = neural_solver_batch(noisy_params, truth_noise)[plot_sample, :, :]
    # U_mcn = neural_solver_batch(mcn_params, truth_noise)[plot_sample, :, :]
else:
    U_d_only = neural_solver_batch(d_only_params, truth)[plot_sample, :, :]
    # U_mc = neural_solver_batch(mc_params, truth)[plot_sample, :, :]
    # U_noisy = neural_solver_batch(noisy_params, truth)[plot_sample, :, :]
    # U_mcn = neural_solver_batch(mcn_params, truth)[plot_sample, :, :]

fontsize = 4
fig = plt.figure(figsize=((pars.n_plot+1)*fontsize,fontsize), dpi=400)
plt.rcParams.update({'font.size': fontsize})
for i in range(pars.n_plot):

    ut = jnp.reshape(U_true[pars.Plot_Steps[i], :], (N, 1))
    ud = jnp.reshape(U_d_only[pars.Plot_Steps[i], :], (N, 1))
    # um = jnp.reshape(U_mc[Plot_Steps[i], :], (N, 1))
    # un = jnp.reshape(U_noisy[Plot_Steps[i], :], (N, 1))
    # umn = jnp.reshape(U_mc[Plot_Steps[i], :], (N, 1))
    
    ax = fig.add_subplot(1, pars.n_plot, i+1)
    l1 = ax.plot(x, ut, '-', linewidth=1.5, label='True')
    l2 = ax.plot(x, ud, ':o', markevery=5, fillstyle='none', linewidth=1.5, label='Data only')
    # l3 = ax.plot(x, um, ':v', markevery=5, fillstyle='none', linewidth=1.5, label='Model constrained (1e5)')
    # l4 = ax.plot(x, un, ':x', markevery=5, linewidth=1.5, label='With noise (0.02)')
    # l5 = ax.plot(x, umn, ':+', markevery=5, linewidth=1.5, label='Model constrained (1e5) and with noise (0.02)')

    # ax.set_aspect('auto', adjustable='box')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('t = ' + str(pars.Plot_Steps[i]))

    if i == pars.n_plot-1:
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='center right')

if input_noise:
    plt.savefig('figs/compare_'+ problem + '_noise.png', bbox_inches='tight')
else:
    plt.savefig('figs/compare_'+ problem + '.png', bbox_inches='tight')
# plt.show()
