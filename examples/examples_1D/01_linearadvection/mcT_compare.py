import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import backend_bases as back
import pandas as pd
import os
import h5py
import pickle
import jax
from jax import vmap, jit
import jax.numpy as jnp
from jax.example_libraries import stax, optimizers

import mcT_forward_schemes_1D as mctf

# get current parameters
import mcT_parameters as pars

N = pars.N
x = np.linspace(0, 2, N)

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

network = 'End'

# data only
d_only_params = unpickle_params('Network/' + network  + '_' + problem + '_seq_n_mc_' + str(pars.n_seq_mc) +'_forward_mc_train_d' + str(pars.num_train) + '_alpha_0_lr_' + str(pars.learning_rate) + '_batch_' + str(pars.batch_size) + '_nseq_' + str(pars.n_seq) + '_layer_' + str(pars.layers) + 'neurons' + str(pars.units) + '_epochs_' + str(pars.num_epochs))

# model-constrained
mc_params = unpickle_params('Network/' + network  + '_' + problem + '_seq_n_mc_' + str(pars.n_seq_mc) +'_forward_mc_train_d' + str(pars.num_train) + '_alpha_' + str(pars.mc_alpha) + '_lr_' + str(pars.learning_rate) + '_batch_' + str(pars.batch_size) + '_nseq_' + str(pars.n_seq) + '_layer_' + str(pars.layers) + 'neurons' + str(pars.units) + '_epochs_' + str(pars.num_epochs))

# with noise
noisy_params = unpickle_params('Network/' + network  + '_' + problem + '_noise_' + str(pars.noise_level) + '_seq_n_mc_' + str(pars.n_seq_mc) +'_forward_mc_train_d' + str(pars.num_train) + '_alpha_0_lr_' + str(pars.learning_rate) + '_batch_' + str(pars.batch_size) + '_nseq_' + str(pars.n_seq) + '_layer_' + str(pars.layers) + 'neurons' + str(pars.units) + '_epochs_' + str(pars.num_epochs))

# model-constrained with noise
mcn_params = unpickle_params('Network/' + network  + '_' + problem + '_noise_' + str(pars.noise_level) + '_seq_n_mc_' + str(pars.n_seq_mc) +'_forward_mc_train_d' + str(pars.num_train) + '_alpha_' + str(pars.mc_alpha) + '_lr_' + str(pars.learning_rate) + '_batch_' + str(pars.batch_size) + '_nseq_' + str(pars.n_seq) + '_layer_' + str(pars.layers) + 'neurons' + str(pars.units) + '_epochs_' + str(pars.num_epochs))

# foward solver
dt = pars.dt
dx = pars.dx
velo = pars.u
def single_solve_forward(un):
    u = mctf.MacCormack(un, velo, dt, dx)
    return u

@jit
def forward_solver(U_test):

    u = U_test[0, :]

    U = jnp.zeros((pars.nt_test_data + 1, N))
    U = U.at[0, :].set(u)

    for i in range(1, pars.nt_test_data + 1):
        u = single_solve_forward(u)
        U = U.at[i, :].set(u)

    return U

forward_solver_batch = vmap(forward_solver, in_axes=(0))

# from network_dense_wave import *
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
    U_fwd = forward_solver_batch(truth_noise)[plot_sample, :, :]
    U_d_only = neural_solver_batch(d_only_params, truth_noise)[plot_sample, :, :]
    U_mc = neural_solver_batch(mc_params, truth_noise)[plot_sample, :, :]
    U_noisy = neural_solver_batch(noisy_params, truth_noise)[plot_sample, :, :]
    U_mcn = neural_solver_batch(mcn_params, truth_noise)[plot_sample, :, :]
else:
    U_fwd = forward_solver_batch(truth)[plot_sample, :, :]
    U_d_only = neural_solver_batch(d_only_params, truth)[plot_sample, :, :]
    U_mc = neural_solver_batch(mc_params, truth)[plot_sample, :, :]
    U_noisy = neural_solver_batch(noisy_params, truth)[plot_sample, :, :]
    U_mcn = neural_solver_batch(mcn_params, truth)[plot_sample, :, :]

u_solutions = np.array([U_fwd, U_d_only, U_mc, U_noisy, U_mcn])

fontsize = 4
fig = plt.figure(figsize=((pars.n_plot+1)*fontsize,fontsize), dpi=400)
plt.rcParams['font.size'] = fontsize
plt.xlabel('x-coordinate')
plt.ylabel('density')

for i in range(pars.n_plot):

    ut = jnp.reshape(U_true[pars.Plot_Steps[i], :], (N, 1))
    uf = jnp.reshape(U_fwd[pars.Plot_Steps[i], :], (N, 1))
    ud = jnp.reshape(U_d_only[pars.Plot_Steps[i], :], (N, 1))
    um = jnp.reshape(U_mc[pars.Plot_Steps[i], :], (N, 1))
    un = jnp.reshape(U_noisy[pars.Plot_Steps[i], :], (N, 1))
    umn = jnp.reshape(U_mcn[pars.Plot_Steps[i], :], (N, 1))
    
    ax = fig.add_subplot(1, pars.n_plot, i+1)
    l1 = ax.plot(x, ut, '-k', linewidth=1.5, label='True')
    l0 = ax.plot(x, ud, '-', linewidth=1.5, label='Forward solver')
    l2 = ax.plot(x, ud, ':', fillstyle='none', linewidth=1, label='Data only')
    l3 = ax.plot(x, um, ':', fillstyle='none', linewidth=1, label='Model constrained (1e5)')
    l4 = ax.plot(x, un, ':', linewidth=1, label='With noise (0.02)')
    l5 = ax.plot(x, umn, ':', linewidth=1, label='Model constrained (1e5) and with noise (0.02)')


    # ax.set_aspect('auto', adjustable='box')
    # ax.set_xticks([])
    # ax.set_yticks([])
    ax.set_title('t = ' + str(pars.Plot_Steps[i]))

    if i == pars.n_plot-1:
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='center right')

handles, labels = ax.get_legend_handles_labels()

if input_noise:
    plt.savefig('figs/compare_'+ problem + '_noise.png', bbox_inches='tight')
else:
    plt.savefig('figs/compare_'+ problem + '.png', bbox_inches='tight')

plt.close()
fig = plt.figure()
ax = plt.axes(xlim=(min(x),max(x)), ylim=(np.min(u_solutions)*0.9,np.max(u_solutions)*1.1))

# animation function.  This is called sequentially
lines=[]     # list for plot lines for solvers and analytical solutions
legends=labels   # list for legends for solvers and analytical solutions

for ii in range(6):
    line, = ax.plot([], [])
    lines.append(line)

plt.xlabel('x-coordinate')
plt.ylabel('density')
plt.legend(legends, loc=3, frameon=False)
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['font.size'] = 18

def init_lines():
    for line in lines:
        line.set_data([], [])
    return lines,

def animate_alt(i):
    new_lines = []
    for k, line in enumerate(lines):
        if (k==0):
            line.set_ydata(U_true[i,:])
        else:
            line.set_ydata(u_solutions[k-1,i,:])
        new_lines.append(line)
    return new_lines

 
# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate_alt, init_func=init_lines, frames=pars.nt_test_data, interval=100, blit=False)
if input_noise:
    anim.save('figs/compare_anim_' + problem + '_noise.gif')
else:
    anim.save('figs/compare_anim_' + problem + '.gif')
if input_noise:
    plt.savefig('figs/compare_last_frame_' + problem + '_noise.png', bbox_inches='tight')
else:
    plt.savefig('figs/compare_last_frame_' + problem + '.png', bbox_inches='tight')
plt.show()
