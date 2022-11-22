import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import wandb

from matplotlib import cm  # Colour map
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import jax
from jax.nn.initializers import normal, zeros
from jax import value_and_grad, vmap, random, jit, lax
import jax.numpy as jnp
from jax.example_libraries import stax, optimizers

import time
import pickle
import h5py
import re

from jaxfluids import InputReader, Initializer, SimulationManager
from jaxfluids.post_process import load_data, create_lineplot

# from jax.config import config
# config.update("jax_enable_x64", True)

#! Step : 0 - Generate_data_initilizers

# initialize physic parameters
# initialize parameters
mc_flag = False
noise_flag = False

import mcT_parameters as pars

mc_alpha = 1e5 if mc_flag else 0
noise_level = 0.02 if noise_flag else 0

# ? Step 0.2 - Uploading wandb
problem = 'linearadvection'
filename = problem + '_seq_n_mc_' + str(pars.n_seq_mc) +'_forward_mc_train_d' + str(pars.num_train) + '_alpha_' + str(mc_alpha) + '_lr_' + str(pars.learning_rate) + '_batch_' + str(pars.batch_size) + '_nseq_' + str(pars.n_seq) + '_layer_' + str(pars.layers) + 'neurons' + str(pars.units) + '_epochs_' + str(pars.num_epochs)

wandb.init(project="mcT-JAXFluids")
wandb.config.problem = problem
wandb.config.mc_alpha = pars.mc_alpha
wandb.config.learning_rate = pars.learning_rate
wandb.config.num_epochs = pars.num_epochs
wandb.config.batch_size = pars.batch_size
wandb.config.n_seq = pars.n_seq
wandb.config.layer = pars.layers
wandb.config.method = 'Dense_net'

#! Step 1: Loading data
# load h5 data and rearrange into dict

Train_data = np.zeros((pars.num_train_samples, pars.nt_train_data+1, pars.N))
train_path = 'data/train'
train_runs = os.listdir(train_path)

print('=' * 20 + ' >>')
print('Loading train data ...')
for ii, run in enumerate(train_runs):
    data_path = os.path.join(train_path, run, 'domain')
    saves = os.listdir(data_path)
    Train_times = np.zeros(pars.nt_train_data+1)
    
    for jj, save in enumerate(saves[0:pars.nt_train_data+1]):
        file = os.path.join(data_path, save)
        f = h5py.File(file, 'r')

        data = f['primes']['density'][()]
        Train_data[ii,jj,:] = data
        
        save_time = f['time'][()]
        Train_times[jj] = save_time

print(Train_data.shape)



if noise_level > 0:
    ns, nt, nx = Train_data.shape
    noise_vec = jax.random.normal(pars.key_data_noise, Train_data.shape)
    for ii in range(ns):
        for jj in range(nt):
                Train_data[ii,jj,:] = Train_data[ii,jj,:] + pars.noise_level * noise_vec[ii,jj,:] * np.max(Train_data[ii,jj,:])

Test_data = np.zeros((pars.num_test_samples, pars.nt_test_data+1, pars.N))
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
        Test_data[ii,jj,:] = data
        
        save_time = f['time'][()]
        Test_times[jj] = save_time

print(Test_data.shape)

#! Step 2: Building up a neural network
# Densely connected feed forward
N = pars.N
units = max(pars.units, N)
forward_pass_int, _ = stax.serial(
    stax.Dense(units, W_init=normal(0.02), b_init=zeros), stax.Relu,
    stax.Dense(N, W_init=normal(0.02), b_init=zeros),
)
_, init_params = forward_pass_int(random.PRNGKey(0), (N,))

W1, b1 = init_params[0]
W2, b2 = init_params[-1]

def ReLU(x):
    """ Rectified Linear Unit (ReLU) activation function """
    return jnp.maximum(0, x)

def Dense(inputs, W, b):
    return jnp.dot(inputs, W) + b

def forward_pass(params, u):
    W1, W2, b1, b2 = params
    u = Dense(ReLU(Dense(u, W1, b1)), W2, b2)
    return u

init_params = [W1, W2, b1, b2]

print('=' * 20 + ' >> Success!')

dt = pars.dt
dx = pars.dx
velo = pars.u
#! Step 3: Forward solver (single time step)
def single_solve_forward(un):
    # use different difference schemes for edge case
    lu = len(un)
    u = un + velo * (- dt / dx * (jnp.roll(un, -1) - jnp.roll(un, 1)) / 2 )
    uleft = un[0] + velo * (dt / dx / 2 * (3*un[0] - 4*un[1] + un[2]))
    uright = un[lu-1] + velo * (dt / -dx / 2 * (3*un[lu-1] - 4*un[lu-2] + un[lu-3]))
    u = u.at[0].set(uleft)
    u = u.at[lu-1].set(uright)
    return u

#@jit
def single_forward_pass(params, un):
    u = un - pars.facdt * dt * forward_pass(params, un)
    return u.flatten()


#! Step 4: Loss functions and relative error/accuracy rate function
# ? 4.1 For one time step data (1, 1, Nx)
def MSE(pred, true):
    return jnp.mean(jnp.square(pred - true))

# def squential_mc(i, args):
    
#     loss_mc, u_mc, u_ml, params = args
#     u_ml_next = single_forward_pass(params, u_ml)
#     u_mc_next = single_solve_forward(u_mc)
    
#     loss_mc += MSE(u_mc, u_ml_next)

#     return loss_mc, u_mc_next, u_ml_next, params

def squential_ml_second_phase(i, args):
    ''' I have checked this loss function!'''

    loss_ml, loss_mc, u_ml, u_true, params = args

    # This is u_mc for the current
    u_mc = single_solve_forward(u_ml)
    
    # This is u_ml for the next step
    u_ml_next = single_forward_pass(params, u_ml)
    
    # # The model-constrained loss 
    # loss_mc += MSE(u_mc, u_true[i+1,:]) 

    # The forward model-constrained loss
    # loss_mc, _, _, _ = lax.fori_loop(0, n_seq_mc, squential_mc, (loss_mc, u_mc, u_ml, params))
    loss_mc += MSE(u_mc, u_ml_next)
    
    # The machine learning term loss
    loss_ml += MSE(u_ml, u_true[i,:])

    return loss_ml, loss_mc, u_ml_next, u_true, params


def loss_one_sample_one_time(params, u):
    loss_ml = 0
    loss_mc = 0

    # first step prediction

    u_ml = single_forward_pass(params, u[0, :])

    # for the following steps up to sequential steps n_seq
    loss_ml,loss_mc, u_ml, _, _ = lax.fori_loop(1, pars.n_seq+1, squential_ml_second_phase, (loss_ml, loss_mc, u_ml, u, params))
    loss_ml += MSE(u_ml, u[-1, :])

    return loss_ml + pars.mc_alpha * loss_mc

loss_one_sample_one_time_batch = vmap(loss_one_sample_one_time, in_axes=(None, 0), out_axes=0)

# ? 4.2 For one sample of (1, Nt, Nx)
#@jit
def loss_one_sample(params, u_one_sample):
    return jnp.sum(loss_one_sample_one_time_batch(params, u_one_sample))

loss_one_sample_batch = vmap(loss_one_sample, in_axes=(None, 0), out_axes=0)

# ? 4.3 For the whole data (n_samples, Nt, Nx)
# ? This step transform data to disired shape for training (n_train_samples, Nt, Nx) -> (n_train_samples, Nt, n_seq, Nx)
#@jit
def transform_one_sample_data(u_one_sample):
    u_out = jnp.zeros((pars.nt_train_data - pars.n_seq - 1, pars.n_seq + 2, N))
    for i in range(pars.nt_train_data-pars.n_seq-1):
        u_out = u_out.at[i, :, :].set(u_one_sample[i:i + pars.n_seq + 2, :])
    return u_out

transform_one_sample_data_batch = vmap(transform_one_sample_data, in_axes=0)

#@jit
def LossmcDNN(params, data):
    return jnp.sum(loss_one_sample_batch(params, transform_one_sample_data_batch(data)))


#! Step 5: Computing test error, predictions over all time steps
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


@jit
def test_acc(params, Test_set):
    return MSE(neural_solver_batch(params, Test_set), Test_set)

#! Step 6: Epoch loops fucntions and training settings
def body_fun(i, args):
    loss, opt_state, data = args

    data_batch = lax.dynamic_slice_in_dim(data, i * pars.batch_size, pars.batch_size)

    loss, gradients = value_and_grad(LossmcDNN)(
        opt_get_params(opt_state), data_batch)

    opt_state = opt_update(i, gradients, opt_state)

    return loss/pars.batch_size, opt_state, data


@jit
def run_epoch(opt_state, data):
    loss = 0
    return lax.fori_loop(0, num_batches, body_fun, (loss, opt_state, data))


def TrainModel(train_data, test_data, num_epochs, opt_state):

    test_accuracy_min = 100
    epoch_min = 1

    for epoch in range(1, num_epochs+1):
        
        t1 = time.time()
        train_loss, opt_state, _ = run_epoch(opt_state, train_data)
        t2 = time.time()

        test_accuracy = test_acc(opt_get_params(opt_state), test_data)

        if test_accuracy_min >= test_accuracy:
            test_accuracy_min = test_accuracy
            epoch_min = epoch
            optimal_opt_state = opt_state

        if epoch % 1000 == 0:  # Print MSE every 1000 epochs
            print("Data_d {:d} n_seq {:d} batch {:d} time {:.2e}s loss {:.2e} TE {:.2e}  TE_min {:.2e} EPmin {:d} EP {} ".format(
                pars.num_train, pars.n_seq, pars.batch_size, t2 - t1, train_loss, test_accuracy, test_accuracy_min, epoch_min, epoch))

        wandb.log({"Train loss": float(train_loss), "Test Error": float(test_accuracy), 'TEST MIN': float(test_accuracy_min), 'Epoch' : float(epoch)})

    return optimal_opt_state, opt_state


num_complete_batches, leftover = divmod(pars.num_train, pars.batch_size)
num_batches = num_complete_batches + bool(leftover)

opt_int, opt_update, opt_get_params = optimizers.adam(pars.learning_rate)
opt_state = opt_int(init_params)

best_opt_state, end_opt_state = TrainModel(Train_data, Test_data, pars.num_epochs, opt_state)

optimum_params = opt_get_params(best_opt_state)
End_params = opt_get_params(end_opt_state)
# from jax.example_libraries.optimizers import optimizers

trained_params = optimizers.unpack_optimizer_state(end_opt_state)
pickle.dump(trained_params, open('Network/End_' + filename, "wb"))

trained_params = optimizers.unpack_optimizer_state(best_opt_state)
pickle.dump(trained_params, open('Network/Best_' + filename, "wb"))


# %% Plot predictions
U_pred = neural_solver_batch(optimum_params, Test_data)[0, :, :]
U_true = Test_data[0, :, :]

x = np.linspace(0, 1, N)


def plot_compare(U_True, U_Pred, filename):

    fig = plt.figure(figsize=(32,10))
    fig.patch.set_facecolor('xkcd:white')

    # Compare solutions
    for i in range(pars.n_plot):
        ut = jnp.reshape(U_True[pars.Plot_Steps[i], :], (N, 1))
        up = jnp.reshape(U_Pred[pars.Plot_Steps[i], :], (N, 1))
        ax = fig.add_subplot(1, pars.n_plot, i+1)
        l1 = ax.plot(x, ut, '-', linewidth=2, label='True')
        l2 = ax.plot(x, up, '--', linewidth=2, label='Predicted')
        ax.set_aspect('auto', adjustable='box')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title('t = ' + str(pars.Plot_Steps[i]))

        if i == 1:
            handles, labels = ax.get_legend_handles_labels()
            fig.legend(handles, labels, loc='upper center')

    plt.savefig('figs/' + filename + '.png', bbox_inches='tight')


plot_compare(U_true, U_pred, filename)
