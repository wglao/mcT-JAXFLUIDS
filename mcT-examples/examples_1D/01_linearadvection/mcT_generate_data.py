import mcT_parameters as pars
import run_linearadvection as run
from jax import random
import numpy as np
import jax.numpy as jnp

a_train = random.normal(pars.key_data_train_a, (pars.num_train_samples, 5))
b_train = random.normal(pars.key_data_train_b, (pars.num_train_samples, 5))

a_test = random.normal(pars.key_data_test_a, (pars.num_test_samples, 5))
b_test = random.normal(pars.key_data_test_b, (pars.num_test_samples, 5))

# create random initial conditions and run jax fluids
# for each training and test dataset
input_reader, initializer, sim_manager = run.setup("linearadvection.json", "numerical_setup.json")

# training sets
initializer.input_reader.end_time = pars.T
initializer.input_reader.save_dt = pars.dt
initializer.input_reader.domain_size['x'][1] = pars.x_max
initializer.input_reader.number_of_cells = jnp.array([pars.N, 1, 1], dtype=int)
initializer.input_reader.initial_condition['u'] = pars.u
# initializer.input_reader.initial_condition['v'] = pars.v
# initializer.input_reader.initial_condition['w'] = pars.w

initializer.input_reader.save_path = "./data/train"
for iii in range(pars.num_train_samples):

    # randomize initial conditions
    initializer.input_reader.initial_condition['rho'] = lambda x:\
        a_train[iii,0]*np.sin(2*np.pi*x/pars.x_max) + b_train[iii,0]*np.sin(2*np.pi*x/pars.x_max)+\
        a_train[iii,1]*np.sin(2*np.pi*x*2/pars.x_max) + b_train[iii,1]*np.sin(2*np.pi*x*2/pars.x_max)+\
        a_train[iii,2]*np.sin(2*np.pi*x*3/pars.x_max) + b_train[iii,2]*np.sin(2*np.pi*x*3/pars.x_max)+\
        a_train[iii,3]*np.sin(2*np.pi*x*4/pars.x_max) + b_train[iii,3]*np.sin(2*np.pi*x*4/pars.x_max)+\
        a_train[iii,4]*np.sin(2*np.pi*x*5/pars.x_max) + b_train[iii,4]*np.sin(2*np.pi*x*5/pars.x_max)

    # don't need return because data is not being plotted
    _,_ = run.sim(initializer, sim_manager)

# test sets
initializer.input_reader.save_path = "./data/test"
for iii in range(pars.num_test_samples):

    # randomize initial conditions
    initializer.input_reader.initial_condition['rho'] = lambda x:\
        a_test[iii,0]*np.sin(2*np.pi*x) + b_test[iii,0]*np.sin(2*np.pi*x)+\
        a_test[iii,1]*np.sin(2*np.pi*x*2) + b_test[iii,1]*np.sin(2*np.pi*x*2)+\
        a_test[iii,2]*np.sin(2*np.pi*x*3) + b_test[iii,2]*np.sin(2*np.pi*x*3)+\
        a_test[iii,3]*np.sin(2*np.pi*x*4) + b_test[iii,3]*np.sin(2*np.pi*x*4)+\
        a_test[iii,4]*np.sin(2*np.pi*x*5) + b_test[iii,4]*np.sin(2*np.pi*x*5)

    # don't need return because data is not being plotted
    _,_ = run.sim(initializer, sim_manager)

