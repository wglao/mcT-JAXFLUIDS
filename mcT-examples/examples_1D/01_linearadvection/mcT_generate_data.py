import mcT_parameters as pars
import run_linearadvection as run
from jax import random
import numpy as np

a_train = random.normal(pars.key_data_train_a, (pars.num_train_samples, 5))
b_train = random.normal(pars.key_data_train_b, (pars.num_train_samples, 5))

a_test = random.normal(pars.key_data_test_a, (pars.num_test_samples, 5))
b_test = random.normal(pars.key_data_test_b, (pars.num_test_samples, 5))

# create random initial conditions and run jax fluids
# for each training and test dataset
input_reader, initializer, sim_manager = run.setup("linearadvection.json", "numerical_setup.json")

# training sets
initializer.input_reader.save_path = "./data/train"
for iii in pars.num_train_samples:

    # randomize initial conditions
    initializer.input_reader.initial_condition['rho'] = lambda x:\
        a_train[iii,0]*np.sin(2*np.pi*x) + b_train[iii,0]*np.sin(2*np.pi*x)+\
        a_train[iii,1]*np.sin(2*np.pi*x*2) + b_train[iii,1]*np.sin(2*np.pi*x*2)+\
        a_train[iii,2]*np.sin(2*np.pi*x*3) + b_train[iii,2]*np.sin(2*np.pi*x*3)+\
        a_train[iii,3]*np.sin(2*np.pi*x*4) + b_train[iii,3]*np.sin(2*np.pi*x*4)+\
        a_train[iii,4]*np.sin(2*np.pi*x*5) + b_train[iii,4]*np.sin(2*np.pi*x*5)

    # don't need return because data is not being plotted
    _,_ = run.sim(initializer, sim_manager)

# test sets
initializer.input_reader.save_path = "./data/test"
for iii in pars.num_test_samples:

    # randomize initial conditions
    initializer.input_reader.initial_condition['rho'] = lambda x:\
        a_test[iii,0]*np.sin(2*np.pi*x) + b_test[iii,0]*np.sin(2*np.pi*x)+\
        a_test[iii,1]*np.sin(2*np.pi*x*2) + b_test[iii,1]*np.sin(2*np.pi*x*2)+\
        a_test[iii,2]*np.sin(2*np.pi*x*3) + b_test[iii,2]*np.sin(2*np.pi*x*3)+\
        a_test[iii,3]*np.sin(2*np.pi*x*4) + b_test[iii,3]*np.sin(2*np.pi*x*4)+\
        a_test[iii,4]*np.sin(2*np.pi*x*5) + b_test[iii,4]*np.sin(2*np.pi*x*5)

    # don't need return because data is not being plotted
    _,_ = run.sim(initializer, sim_manager)

