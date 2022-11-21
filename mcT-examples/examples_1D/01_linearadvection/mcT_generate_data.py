import json
import os
import mcT_parameters as pars
import run_linearadvection as run
from jax import random
import numpy as np
import jax.numpy as jnp

from jaxfluids import InputReader, Initializer, SimulationManager

a_train = random.normal(pars.key_data_train_a, (pars.num_train_samples, 5))
b_train = random.normal(pars.key_data_train_b, (pars.num_train_samples, 5))

a_test = random.normal(pars.key_data_test_a, (pars.num_test_samples, 5))
b_test = random.normal(pars.key_data_test_b, (pars.num_test_samples, 5))

# create random initial conditions and run jax fluids
# for each training and test dataset

f = open('linearadvection.json', 'r+')
setup = json.load(f)
f.close()

setup['general']['end_time'] = pars.T
setup['general']['save_dt'] = pars.dt
setup['domain']['x']['range'] = [0.0, pars.x_max]
setup['domain']['x']['cells'] = pars.N
setup['initial_condition']['u'] = pars.u
# setup['initial_condition']['v'] = pars.v
# setup['initial_condition']['w'] = pars.w

# training sets

setup['general']['save_path'] = "./data/train"
for iii in range(pars.num_train_samples):

    # randomize initial conditions
    setup['initial_condition']['rho'] = "lambda x: " + str(np.max(np.abs(a_test[iii,:]))) + " + " + str(np.max(np.abs(b_test[iii,:]))) + " + 0.1*(" +\
        str(a_train[iii,0]) + "*np.sin(2*np.pi*x*" + str(a_train[iii,0]) + ") + " + str(b_train[iii,0]) + "*np.sin(2*np.pi*x*" + str(b_train[iii,0]) + ") + " +\
        str(a_train[iii,1]) + "*np.sin(2*np.pi*x*" + str(a_train[iii,1]) + ") + " + str(b_train[iii,1]) + "*np.sin(2*np.pi*x*" + str(b_train[iii,1]) + ") + " +\
        str(a_train[iii,2]) + "*np.sin(2*np.pi*x*" + str(a_train[iii,2]) + ") + " + str(b_train[iii,2]) + "*np.sin(2*np.pi*x*" + str(b_train[iii,2]) + ") + " +\
        str(a_train[iii,3]) + "*np.sin(2*np.pi*x*" + str(a_train[iii,3]) + ") + " + str(b_train[iii,3]) + "*np.sin(2*np.pi*x*" + str(b_train[iii,3]) + ") + " +\
        str(a_train[iii,4]) + "*np.sin(2*np.pi*x*" + str(a_train[iii,4]) + ") + " + str(b_train[iii,4]) + "*np.sin(2*np.pi*x*" + str(b_train[iii,4]) + "))"

    f_new = open('next_run.json', 'w+')
    json.dump(setup, f_new, indent=4)
    f_new.close()
    
    # don't need sim return because data is not being plotted
    _, initializer, sim_manager = run.setup("next_run.json", "numerical_setup.json")
    _,_ = run.sim(initializer, sim_manager)

# test sets

setup['general']['save_path'] = "./data/test"
for iii in range(pars.num_test_samples):

    # randomize initial conditions
    setup['initial_condition']['rho'] = "lambda x: 1 + " +\
        str(a_test[iii,0]) + "*0.1*np.sin(2*np.pi*x*" + str(a_test[iii,0]) + ")**2 + " + str(b_test[iii,0]) + "*0.1*np.sin(2*np.pi*x*" + str(b_test[iii,0]) + ")**2 + " +\
        str(a_test[iii,1]) + "*0.1*np.sin(2*np.pi*x*" + str(a_test[iii,1]) + ")**2 + " + str(b_test[iii,1]) + "*0.1*np.sin(2*np.pi*x*" + str(b_test[iii,1]) + ")**2 + " +\
        str(a_test[iii,2]) + "*0.1*np.sin(2*np.pi*x*" + str(a_test[iii,2]) + ")**2 + " + str(b_test[iii,2]) + "*0.1*np.sin(2*np.pi*x*" + str(b_test[iii,2]) + ")**2 + " +\
        str(a_test[iii,3]) + "*0.1*np.sin(2*np.pi*x*" + str(a_test[iii,3]) + ")**2 + " + str(b_test[iii,3]) + "*0.1*np.sin(2*np.pi*x*" + str(b_test[iii,3]) + ")**2 + " +\
        str(a_test[iii,4]) + "*0.1*np.sin(2*np.pi*x*" + str(a_test[iii,4]) + ")**2 + " + str(b_test[iii,4]) + "*0.1*np.sin(2*np.pi*x*" + str(b_test[iii,4]) + ")**2"

    f_new = open('next_run.json', 'w+')
    json.dump(setup, f_new, indent=4)
    f_new.close()

    # don't need sim return because data is not being plotted
    _, initializer, sim_manager = run.setup("next_run.json", "numerical_setup.json")
    _,_ = run.sim(initializer, sim_manager)

# clean up
os.remove("next_run.json")
