import json
import os
import mcT_parameters as pars
import run_sod as run
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
f = open('numerical_setup.json', 'r+')
num_setup = json.load(f)
f.close()

# use fixed timesteps for compatability with mcTangent
if 'fixed_timestep' not in list(num_setup['conservatives']['time_integration'].keys()):
    num_setup['conservatives']['time_integration']['fixed_timestep'] = pars.dt/10

    f = open('numerical_setup.json', 'w+')
    json.dump(num_setup, f, indent=4)
    f.close()

def generate(run_type):
    a_arr = np.array([])
    b_arr = np.array([])
    num_samples = 0

    f = open('sod.json', 'r+')
    setup = json.load(f)
    f.close()

    setup['general']['end_time'] = pars.T
    setup['general']['save_dt'] = pars.dt
    setup['domain']['x']['range'] = [0.0, pars.x_max]
    setup['domain']['x']['cells'] = pars.N

    if run_type == 'train':
        setup['general']['save_path'] = "./data/train"
        a_arr = a_train
        b_arr = b_train
        num_samples = pars.num_train_samples
    else:
        setup['general']['save_path'] = "./data/test"
        a_arr = a_test
        b_arr = b_test
        num_samples = pars.num_test_samples
    
    for iii in range(num_samples):

        # random deltas
        a1 = str(np.abs(a_arr[iii,0]))
        a2 = str(np.abs(a_arr[iii,1]))
        a3 = str(np.abs(a_arr[iii,2]))
        b1 = str(np.abs(b_arr[iii,0]))
        b2 = str(np.abs(b_arr[iii,1]))
        b3 = str(np.abs(b_arr[iii,2]))

        setup['initial_condition']['rho'] = "lambda x: 4*"+a1+"*(x <= "+a2+") + 0.25*"+a3+"*(x > "+a2+")"
        setup['initial_condition']['p'] = "lambda x: 4*"+b1+"*(x <= "+b2+") + 0.25*"+b3+"*(x > "+b2+")"

        f = open('sod.json', 'w+')
        json.dump(setup, f, indent=4)
        f.close()

        # don't need sim return because data is not being plotted
        _, initializer, sim_manager = run.setup("sod.json", "numerical_setup.json")
        _,_ = run.sim(initializer, sim_manager)

# training sets
generate('train')

# test sets
generate('test')
