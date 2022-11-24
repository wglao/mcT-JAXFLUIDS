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

    f = open('linearadvection.json', 'r+')
    setup = json.load(f)
    f.close()

    setup['general']['end_time'] = pars.T
    setup['general']['save_dt'] = pars.dt
    setup['domain']['x']['range'] = [0.0, pars.x_max]
    setup['domain']['x']['cells'] = pars.N
    setup['initial_condition']['u'] = pars.u

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

        # randomize initial conditions
        # random sines
        # setup['initial_condition']['rho'] = "lambda x: 1 + 0.1*(" +\
        #     str(a_arr[iii,0]) + "*np.sin(2*np.pi*x*" + str(a_arr[iii,0]) + "/" + str(pars.x_max) + ")**2 + " + str(b_arr[iii,0]) + "*np.sin(2*np.pi*x*" + str(b_arr[iii,0]) + "/" + str(pars.x_max) + ")**2 + " +\
        #     str(a_arr[iii,1]) + "*np.sin(2*np.pi*x*" + str(a_arr[iii,1]) + "/" + str(pars.x_max) + ")**2 + " + str(b_arr[iii,1]) + "*np.sin(2*np.pi*x*" + str(b_arr[iii,1]) + "/" + str(pars.x_max) + ")**2 + " +\
        #     str(a_arr[iii,2]) + "*np.sin(2*np.pi*x*" + str(a_arr[iii,2]) + "/" + str(pars.x_max) + ")**2 + " + str(b_arr[iii,2]) + "*np.sin(2*np.pi*x*" + str(b_arr[iii,2]) + "/" + str(pars.x_max) + ")**2 + " +\
        #     str(a_arr[iii,3]) + "*np.sin(2*np.pi*x*" + str(a_arr[iii,3]) + "/" + str(pars.x_max) + ")**2 + " + str(b_arr[iii,3]) + "*np.sin(2*np.pi*x*" + str(b_arr[iii,3]) + "/" + str(pars.x_max) + ")**2 + " +\
        #     str(a_arr[iii,4]) + "*np.sin(2*np.pi*x*" + str(a_arr[iii,4]) + "/" + str(pars.x_max) + ")**2 + " + str(b_arr[iii,4]) + "*np.sin(2*np.pi*x*" + str(b_arr[iii,4]) + "/" + str(pars.x_max) + ")**2)"

        # random deltas
        a1 = str(np.abs(a_arr[iii,0]))
        a2 = str(np.abs(a_arr[iii,1]))
        a3 = str(np.abs(a_arr[iii,2]))
        a4 = str(np.abs(a_arr[iii,3]))
        a5 = str(np.abs(a_arr[iii,4]))
        b1 = str(np.abs(b_arr[iii,0]))
        b2 = str(np.abs(b_arr[iii,1]))

        setup['initial_condition']['rho'] = "lambda x: ((x>=0.2) & (x<=0.4)) * ( "+a1+"*(np.exp(-334.477 * (x-0.3-0.005)**2) + np.exp(-334.477 * (x - 0.3 + 0.005)**2) + "+a2+" * np.exp(-334.477 * (x - 0.3)**2))) + ((x>=0.6) & (x<=0.8)) * "+b1+" + "+b2+" + ((x>=1.0) & (x<=1.2)) * (1 - np.abs(10 * (x - 1.1))) + ((x>=1.4) & (x<=1.6)) * ("+a3+"*(np.sqrt(np.maximum( 1 - 100 * (x - 1.5 - 0.005)**2, 0)) + "+a4+"*np.sqrt(np.maximum( 1 - 100 * (x - 1.5 + 0.005)**2, 0)) + "+a5+"*np.sqrt(np.maximum( 1 - 100 * (x - 1.5)**2, 0))) ) + ~( ((x>=0.2) & (x<=0.4)) | ((x>=0.6) & (x<=0.8)) | ((x>=1.0) & (x<=1.2)) | ((x>=1.4) & (x<=1.6)) ) * 0.01"


        f = open('linearadvection.json', 'w+')
        json.dump(setup, f, indent=4)
        f.close()

        # don't need sim return because data is not being plotted
        _, initializer, sim_manager = run.setup("linearadvection.json", "numerical_setup.json")
        _,_ = run.sim(initializer, sim_manager)

# training sets
generate('train')

# test sets
generate('test')
