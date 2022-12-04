from typing import Tuple, NamedTuple
import shutil, os, json, re
import matplotlib.pyplot as plt
from jaxfluids import InputReader, Initializer, SimulationManager
from jaxfluids.post_process import load_data, create_lineplot
import jax.numpy as jnp

import mcT_adv_setup as setup

class Sim(NamedTuple):

    domain: str
    case: dict
    numerical: dict

class Data():
    """
    Manages data for training and testing
    """

    def __init__(self,save_path):
        
        self.save_path = save_path
        self.check_sims()

    def check_sims(self) -> list:
        self.sims = os.listdir(self.save_path)
        self.sims.sort()

    def next_sim(self) -> Sim:

        sim_path = self.sims.pop()
        sim_name = re.split(r'-\d+',sim_path)[0] if re.split(r'-\d+',sim_path) else sim_path
        domain = os.path.join(self.save_path,sim_path,'domain')
        
        f = open(os.path.join(self.save_path,sim_path,sim_name+'.json'),'r')
        case_setup = json.load(f)
        f.close()
        
        f = open(os.path.join(self.save_path,sim_path,'numerical_setup.json'),'r')
        num_setup = json.load(f)
        f.close()

        return Sim(domain, case_setup, num_setup)

    def generate(self):
        shutil.rmtree(self.save_path)

        for ii in range(setup.cases.size()):
            case_setup = setup.cases.next()
            num_setup = setup.numerical

            input_reader = InputReader(case_setup, num_setup)
            initializer  = Initializer(input_reader)
            sim_manager  = SimulationManager(input_reader)

            buffer_dictionary = initializer.initialization()
            sim_manager.simulate(buffer_dictionary)

data = Data(setup.save_path)

if __name__ == "__main__":
    
    data.generate()
