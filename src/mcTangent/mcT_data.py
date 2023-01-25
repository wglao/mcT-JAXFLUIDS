from typing import Tuple, NamedTuple, Iterable, Union
import shutil, os, json, re
import matplotlib.pyplot as plt
import jax
from jax import vmap, pmap, lax
from jaxfluids import InputReader, Initializer, SimulationManager
from jaxfluids.post_process import load_data, create_lineplot
import jax.numpy as jnp
import numpy as np

import mcT_setup as setup

class Sim(NamedTuple):

    domain: str
    case: dict
    numerical: dict

    def load(self, quantities: Iterable[str] = ['density','velocityX','velocityY','velocityZ','pressure'], dtype: Union[str, type] = 'ARRAY'):
        out = load_data(self.domain,quantities)
        if isinstance(dtype,type):
            dtype = 'DICT' if dtype == dict else 'ARRAY'
        if dtype == 'DICT':
            return out
        if dtype == 'ARRAY':
            out = (out[0], out[1], out[2], np.array([out[3][quant] for quant in out[3].keys()]))
            return out

class Data():
    """
    Manages data for training and testing
    """

    def __init__(self,save_path):
        self.save_path = save_path
        self.check_sims()

    def check_sims(self) -> list:
        sim_list = os.listdir(self.save_path)
        sim_files = [os.path.join(self.save_path,sim) for sim in sim_list]
        sim_files.sort(key=lambda x: os.path.getmtime(x), reverse=False)
        sim_list = [os.path.split(sf)[1] for sf in sim_files]
        self.sims = sim_list

    def size(self) -> int:
        return len(self.sims)

    def next_sim(self) -> Sim:
        
        # takes sim out of queue and puts it at the back
        sim_path = self.sims.pop(0)
        # cycle sim to other end of list
        self.sims.append(sim_path)
        
        sim_name = re.split(r'-\d+',sim_path)[0] if re.split(r'-\d+',sim_path) else sim_path
        domain = os.path.join(self.save_path,sim_path,'domain')
        
        f = open(os.path.join(self.save_path,sim_path,sim_name+'.json'),'r')
        case_setup = json.load(f)
        f.close()
        
        f = open(os.path.join(self.save_path,sim_path,'numerical_setup.json'),'r')
        num_setup = json.load(f)
        f.close()

        return Sim(domain, case_setup, num_setup)

    def generate(self, reset=False):
        if reset:
            shutil.rmtree(self.save_path)
            
        self.check_sims()
        if self.size() < setup.cases.size():
            for _ in range(self.size()):
                _ = setup.cases.next()
            n_to_gen = setup.cases.size()

            for _ in range(n_to_gen):
                case_setup = setup.cases.next()
                num_setup = setup.numerical

                self.check_sims()
                if self.size() >= setup.num_train:
                    case_setup['general']['end_time'] = setup.t_max*setup.test_ratio

                input_reader = InputReader(case_setup, num_setup)
                initializer  = Initializer(input_reader)
                sim_manager  = SimulationManager(input_reader)

                buffer_dictionary = initializer.initialization()
                sim_manager.simulate(buffer_dictionary)
        del setup.cases
    
    def _load(self, sim: Sim):
        return sim.load()[3]

    def load_all(self):
        data_train = np.zeros((setup.num_train,5,setup.nt+1,setup.nx_fine,setup.ny_fine,setup.nz_fine))
        for ii in range(setup.num_train):
            data_train[ii,...] = self._load(self.next_sim())
        
        data_test = np.zeros((setup.num_test,5,int(setup.nt*setup.test_ratio)+1,setup.nx_fine,setup.ny_fine,setup.nz_fine))
        for ii in range(setup.num_test):
            data_test[ii,...] = self._load(self.next_sim())
        # data_train = data_test[:,:,:setup.nt+1,...]  #delete later
        self.check_sims()
        return data_train, data_test

            

data = Data(setup.save_path)

if __name__ == "__main__":
    
    data.generate()
