import matplotlib.pyplot as plt
from jaxfluids import InputReader, Initializer, SimulationManager
from jaxfluids.post_process import load_data, create_lineplot
import jax.numpy as jnp
import mcT_adv_setup as setup
import mcT_adv_data as dat

# SETUP SIMULATION
sim = dat.data.next_sim()
input_reader = InputReader(sim.case, sim.numerical)
# initializer  = Initializer(input_reader)
sim_manager  = SimulationManager(input_reader)

# RUN SIMULATION
# buffer_dictionary = initializer.initialization()
# sim_manager.simulate(buffer_dictionary)

# # LOAD DATA
# path = 'results/linearadvection/domain'
# path = sim_manager.output_writer.save_path_domain
quantities = ["density"]
cell_centers, cell_sizes, times, data = sim.load(quantities=quantities, dtype='DICT')

# print(data['density'][1,...])
# print(data['density'].shape)
# test = jnp.reshape(data['density'], (3,200))
# print(test[1,:])

# PLOT
nrows_ncols = (1,1)
create_lineplot(data, cell_centers, times, nrows_ncols=nrows_ncols, interval=100)

fig, ax = plt.subplots()
ax.plot(cell_centers[0], data["density"][-1,:,0,0])
ax.plot(cell_centers[0], data["density"][0,:,0,0], color="black")
plt.show()