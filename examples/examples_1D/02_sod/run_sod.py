import matplotlib.pyplot as plt
from jaxfluids import InputReader, Initializer, SimulationManager
from jaxfluids.post_process import load_data, create_lineplot

# SETUP SIMULATION
def setup(problem, numerical):
    input_reader = InputReader(problem, numerical)
    initializer  = Initializer(input_reader)
    sim_manager  = SimulationManager(input_reader)
    return input_reader, initializer, sim_manager

# RUN SIMULATION
def sim(initializer, sim_manager):
    buffer_dictionary = initializer.initialization()
    sim_manager.simulate(buffer_dictionary)
    return buffer_dictionary, sim_manager

# LOAD DATA
def load(sim_manager):
    path = sim_manager.output_writer.save_path_domain
    quantities = ["density", "velocityX", "pressure"]
    cell_centers, cell_sizes, times, data_dict = load_data(path, quantities)
    return cell_centers, cell_sizes, times, data_dict

# PLOT
def plot(cell_centers, cell_sizes, times, data_dict):
    nrows_ncols = (1,3)
    create_lineplot(data_dict, cell_centers, times, nrows_ncols=nrows_ncols, interval=100)

    fig, ax = plt.subplots(ncols=3)
    ax[0].plot(cell_centers[0], data_dict["density"][-1  ,:,0,0])
    ax[1].plot(cell_centers[0], data_dict["velocityX"][-1,:,0,0])
    ax[2].plot(cell_centers[0], data_dict["pressure"][-1 ,:,0,0])
    plt.show()

if __name__ == "__main__":
    input_reader, initializer, sim_manager = setup("sod.json", "numerical_setup.json")
    buffer_dictionary, sim_manager = sim(initializer, sim_manager)
    cell_centers, cell_sizes, times, data_dict = load(sim_manager)
    plot(cell_centers, times, data_dict)