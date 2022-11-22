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
    quantities = ["density"]
    cell_centers, cell_sizes, times, data_dict = load_data(path, quantities)
    return cell_centers, cell_sizes, times, data_dict

# PLOT
def plot(cell_centers, times, data_dict):
    nrows_ncols = (1,1)
    create_lineplot(data_dict, cell_centers, times, nrows_ncols=nrows_ncols, interval=100)

    fig, ax = plt.subplots()
    ax.plot(cell_centers[0], data_dict["density"][-1,:,0,0])
    # ax.plot(cell_centers[0], data_dict["density"][1,:,0,0], '--')
    ax.plot(cell_centers[0], data_dict["density"][0,:,0,0], color="black")
    plt.show()

if __name__ == "__main__":
    input_reader, initializer, sim_manager = setup("linearadvection.json", "numerical_setup.json")
    buffer_dictionary, sim_manager = sim(initializer, sim_manager)
    cell_centers, cell_sizes, times, data_dict = load(sim_manager)
    plot(cell_centers, times, data_dict)
