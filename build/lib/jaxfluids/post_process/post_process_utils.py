#*------------------------------------------------------------------------------*
#* JAX-FLUIDS -                                                                 *
#*                                                                              *
#* A fully-differentiable CFD solver for compressible two-phase flows.          *
#* Copyright (C) 2022  Deniz A. Bezgin, Aaron B. Buhendwa, Nikolaus A. Adams    *
#*                                                                              *
#* This program is free software: you can redistribute it and/or modify         *
#* it under the terms of the GNU General Public License as published by         *
#* the Free Software Foundation, either version 3 of the License, or            *
#* (at your option) any later version.                                          *
#*                                                                              *
#* This program is distributed in the hope that it will be useful,              *
#* but WITHOUT ANY WARRANTY; without even the implied warranty of               *
#* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the                *
#* GNU General Public License for more details.                                 *
#*                                                                              *
#* You should have received a copy of the GNU General Public License            *
#* along with this program.  If not, see <https://www.gnu.org/licenses/>.       *
#*                                                                              *
#*------------------------------------------------------------------------------*
#*                                                                              *
#* CONTACT                                                                      *
#*                                                                              *
#* deniz.bezgin@tum.de // aaron.buhendwa@tum.de // nikolaus.adams@tum.de        *
#*                                                                              *
#*------------------------------------------------------------------------------*
#*                                                                              *
#* Munich, April 15th, 2022                                                     *
#*                                                                              *
#*------------------------------------------------------------------------------*

import os
from typing import List, Dict, Tuple

import h5py
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

def load_data(path: str, quantities: List, start: int = 0, end: int = None,
        N: int = 1) -> Tuple[List, List, List, Dict]:

    """Loads .h5 result files generated by a jaxfluids simulation.

    :param path: Path to the folder containing the .h5 files for the time snapshots
    :type path: str
    :param quantities: Quantities to load
    :type quantities: List
    :param start: Starting time snapshot to load, defaults to 0
    :type start: int, optional
    :param end: Ending time snapshot to load, defaults to None
    :type end: int, optional
    :param N: Interval of time snapshots to load, defaults to 1
    :type N: int, optional
    :return: Cell centers coordinates, cell sizes, times and dictionary of buffers
    :rtype: Tuple
    """

    # SANITY CHECK
    available_quantities = {
        "primes" : ["density", "velocity", "velocityX", "velocityY", "velocityZ", "pressure", "temperature"],
        "cons": ["mass", "momentum", "momentumX", "momentumY", "momentumZ", "energy"],
        "levelset": ["levelset", "volume_fraction", "mask_real", "normal", "interface_pressure", "interface_velocity"],
        "real_fluid": [ "real_density", "real_velocity", "real_velocityX", "real_velocityY", "real_velocityZ", "real_pressure", "real_temperature",
                        "real_mass", "real_momentum", "real_momentumX", "real_momentumY", "real_momentumZ", "real_energy" ],
        "miscellaneous": ["mach_number", "schlieren", "absolute_velocity", "vorticity", "absolute_vorticity"],
    }
    for quant in quantities:
        flag = False
        for key in available_quantities:
            if quant in available_quantities[key]:
                flag = True
        assert flag, "%s not available." % quant

    # SEARCH DIRECTORY FOR FILES
    files = []
    times = []
    for file in os.listdir(path):
        if file.endswith("h5"):
            if "nan" in file:
                continue 
            files.append(file)
            times.append(float(os.path.splitext(os.path.basename(file))[0][5:]))
    indices     = np.argsort(np.array(times))
    times       = np.array(times)[indices][start:end:N]
    files       = np.array(files)[indices][start:end:N]
    no_times    = len(times)


    with h5py.File(os.path.join(path, files[0]), "r") as h5file:

        # DOMAIN INFORMATION
        x = h5file["mesh/gridX"][:]
        y = h5file["mesh/gridY"][:]
        z = h5file["mesh/gridZ"][:]
        dx = h5file["mesh/cellsizeX"][()]
        dy = h5file["mesh/cellsizeY"][()]
        dz = h5file["mesh/cellsizeZ"][()]

        cell_centers    = [x, y, z]
        cell_sizes      = [dx, dy, dz]

        nx = len(x)
        ny = len(y)
        nz = len(z)

        # CHECK IF PRESENT H5FILES CONTAIN FLUID-FLUID RESULTS
        fluid_fluid = False
        if "real_fluid" in h5file.keys():
            fluid_fluid = True
        elif "primes" in h5file.keys():
            first_key = list(h5file["primes"].keys())[0]
            if first_key[-1] == "0":
                fluid_fluid = True
        elif "cons" in h5file.keys():
            first_key = list(h5file["cons"].keys())[0]
            if first_key[-1] == "0":
                fluid_fluid = True

    # CREATE BUFFERS
    data_dictionary = {}
    for quant in quantities:
        for key in available_quantities:
            if quant in available_quantities[key]:
                quantity_type = key
                break

        # PRIMITIVES
        if quantity_type in ["primes", "cons"]:
            if fluid_fluid:
                if quant in ["velocity", "momentum"]:
                    data_dictionary[quant] = np.zeros((no_times, 2, 3, nx, ny, nz))
                else:
                    data_dictionary[quant] = np.zeros((no_times, 2, nx, ny, nz))
            else:
                if quant in ["velocity", "momentum"]:
                    data_dictionary[quant] = np.zeros((no_times, 3, nx, ny, nz))
                else:
                    data_dictionary[quant] = np.zeros((no_times, nx, ny, nz))

        # REAL FLUID
        elif quantity_type in ["levelset", "real_fluid"]:
            if quant in ["normal", "real_velocity", "real_momentum"]:
                data_dictionary[quant] = np.zeros((no_times, 3, nx, ny, nz))
            elif quant == "interface_pressure":
                data_dictionary[quant] = np.zeros((no_times, 2, nx, ny, nz))
            else:
                data_dictionary[quant] = np.zeros((no_times, nx, ny, nz))

        # MISCELLANEOUS
        elif quantity_type == "miscellaneous":
            if quant == "vorticity":
                data_dictionary[quant] = np.zeros((no_times, 3, nx, ny, nz))
            else:
                data_dictionary[quant] = np.zeros((no_times, nx, ny, nz))

        # MASS FLOW FORCING
        elif quant == "mass_flow_forcing":
            data_dictionary[quant] = np.zeros(no_times)

    # FILL BUFFERS
    for i, file in enumerate(files):
        print("Loading time snapshot %.4e" % times[i])
        with h5py.File(os.path.join(path, file), "r") as h5file:

            for quant in quantities:
                
                # LEVELSET
                if quant in available_quantities["levelset"]:
                    data_dictionary[quant][i] = h5file["levelset/" + quant][:].T

                # PRIMITIVES
                elif quant in available_quantities["primes"]:
                    for j in range(2 if fluid_fluid else 1):
                        quant_name_h5 = quant + "_%d" % j if fluid_fluid else quant
                        s_1 = np.s_[i,j] if fluid_fluid else np.s_[i]
                        data_dictionary[quant][s_1] = h5file["primes/" + quant_name_h5][:].T

                # CONSERVATIVES
                elif quant in available_quantities["cons"]:
                    for j in range(2 if fluid_fluid else 1):
                        quant_name_h5 = quant + "_%d" % j if fluid_fluid else quant
                        s_1 = np.s_[i,j] if fluid_fluid else np.s_[i]
                        data_dictionary[quant][s_1] = h5file["cons/" + quant_name_h5][:].T

                # REAL FLUID
                elif quant in available_quantities["real_fluid"]:
                    quant_name_h5 = quant.split("_")[-1]
                    data_dictionary[quant][i] = h5file["real_fluid/" + quant_name_h5][:].T

                # MISCELLANEOUS
                elif quant in available_quantities["miscellaneous"]:
                    data_dictionary[quant][i] = h5file["miscellaneous/" + quant][:].T                    

                # MASS FLOW FORCING
                elif quant == "mass_flow_forcing":
                    data_dictionary[quant][i] = h5file["mass_flow_forcing/scalar_value"][()]

    return cell_centers, cell_sizes, times, data_dictionary


def create_contourplot(data_dictionary: Dict, cell_centers: List, times: np.ndarray,
        levelset: np.ndarray = None, nrows_ncols: Tuple = None, plane: str = "xy",
        plane_value: float = 0.0, static_time: float = None, save_fig: str = None, save_anim: str = None,
        interval: int = 50, dpi: int = 100) -> None:
    """Creates a subplot of 2D pcolormesh plots and subsequently creates an animation.
    If the levelset argument is provided, the interface
    will be visualized as an isoline. If the static_time argument is provided, no animation is created,
    but a plot at this time will be shown.

    :param data_dictionary: Data buffers
    :type data_dictionary: Dict
    :param cell_centers: Cell center coordinates
    :type cell_centers: List
    :param times: Times
    :type times: np.ndarray
    :param levelset: Levelset buffer, defaults to None
    :type levelset: np.ndarray, optional
    :param nrows_ncols: Shape of the subplot, defaults to None
    :type nrows_ncols: Tuple, optional
    :param plane: Plane to slice, defaults to "xy"
    :type plane: str, optional
    :param plane_value: Value of remaining axis to slice, defaults to 0.0
    :type plane_value: float, optional
    :param static_time: Static time to plot, defaults to None
    :type static_time: float, optional
    :param save_fig: Path to save the static time figure, defaults to None
    :type save_fig: str, optional
    :param save_anim: Path to save the animation, defaults to None
    :type save_anim: str, optional
    :param interval: Interval in ms for the animation, defaults to 50
    :type interval: int, optional
    :param dpi: Resolution in dpi, defaults to 100
    :type dpi: int, optional
    """

    # CELL CENTERS
    x, y, z = cell_centers
    
    # ROWS AND COLUMNS
    nrows_ncols = (1, len(data_dictionary.keys())) if nrows_ncols == None else nrows_ncols

    # SLICE PLANE
    if plane == "xy":
        X1, X2 = np.meshgrid(x,y,indexing="ij")
        index = np.argmin(np.abs(z - plane_value))
        data = [ data_dictionary[quant][:,:,:,index] for quant in data_dictionary.keys() ]
        levelset = levelset[:,:,:,index] if type(levelset) != type(None) else None
    elif plane == "xz":
        X1, X2 = np.meshgrid(x,z,indexing="ij")
        index = np.argmin(np.abs(y - plane_value))
        data = [ data_dictionary[quant][:,:,index,:] for quant in data_dictionary.keys() ]
        levelset = levelset[:,:,index,:] if type(levelset) != type(None) else None
    elif plane == "yz":
        X1, X2 = np.meshgrid(y,z,indexing="ij")
        index = np.argmin(np.abs(x - plane_value))
        data = [ data_dictionary[quant][:,index,:,:] for quant in data_dictionary.keys() ]
        levelset = levelset[:,index,:,:] if type(levelset) != type(None) else None
    else:
        assert False, "plane does not exist"

    # TITLES
    titles = data_dictionary.keys()

    # MINMAX VALUES
    minmax = []
    for i in range(len(data)):
        d = data[i]
        minmax.append([np.min(d), np.max(d) + np.finfo(float).eps])

    # STATIC PLOT
    if static_time != None:
        index_time = np.argmin(np.abs(times - static_time))
    else:
        index_time = 0

    # CREATE SUBPLOTS
    fig, axes, quadmesh_list, pciset_list = create_figure_2D(data, levelset,
        nrows_ncols, titles, X1, X2, minmax, index_time)

    # SAVE STATIC PLOT
    if save_fig != None:
        fig.savefig("%s.pdf" % save_fig, bbox_inches="tight")
    if static_time != None:
        plt.show()
    
    # CREATE ANIMATION
    ani = FuncAnimation(fig, update_2D, frames=data[0].shape[0], fargs=(X1, X2,
        data, levelset, axes, quadmesh_list, pciset_list), interval=interval, blit=True, repeat=True)
    if save_anim is not None:
        if save_anim.endswith((".avi", ".mp4")):
            ani.save(save_anim, dpi=dpi)
        else:
            ani.save("%s.gif" % save_anim, dpi=dpi)

    plt.show()

def create_lineplot(data_dictionary: List, cell_centers: List, times: np.ndarray, nrows_ncols: Tuple,
        axis: str="x", values: List=[0.0,0.0], static_time: float = None, save_fig: str = None,
        save_anim: str = None, interval: int = 50, dpi: int = 100) -> None:
    """Creates a subplot of 1D line plots. If the levelset argument is provided, the interface
    will be visualized as an isoline. If the static_time argument is provided, no animation is created,
    but a plot at this time will be shown.

    :param data_dictionary: Data buffers
    :type data_dictionary: List
    :param cell_centers: Cell center coordinates
    :type cell_centers: List
    :param times: Time buffer
    :type times: np.ndarray
    :param nrows_ncols: Shape of subplots
    :type nrows_ncols: Tuple
    :param axis: Axis to slice, defaults to "x"
    :type axis: str, optional
    :param values: Values of remaining axis to slice, defaults to [0.0,0.0]
    :type values: List, optional
    :param static_time: Static time, defaults to None
    :type static_time: float, optional
    :param save_fig: Path to save static plot, defaults to None
    :type save_fig: str, optional
    :param save_anim: Path to save animation, defaults to None
    :type save_anim: str, optional
    :param interval: Interval in ms for animation, defaults to 50
    :type interval: int, optional
    :param dpi: Resolution in dpi, defaults to 100
    :type dpi: int, optional
    """

    # SLICE AXIS
    x, y, z = cell_centers
    if axis == "x":
        coord = x
        index1 = np.argmin(np.abs(y - values[0]))
        index2 = np.argmin(np.abs(z - values[1]))
        data = [ data_dictionary[quant][:,:,index1,index2] for quant in data_dictionary.keys() ]
    elif axis == "y":
        coord = y
        index1 = np.argmin(np.abs(x - values[0]))
        index2 = np.argmin(np.abs(z - values[1]))
        data = [ data_dictionary[quant][:,index1,:,index2] for quant in data_dictionary.keys() ]
    elif axis == "z":
        coord = z
        index1 = np.argmin(np.abs(x - values[0]))
        index2 = np.argmin(np.abs(y - values[1]))
        data = [ data_dictionary[quant][:,index1,index2,:] for quant in data_dictionary.keys() ]
    else:
        assert False, "axis does not exist"

    # STATIC TIME INDEX
    if static_time != None:
        index_time = np.argmin(np.abs(times - static_time))
    else:
        index_time = 0

    # CREATE SUBPLOT
    fig, axes = plt.subplots(nrows_ncols[0], nrows_ncols[1])
    if type(axes) == np.ndarray:
        axes = axes.ravel()
    else:
        axes = [axes]
    fig.set_size_inches([20,8])

    # PLOT DATA
    lines = []
    for i, (quant, ax) in enumerate(zip(data_dictionary.keys(), axes)):
        lines.append( ax.plot(coord, data[i][index_time], "o", label=quant)[0] )
        ax.set_ylim([np.min(data[i]), np.max(data[i])])
        ax.set_title(quant)
    
    # SAVE STATIC PLOT
    if save_fig != None:
        fig.savefig("%s.pdf" % save_fig, bbox_inches="tight")
    if static_time != None:
        plt.show()

    # CREATE/SAVE ANIMATION
    ani = FuncAnimation(fig, update_1D, frames=len(times), fargs=(data, lines), interval=interval, blit=False, repeat=True)
    if save_anim is not None:
        if save_anim.endswith((".avi", ".mp4")):
            ani.save(save_anim, dpi=dpi)
        else:
            ani.save("%s.gif" % save_anim, dpi=dpi)

    plt.show()

def update_2D(i: int, X: np.ndarray, Y: np.ndarray, data: List, levelset: np.ndarray,
        axes: np.ndarray, quadmesh_list: List, pciset_list:List) -> List:
    """Update function for FuncAnimation for the create_contourplot() function.

    :param i: Time snapshot index
    :type i: int
    :param X: X meshgrid
    :type X: np.ndarray
    :param Y: Y Meshgrid
    :type Y: np.ndarray
    :param data: Data buffers
    :type data: List
    :param levelset: Levelet buffer
    :type levelset: np.ndarray
    :param axes: Axes
    :type axes: np.ndarray
    :param quadmesh_list: Quadmesh objects returned by pcolormesh()
    :type quadmesh_list: List
    :param pciset_list: Pciset objects return by contour()
    :type pciset_list: List
    :return: List of collections
    :rtype: List
    """

    list_of_collections = []

    for z, ax, quadmesh, pciset in zip(data, axes, quadmesh_list, pciset_list):

        if type(levelset) != type(None) and type(pciset[0]) != type(None):
            for tp in pciset[0].collections:
                tp.remove()
            pciset[0] = ax.contour(X, Y, np.squeeze(levelset[i,:,:]), levels=[0.0], colors="black", linewidths=4)
            list_of_collections += pciset[0].collections

        quadmesh.set_array(z[i,:,:].flatten())
        list_of_collections.append(quadmesh)

    return list_of_collections

def update_1D(i: int, data: List, lines: List) -> List:
    """Update function for FuncAnimation for the create_lineplot() function.

    :param i: Time snapshot index
    :type i: int
    :param data: Data buffers
    :type data: List
    :param lines: Line objects
    :type lines: List
    :return: List of collections
    :rtype: List
    """
    list_of_collections = []
    for d, line in zip(data, lines):
        line.set_ydata(d[i,:])
        list_of_collections.append(line)
    return list_of_collections

def create_figure_2D(data: List, levelset: np.ndarray, nrows_ncols: Tuple, titles: List,
        X: np.ndarray, Y: np.ndarray, minmax: List, index: int = 0) -> Tuple:

    # CREATE SUBFIGURE
    fig, axes = plt.subplots(nrows=nrows_ncols[0], ncols=nrows_ncols[1], squeeze=False)
    if type(axes) != np.ndarray:
        axes = [axes]
    else:
        axes = axes.ravel()
    fig.set_size_inches(20,10,True)

    # KEYWORD ARGUMENTS FOR PCOLORMESH - SCHLIEREN ARE ALWAYS PLOTTED USING A LOG SCALE
    kwargs = []
    for i, quantity in enumerate(titles):
        if quantity == "schlieren":
            data[i] = np.clip(data[i], 1e-10, 1e10)
            kwargs.append(dict(cmap="binary", norm=matplotlib.colors.LogNorm(vmin=minmax[i][0], vmax=minmax[i][1])))
        else:
            kwargs.append(dict(cmap="seismic", vmin=minmax[i][0], vmax=minmax[i][1]))

    quadmesh_list = []
    pciset_list = []

    for ax, z, kw, title in zip(axes, data, kwargs, titles):

        # PCOLORMESH
        quadmesh = ax.pcolormesh(X, Y, z[index,:,:], **kw, shading="auto")
        quadmesh_list.append(quadmesh)

        # ZERO LEVELSET
        if type(levelset) != type(None):
            pciset = [ax.contour(X, Y, np.squeeze(levelset[index,:,:]), levels=[0.0], colors="black", linewidths=4)]
            pciset_list.append(pciset)
        else:
            pciset_list.append([None])

        # AXIS TICKS
        ax.set_title(title.upper(), fontsize=10, pad=10)
        ax.tick_params(labelsize=10, left=False, bottom=False, labelleft=False, labelbottom=False)
        ax.set_aspect("equal")

    return fig, axes, quadmesh_list, pciset_list


def generate_png(X: np.ndarray, Y: np.ndarray, data_dictionary: Dict[str, np.ndarray], save_path: str,
        levelset: np.ndarray = None, nrows_ncols: Tuple = None, dpi=200) -> None:
    """Generates a subplot of pcolormesh and saves it as png. If the levelset argument is provided,
    the interface is plotted as an isoline. 

    :param X: X meshgrid
    :type X: np.ndarray
    :param Y: Y meshgrid
    :type Y: np.ndarray
    :param data_dictionary: Data buffer
    :type data_dictionary: Dict[str, np.ndarray]
    :param save_path: Save path
    :type save_path: str
    :param levelset: Levelset buffer, defaults to None
    :type levelset: np.ndarray, optional
    :param nrows_ncols: Shape of the subplot, defaults to None
    :type nrows_ncols: Tuple, optional
    :param dpi: Resolution in dpi, defaults to 200
    :type dpi: int, optional
    """

    quantities = list(data_dictionary.keys())

    no_timesnapshots = len(data_dictionary[quantities[0]])
    
    kwargs = []
    for quant in quantities:
        if quant == "schlieren":
            kwargs.append( {"cmap": "binary", "norm": matplotlib.colors.LogNorm(vmin=np.min(data_dictionary[quant]), vmax=np.max(data_dictionary[quant]))} )
        else:
            kwargs.append( {"cmap": "seismic", "vmin": np.min(data_dictionary[quant]), "vmax": np.max(data_dictionary[quant])} )

    fontsize = 20

    nrows_ncols = (1,len(quantities)) if nrows_ncols == None else nrows_ncols
    assert np.prod(nrows_ncols) == len(quantities), "Amount of requested axes and available quantities do not match."

    for i in range(no_timesnapshots):

        filename = "image_%04d" % i

        fig, axes = plt.subplots(nrows_ncols[0], nrows_ncols[1])
        fig.set_size_inches(nrows_ncols[0]*10,nrows_ncols[1]*10,True)
        fig.tight_layout()
        axes = axes.flatten()

        for ax, quantity, kw in zip(axes, data_dictionary, kwargs):

            pcolormesh = ax.pcolormesh(X, Y, data_dictionary[quantity][i,:,:,0], **kw, shading="auto")
            if type(levelset) != type(None):
                ax.contour(X, Y, levelset[i,:,:,0], levels=[0.0], colors="black", linewidths=3)

            # AXIS TICKS
            ax.set_title(quantity.upper(), fontsize=fontsize, pad=10)
            ax.tick_params(labelsize=fontsize, left=False, bottom=False, labelleft=False, labelbottom=False)
            ax.set_aspect("equal")

        print("Saving %s" % filename)
        fig.savefig(os.path.join(save_path, filename), dpi=dpi, bbox_inches="tight")
        plt.clf()
        plt.cla()
        plt.close(fig)
        del fig, axes

def create_xdmf_from_h5(path: str) -> None:
    """Creates .xdmf files from .h5 files for Paraview visualization.

    :param path: Path to .h5 files.
    :type path: str
    """

    # SEARCH DIRECTORY FOR FILES
    files = []
    times = []
    for file in os.listdir(path):
        if file.endswith("h5"):
            if "nan" in file:
                continue 
            files.append(file)
            times.append(float(os.path.splitext(os.path.basename(file))[0][5:]))
    indices     = np.argsort(np.array(times))
    times       = np.array(times)[indices]
    files       = np.array(files)[indices]
    no_times    = len(times)

    # DOMAIN INFORMATION
    with h5py.File(os.path.join(path, files[0]), "r") as h5file:
        x = h5file["mesh/gridX"][:]
        y = h5file["mesh/gridY"][:]
        z = h5file["mesh/gridZ"][:]
        dx = h5file["mesh/cellsizeX"][()]
        dy = h5file["mesh/cellsizeY"][()]
        dz = h5file["mesh/cellsizeZ"][()]
        cell_centers    = [x, y, z]
        cell_sizes      = [dx, dy, dz]
        number_of_cells = [len(x), len(y), len(z)]
        cell_faces = []
        for xi, dxi in zip(cell_centers, cell_sizes):
            cell_faces.append( np.linspace(xi[0] - dxi/2, xi[-1] + dxi/2, len(xi)+1) )

    for i, (file, time) in enumerate(zip(files, times)):

        filename = "data_%.6f" % time

        h5file_name   = filename + ".h5"
        xdmffile_path = filename + ".xdmf"
        print("Writing file %s" % xdmffile_path)
        xdmffile_path = os.path.join(path, xdmffile_path)

        # XDMF QUANTITIES
        xdmf_quants = []

        with h5py.File(os.path.join(path, file), "r") as h5file:

            # CONSERVATIVES AND PRIMITIVES 
            for key in ["cons", "primes", "real_fluid", "miscellaneous", "levelset"]: 
                if key in h5file.keys():
                    for quantity in h5file[key]:
                        xdmf_quants.append(get_xdmf(key, quantity, h5file_name, *number_of_cells))

        # XDMF START
        xdmf_str_start ='''<?xml version="1.0" ?>
        <!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
        <Xdmf Version="3.0">
        <Domain>
            <Grid Name="Cube">
                <Geometry Type="VXVYVZ">
                    <DataItem Format="HDF" NumberType="Float" Precision="%i" Dimensions="%i">%s:mesh/gridFX</DataItem>
                    <DataItem Format="HDF" NumberType="Float" Precision="%i" Dimensions="%i">%s:mesh/gridFY</DataItem>
                    <DataItem Format="HDF" NumberType="Float" Precision="%i" Dimensions="%i">%s:mesh/gridFZ</DataItem>
                </Geometry>
                <Topology Dimensions="%i %i %i" Type="3DRectMesh"/>''' %( # 1 512 128
                    8, len(cell_faces[0]), h5file_name, 
                    8, len(cell_faces[1]), h5file_name, 
                    8, len(cell_faces[2]), h5file_name,
                    len(cell_faces[0]), len(cell_faces[1]), len(cell_faces[2]))

        # XDMF END
        xdmf_str_end = '''</Grid>
        </Domain>
        </Xdmf>'''

        # JOIN FINAL XDMF STR AND WRITE TO FILE
        xdmf_str = "\n".join([xdmf_str_start] + xdmf_quants + [xdmf_str_end])
        with open(xdmffile_path, "w") as xdmf_file:
            xdmf_file.write(xdmf_str)
    

def get_xdmf(group: str, quant: str, h5file_name: str, Nx: int, Ny: int, Nz: int) -> str:
    if quant in ["velocity", "momentum", "vorticity", "normal"]:
        xdmf ='''<Attribute Name="%s" AttributeType="Vector" Center="Cell">
        <DataItem Format="HDF" NumberType="Float" Precision="%i" Dimensions="%i %i %i %i">%s:%s/%s</DataItem>
        </Attribute>''' %(quant, 8 , Nz, Ny, Nx, 3, h5file_name, group, quant)
    else:
        xdmf ='''<Attribute Name="%s" AttributeType="Scalar" Center="Cell">
            <DataItem Format="HDF" NumberType="Float" Precision="%i" Dimensions="%i %i %i">%s:%s/%s</DataItem>
        </Attribute>''' %(quant, 8 , Nz, Ny, Nx, h5file_name, group, quant)
    return xdmf