from parameters import *
import numpy as np
import os
import shutil
from datetime import datetime
import scipy.sparse as sp
import re
from collections import defaultdict
import spatial

# Miscellanous functions

#_____________________________________________________________________________________
def manage_folder(T=T, total_occupancy=total_occupancy, obs=''):
    """Function handling results folder creation/backup

    Returns:
        str: path to results folder
    """

    if vacancies:
        subdir = '_WV'
    else:
        subdir = ''

    if obs == "":
        folder_path = f'output/KMC_simulations/{structure}_{mode}{subdir}_{T}_{total_occupancy}_{functional}'
    elif obs != "":
        folder_path = f'output/diffusion/{obs}'

    # Check if the folder exists
    if os.path.exists(folder_path):

        # Make a backup by copying the folder and appending a timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_folder_path = f"{folder_path}_backup_{timestamp}"
        shutil.copytree(folder_path, backup_folder_path)
        print(50*'_')
        print(f"Backup created at: {backup_folder_path}")

    else:
        # Create the folder if it doesn't exist
        os.makedirs(folder_path)
        print(50*'_')
        print(f"Folder created at: {folder_path}")

    return folder_path



#_____________________________________________________________________________________
def get_params():
    """Function gathering which parameter to scale the mp simulation on

    Returns:
        list, densities
        list, temperatures
    """

    if parameter == 'Density':
        total_occupancies = np.array(range(min_value, int(max_value+step), step))/100
        temperatures = [T]*len(total_occupancies)
    if parameter == 'Temperature':
        temperatures = np.array(range(min_value, int(max_value+step), step))
        total_occupancies = [total_occupancy]*len(temperatures)

    if len(np.unique(total_occupancies)) > len(np.unique(temperatures)):
        scale = len(total_occupancies)
    else:
        scale = len(temperatures)

    return total_occupancies, temperatures, scale



#_____________________________________________________________________________________
def associate_ids_to_map(neighbors_map, catalog, guest_amount, system_size):
    """Function associating IDs of neighbors in catalog to their neighbor map

    Args:
        neighbors_map (numpy ndarray): Array containing every neighbor for every site
        catalog (numpy ndarray): Array containing every site information
        guest_amount (int): amount of sites in the unit cell
    Returns:
        numpy ndarray: Updated neighbor map
    """

    # Defining a dictionary containing as key the integer coordinates of each sites,
    # and a values its ID.

    size = np.max(catalog[:guest_amount,1:4], axis=0)+1

    inventory = {}

    for i in range(len(catalog)):
        inventory[tuple(catalog[i,1:4])] = i

    # Tiling unit map to system size

    whole_map = np.tile(neighbors_map , int(len(catalog)/len(neighbors_map)))
    print('Here', int(len(catalog)/len(neighbors_map)))
    new_map = np.zeros(len(whole_map), dtype=object)

    # Setting ID wrt neighboring site
    for i in range(len(whole_map)):
        temp_ids = np.zeros(len(whole_map[i]), dtype=int)
        pbc = (whole_map[i][:,0:3]+catalog[i,1:4])%(size*system_size)


        for j in range(len(whole_map[i])):
            temp_ids[j] = inventory[tuple(pbc[j])]

        new_map[i] = temp_ids

        del temp_ids, pbc

    del whole_map

    return new_map



#_____________________________________________________________________________________
def save_results(data, parameter_dictionary, T=T, total_occupancy=total_occupancy):
    """Function generating the KMC simulation output files

    Args:
        data (object): contains observables
        T (int, optional): temperature. Defaults to T.
        total_occupancy (float, optional): total occupancy... Defaults to total_occupancy.
    """

    folder_path = manage_folder(T, total_occupancy)
    sp.save_npz(f'{folder_path}/trajectories', data[3])
    np.savetxt(f'{folder_path}/times.dat', data[2])
    np.savetxt(f'{folder_path}/probes.dat',data[1], fmt='%i')
    sp.save_npz(f'{folder_path}/occupancies', data[0])
    np.savetxt(f'{folder_path}/events.dat', data[4], fmt='%i')

    with open(f'{folder_path}/parameter_history.dat', "w") as file:
        for key, value in parameter_dictionary.items():
            file.write(f"{key}:{value}\n")


#_____________________________________________________________________________________
def progress_bar():
    """A simple progress bar

    Returns:
        object: updated progress bar
    """

    if preset == 'local':
        import progressbar
        widgets = [' [',
                progressbar.Timer(),
                '] ',
                progressbar.Bar('*'),' (',
                progressbar.ETA(), ') ',
                ]
        bar = progressbar.ProgressBar(widgets=widgets, maxval=iterations-1).start()

    else:
        class bar:
            def update(self):
                pass

    return bar



#_____________________________________________________________________________________
def display_results_after_iteration(candidates, index_to_transitions, pathway_dicts, gauge, rate_constants):
    """Only used in debug mode: allows for the user to see what is going on in the KMC

    Args:
        candidates (numpy ndarray (int)): contains cages from which a molecule can perform a specific transition
        index_to_transitions (dict): contains id as keys and transitions as values
        pathway_dicts (dict): contains pathways names
        gauge (numpy ndarray (float)): contains the cumulative sum of rate constants ponderated by the amount of candidates
        rate_constants (numpy ndarray (float)): contains rate constants computed with the Arrhenius law
    """

    print(50*'_')
    print(f'Candidates are now:')
    print(f'Transition'.ljust(10), 'Pathway'.ljust(10), 'Amount'.ljust(10), 'Rate Constant / Ponderated'.ljust(10))
    for i in range(len(candidates)):

        length = len(candidates[i][candidates[i] != -1])
        if length > 0:
            print(f'{index_to_transitions[i]}'.ljust(10), 
                f'({pathway_dicts[i]})'.ljust(10), 
                f'{length}'.ljust(10), 
                f'{rate_constants[i]} / {rate_constants[i]*length}'.ljust(10))

    print(50*'_')
    print(f'Gauge is now: {gauge}')
    print(50*'_')



#_____________________________________________________________________________________
def read_atomic_masses():
    """Simply loading atomic masses database (gravity centers computations purpose)

    Returns:
        dict: atomic masses for each element
    """

    with open('static/atomic_masses.dat', 'r') as file:
        atomic_masses = eval(file.read())

    return atomic_masses



#_____________________________________________________________________________________
def detect_guest_molecule(chemical_formulae):
    """Function using regex to detect guest molecules in structure

    Args:
        chemical_formulae (str): guest molecule (has to be put without blanks (ie. no C O2, CO2 instead))

    Returns:
        dict: amount of atoms constituing the molecule
    """

    element_counts = defaultdict(int)

    matches = re.findall(r'([A-Z][a-z]*)(\d*)', chemical_formulae)

    for (element, count) in matches:

        count = int(count) if count else 1
        element_counts[element] += count

    return dict(element_counts)



#_____________________________________________________________________________________
def extract_molecule_coordinates(filename, formula, num_molecules=8, atom_type='O'):
    """Function extracting coordinates of each molecule compound in the given structure and grouping them by locations

    Args:
        filename (str): path to file
        formula (str): chemical formulae
        num_molecules (int, optional): amount of guests in structure (sI:8, sII:24). Defaults to 8.
        atom_type (str, optional): remaining oxygens. Defaults to 'O'.

    Raises:
        ValueError: if not enough compounds are detected

    Returns:
        numpy ndarray: contains each compound coordinates (cartesian) and atomic mass
        numpy ndarray: contains remaining oxygens coordinates (cartesian)
        numpy ndarray: lattice parameter
    """

    atom_requirements = detect_guest_molecule(formula)
    atomic_masses = read_atomic_masses()

    with open(filename, 'r') as file:
        lines = file.readlines()

    lparam_x = float(list(filter(None, lines[2].split(' ')))[0])
    lparam_y = float(list(filter(None, lines[3].split(' ')))[1])
    lparam_z = float(list(filter(None, lines[4].split(' ')))[2])
    lattice_parameter = np.array([lparam_x, lparam_y, lparam_z])

    atom_types = lines[5].split()
    atom_counts = list(map(int, lines[6].split()))
    atom_coords = {}

    start_index = 8
    coordinates = [line.strip() for line in lines[start_index:]]

    index = 0
    for atom, count in zip(atom_types, atom_counts):
        atom_coords[atom] = coordinates[index:index + count]
        index += count


    molecules = []
    for molecule_index in range(num_molecules):
        molecule = []
        for atom, required_count in atom_requirements.items():
            if atom not in atom_coords or len(atom_coords[atom]) < required_count * num_molecules:
                raise ValueError(f"Not enough {atom} atoms to form the specified number of molecules.")


            start = -(molecule_index + 1) * required_count
            end = None if start == -required_count else start + required_count

            atom_coordinates = [list(map(float, coord.split())) for coord in atom_coords[atom][start:end]]

            for coord in atom_coordinates:
                molecule.append(coord + [atomic_masses[atom]])  # Add atomic mass to the coordinates
        molecules.append(molecule)


    molecule_array = np.array(molecules, dtype=float)

    remaining_oxygen_coords = []
    if atom_type in atom_coords:
        total_oxygen_needed = atom_requirements.get(atom_type, 0) * num_molecules
        remaining_oxygen_coords = atom_coords[atom_type][:len(atom_coords[atom_type])-total_oxygen_needed]

    oxygen_coords_array = np.array([list(map(float, coord.split())) for coord in remaining_oxygen_coords])

    if lines[7].strip() == 'Direct':
        print(molecule_array[0,:,:])
        molecule_array[:,:,:3] = molecule_array[:,:,:3] * lattice_parameter
        print(molecule_array[0,:,:])
        oxygen_coords_array = oxygen_coords_array * lattice_parameter

    if structure == 'sI':
        N_H2O = 46
    elif structure == 'sII':
        N_H2O = 136

    return molecule_array, oxygen_coords_array, lattice_parameter, N_H2O



#_____________________________________________________________________________________
def compute_guest_gravity_center(molecules, lattice_parameter):
    """Computes the guest molecules gravity centers

    Args:
        molecules (numpy ndarray): contains each compound of each molecule locations
        lattice_parameter (numpy ndarray): lattice parameter ?!

    Returns:
        numpy ndarray: contains all gravity centers (cartesian)
    """

    gravity_centers = np.zeros((molecules.shape[0], 3))

    for i in range(molecules.shape[0]):

        for j in range(1, molecules.shape[1]):

            for k in range(molecules.shape[2]-1):

                if abs(molecules[i,0,k] - molecules[i,j,k]) > 0.5*lattice_parameter[k]:

                    if molecules[i,0,k] > molecules[i,j,k]:
                        molecules[i,j,k] += lattice_parameter[k]
                    elif molecules[i,0,k] < molecules[i,j,k]:
                        molecules[i,j,k] -= lattice_parameter[k]
        
        for k in range(molecules.shape[1]):
            gravity_centers[i,:] += molecules[i,k,:3]*molecules[i,k,-1]

        gravity_centers[i,:] = (gravity_centers[i,:] / (np.sum(molecules[i,:,-1])))%lattice_parameter

    gravity_centers = gravity_centers[::-1, :]

    return gravity_centers



#_____________________________________________________________________________________
def map_centers_to_system_size(gravity_centers, num_molecules, lattice_parameter, system_size):
    """Scales gravity centers to system size by cloning them with respect to the lattice parameter

    Args:
        gravity_centers (numpy ndarray): guess what it is
        num_molecules (int): amount of guest molecules in the structure
        lattice_parameter (numpy ndarray): guess what it is (more difficult)
        system_size (int): size of the system along a coordinate (system_size**3 is the whole system)

    Returns:
        numpy ndarray: gravity centers
    """

    # toto and toto2 are a super duper trick, they are useless here
    toto = np.zeros((num_molecules,3))
    toto2 = np.zeros((num_molecules,3))

    toto, toto2, gravity_centers = spatial.add_unit_cell(gravity_centers, toto, toto2, 0, lattice_parameter, system_size)
    toto, toto2, gravity_centers = spatial.add_unit_cell(gravity_centers, toto, toto2, 1, lattice_parameter, system_size)
    toto, toto2, gravity_centers = spatial.add_unit_cell(gravity_centers, toto, toto2, 2, lattice_parameter, system_size)

    return gravity_centers



#_____________________________________________________________________________________
def get_cages_amount(system_size):
    """Function computing the amount of cages for a given structure and system size

    Returns:
        int: amount of small cages
        int: amount of large cages
    """

    if structure == 'sI':
        small = 2/8 * 8 * system_size**3
        large = 6/8 * 8 * system_size**3

    elif structure == 'sII':
        small = 16/24 * 24 * system_size**3
        large = 8/24 * 24 * system_size**3

    return small, large



#_____________________________________________________________________________________
def unwrap_occupancies(occupancies, iterations):
    """Function decoding occupancies from KMC simulations

    Args:
        occupancies (sp lil matrix): sparse matrix of occupancies

    Returns:
        numpy ndarray: cage occupancies along simulation
    """

    unwrapped_occupancies = np.zeros((occupancies.shape[0], iterations))

    for i in range(occupancies.shape[0]):
        unwrapped_occupancies[i, :] = np.cumsum(occupancies.tocsr()[i,:].toarray()).astype(int)

    return unwrapped_occupancies



#_____________________________________________________________________________________
def unwrap_events(events, iterations):
    """Function decoding occupancies from KMC simulations

    Args:
        occupancies (sp lil matrix): sparse matrix of occupancies

    Returns:
        numpy ndarray: cage occupancies along simulation
    """

    unwrapped_events = np.zeros((events.shape[1], iterations))

    for i in range(events.shape[1]):
        unwrapped_events[i, :] = np.cumsum(events[:,i]).astype(int)

    return unwrapped_events



#_____________________________________________________________________________________
def copy_parameters():
    parameter_dictionary = {}

    parameter_dictionary['Temperature'] = T
    parameter_dictionary['Total_occupancy'] = total_occupancy
    parameter_dictionary['System_size'] = system_size
    parameter_dictionary['Iterations'] = iterations
    parameter_dictionary['Probe_amount'] = probe_amount
    parameter_dictionary['Probe_size'] = probe_size
    parameter_dictionary['Structure'] = structure
    parameter_dictionary['Guest'] = guest_molecule
    parameter_dictionary['Mode'] = mode
    parameter_dictionary['Vacancies'] = vacancies
    parameter_dictionary['Functional'] = functional
    parameter_dictionary['Preset'] = preset

    return parameter_dictionary



#_____________________________________________________________________________________
def read_parameters(folder):
    data = {}
    with open(f"output/KMC_simulations/{folder}/parameter_history.dat", "r") as file:
        for line in file:
            key, value = line.strip().split(":")
            data[key] = value
    return data