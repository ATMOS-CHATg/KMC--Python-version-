import numpy as np
from parameters import *
import spatial
import utils
import scipy.sparse as sp
import os

# Contains functions which load and build the system to perform the KMC on

#_____________________________________________________________________________________
def load_data():
    """Weeeell... It load the data from a VASP POSCAR file ?

    Returns:
        list: lattice_parameters along each direction
        numpy ndarray (float): nitrogen locations
        numpy ndarray (float): oxygen locations
        int: amount of oxygens
        int: amount of nitrogens
        int: amount of sites in structure
    """

    with open(f'static/{structure}', 'r') as f:
        lines = f.readlines()
        x, y, z = np.array([]), np.array([]), np.array([])
        scaling_factor = float(lines[1])
        lparam_x = float(list(filter(None, lines[2].split(' ')))[0])
        lparam_y = float(list(filter(None, lines[3].split(' ')))[1])
        lparam_z = float(list(filter(None, lines[4].split(' ')))[2])
        atom_amounts = np.array(list(filter(None, lines[6].split(' '))), dtype=int)
        atom_names = lines[5]

        for i in range(8, len(lines)):
            r = list(filter(None, lines[i].split(' ')))
            x = np.append(x, float(r[0]))
            y = np.append(y, float(r[1]))
            z = np.append(z, float(r[2].strip()))

        r = np.array([x, y, z]).transpose()
        rO = r[0:atom_amounts[0], :]
        rH = r[atom_amounts[0] + 1:atom_amounts[0] + 1 + atom_amounts[1], :]
        rN = r[atom_amounts[0] + atom_amounts[1]:, :]
        guest_amount = int(len(rN)/2)

    return [lparam_x, lparam_y, lparam_z], rN, rO, atom_amounts[0], atom_amounts[2], guest_amount



#_____________________________________________________________________________________
def init_event(catalog, neighbors_map, states, transition_type, system_size=system_size):
    """Function computing the amount of possible events corresponding to a given transition
    and inventoring the candidates

    Args:
        catalog (numpy ndarray): array containing informations for each site
        neighbors_map (numpy ndarray): array containing informations for each site's neighbors
        states (list): list containing the state of the departure and arrival sites (how they are filled)
        transition_type (int): integer corresponding to the transition (L6L......)

    Returns:
        list, candidates: candidate ids for a given transition type
        int, amount_of_candidates: amount of candidates for a given transition type
    """

    departure, arrival = states

    # Amplify the length of candidates array depending on mode
    if mode == 'DO':
        amplifier = 2
    else:
        amplifier = 1

    # Scaling it to the maximum amount of a given pathway in the unit cell
    if structure == 'sI':
        max_transition_amount = 48

    if structure == 'sII':
        max_transition_amount = 96

    candidates = -1 * np.ones(int(amplifier*max_transition_amount*system_size**3), dtype=int)
    amount_of_candidates = 0
    window = 0

    # Checking for transitions for occupied sites
    for i in catalog[catalog[:,5] == departure][:,0]:
        # Checking where are given transitions
        transitions = np.where(neighbors_map[i][:,-1] == transition_type)[0]

        # Checking where are available neighboring sites
        available = np.where(catalog[neighbors_map[i][transitions,0]][:,5] == arrival)[0]

        # Adding corresponding amount of transitions
        if departure == 0:
            matching_candidates_amount = 2*len(available)
            amount_of_candidates += matching_candidates_amount
        else:
            matching_candidates_amount = len(available)
            amount_of_candidates += matching_candidates_amount

        # Adding transition times the considered ID
        if matching_candidates_amount>0:
            candidates[window : matching_candidates_amount + window] = matching_candidates_amount*[i]
            window += matching_candidates_amount

    return candidates, amount_of_candidates



#_____________________________________________________________________________________
def init_events(catalog, neighbors_map, rate_constants, forbidden_transitions):
    """Function initializing all possibles transitions and their corresponding candidates

    Args:
        catalog (numpy ndarray (int)): contains site states
        neighbors_map (numpy ndarray (object)): contains neighbors states for each site
        rate_constants (numpy ndarray (float)): contains rate constants computed with the Arrhenius law
        forbidden_transitions (list): contains forbidden transitions

    Returns:
        numpy ndarray, amounts: amount of candidates for each transition
        numpy ndarray, candidates: candidates IDs for each transition
    """

    candidates = np.zeros(len(rate_constants[rate_constants != 0]),dtype='object')
    amounts = np.zeros(len(rate_constants[rate_constants != 0]),dtype=int)

    pathways = list(range(4))
    status = [[1,2], [1,1], [0,2], [0,1]]

    pos = 0
    for current_pathway in range(len(pathways)):

        for current_status in range(len(status)):

            if not tuple([*status[current_status], pathways[current_pathway]]) in forbidden_transitions:
                candidates[pos], amounts[pos] = init_event(catalog, neighbors_map, status[current_status], pathways[current_pathway])
                pos+=1

    return amounts, candidates



#_____________________________________________________________________________________
def init_guest_locations(catalog):
    """Function initializing the object containing molecules locations

    Args:
        catalog (numpy ndarray (int)): contains site states

    Returns:
        scipy lil matrix: sparse matrix containing molecules locations
        numpy ndarray (int): initial location
    """

    if mode == 'DO':
        current_guest_locations = np.zeros(int(len(catalog[catalog[:,7] != -1])+len(catalog[catalog[:,8] != -1])), dtype=int)
    else:
        current_guest_locations = np.zeros(len(catalog[catalog[:,7] != -1]), dtype=int)

    for i in range(len(catalog[catalog[:,7] != -1])):
        current_guest_locations[catalog[catalog[:,7] != -1][i,7]] = catalog[catalog[:,7] != -1][i,0]
    
    if mode == 'DO':
        for j in range(len(catalog[catalog[:,8] != -1])):
            current_guest_locations[catalog[catalog[:,8] != -1][j,8]] = catalog[catalog[:,8] != -1][j,0]

    guest_locations = sp.lil_matrix((iterations, len(current_guest_locations)), dtype=int)
    guest_locations[0,:] = current_guest_locations

    return guest_locations, current_guest_locations



#_____________________________________________________________________________________
def init_occupancies(catalog):
    """Function initializing the occupancies of cages object

    Args:
        catalog (numpy ndarray (int)): contains site states

    Returns:
        scipy lil matrix: sparse matrix containing cage occupancies
        numpy ndarray (int): initial occupancies
    """

    small_cages = catalog[catalog[:,4]==1]
    large_cages = catalog[catalog[:,4]==0]

    # ss small simple, ls large simple, ld large double
    ss_occupancy = len(small_cages[small_cages[:,5]==1])
    ls_occupancy = len(large_cages[large_cages[:,5]==1])
    ld_occupancy = len(large_cages[large_cages[:,5]==0])

    occupancies = sp.lil_matrix((3,iterations), dtype=int)
    occupancies[:, 0] = np.array([ss_occupancy, ls_occupancy, ld_occupancy])

    return occupancies, np.array([ss_occupancy, ls_occupancy, ld_occupancy])



#_____________________________________________________________________________________
def build_neighbors_map(catalog, guest_amount, system_size=system_size):
    """Function building complete neighbors map object.

    Args:
        catalog (numpy ndarray): array containing every site information
        guest_amount (int): amount of cages in the unit cell

    Returns:
        numpy ndarray: complete neighbors map
    """

    neighbors_map = spatial.get_neighbors(guest_amount)
    cages_type = spatial.get_neighbors_cage_type(neighbors_map, catalog, guest_amount)
    faces_type = spatial.get_neighbors_face_type(neighbors_map, catalog, guest_amount)
    ids = utils.associate_ids_to_map(neighbors_map, catalog, guest_amount, system_size)

    final_map = np.zeros(len(ids), dtype='object')

    for i in range(len(ids)):
        final_map[i] = np.column_stack((ids[i], cages_type[i%guest_amount]))
        final_map[i] = np.column_stack((final_map[i], faces_type[i%guest_amount])).astype(int)

    return final_map



#_____________________________________________________________________________________
def build_cages_catalog(int_map, cages, guest_amount, system_size=system_size):
    """Function building the object containing all informations on cages (main object)

    Args:
        int_map (numpy ndarray): contains cages centers integer mapping
        cages (list): contains the cages type [0->large, 1->small]
        guest_amount (int): size of the unit cell in the integer coordinate system.

    Returns:
        numpy ndarray: object containing sites status
    """

    sites = np.zeros((system_size**3*guest_amount, 9), dtype=int)
    size = 0

    for i in range(system_size**3*guest_amount):

        # Defines the current site
        current_site = int_map[i, :]
        current_site = np.append(i, current_site)
        current_site = np.append(current_site, [cages[size], 2, 0, -1, -1])
        size += 1
        if size == guest_amount:
            size = 0

        sites[i, :] = current_site

    return sites



#_____________________________________________________________________________________
def build_system(cage_centers, lattice_parameter, system_size=system_size):
    """Function building the system, ie. cages, sites and locations

    Args:
        cage_centers (numpy ndarray): contains the cage centers.
        system_size (int): size of the system
        lattice_param (float): lattice parameter extracted from the POSCAR file
        structure (str): 'sI' or 'sII'

    Returns:
        numpy ndarray: centers: updated centers
        numpy ndarray: int_map: integer coordinates
        numpy ndarray: cages: contains the type of cage, 0 (1) small (large)
    """

    script_dir = os.path.dirname(__file__)

    if structure == "sI":
        cages = [0] * 6 + [1] * 2
        int_map = np.loadtxt(script_dir+'/static/sites_sI.dat', dtype=int)
        
    elif structure == 'sII':
        cages = [0] * 8 + [1] * 16
        int_map = np.loadtxt(script_dir+'/static/sites_sII.dat', dtype=int)

    int_map, cages, cage_centers = spatial.add_unit_cell(cage_centers, int_map, cages, 0, lattice_parameter, system_size)
    int_map, cages, cage_centers = spatial.add_unit_cell(cage_centers, int_map, cages, 1, lattice_parameter, system_size)
    int_map, cages, cage_centers = spatial.add_unit_cell(cage_centers, int_map, cages, 2, lattice_parameter, system_size)

    del script_dir

    return int_map, cages, cage_centers



#_____________________________________________________________________________________
def fill_system(catalog, total_occupancy):
    """Function to randomly fill the structure with molecules

    Args:
        catalog (numpy ndarray (int)): contains site states

    Returns:
        numpy ndarray: updated catalog

    """

    molecule = 0

    if mode == 'DO':
        large = np.where(catalog[:,4]==0)[0]
        small = np.where(catalog[:,4]==1)[0]

        len_large_DO = len(large)
        large = np.concatenate((large,large))
        
        cages = np.concatenate((large,small))
        np.random.shuffle(cages)
        cages = cages[0:int(total_occupancy*(len(catalog) + len_large_DO))]

        for i in range(len(cages)):

            if catalog[cages[i],7] != -1 and not catalog[cages[i],4] == 1:
                catalog[cages[i],8] = molecule
                catalog[cages[i],5] = 0
                molecule += 1

            else:
                catalog[cages[i],7] = molecule
                catalog[cages[i],5] = 1
                molecule += 1

    elif mode == 'SO':
        cages = np.where(catalog[:,4]>=0)[0]
        np.random.shuffle(cages)
        cages = cages[0:int(total_occupancy*len(catalog))]

        for i in range(len(cages)):
            catalog[cages[i], 5] = 1
            catalog[cages[i], 7] = molecule
            molecule += 1

    return catalog