import numpy as np
import os
from parameters import *
import random
from scipy.interpolate import interp1d

# Contain functions which handle system topography, move molecules and unwrap trajectories

#_____________________________________________________________________________________
def gravity_center(r, m1=1, m2=1):
    """Computes the diatomic molecules gravity centers
    (It is only used to get the centers of cages in order to find
    the N nearest neighbors for setting up probes)

    Args:
        r (numpy ndarray): array containing the diatomic molecules positions
        m1 (int, optional): mass to ponderate if two different atoms. Defaults to 1.
        m2 (int, optional): mass to ponderate if two different atoms. Defaults to 1.

    Returns:
        numpy ndarray: array containing the diatomic molecules gravity centers
    """

    centers = np.zeros((int(len(r)/2), 3))

    for i in range(len(r[0])):

        # coordinates of particles with mass m1
        r_plus = r[1::2, i]  

        # coordinates of particles with mass m2
        r_minus = r[0::2, i] 

        # Compute gravity centers
        centers[:, i] = (r_plus * m1 + r_minus * m2) / (m1 + m2)

    return centers



#_____________________________________________________________________________________
def add_unit_cell(cage_centers, int_map, cages, axis, lattice_parameter, N=1):
    """Function adding N unit cells along a given axis
    !!! WARNING: when used multiple times (eg. N=2 along x), the next use
    (eg. N=2 along y) will add x*y cells !!!

    Args:
        centers (numpy ndarray): array containing the coordinates of gravity centers
        int_map (numpy ndarray): array containing the new cage classification
        cages (list): list containing cages type in unit cell
        axis (int): axis along which to add the pattern, 0:x, 1:y, 2:z
        lattice_param (float): lattice parameter
        N (int, optional): amount of added pattern. Defaults to 1.

    Returns:
        numpy ndarray: updated integer map
        list: updated cage types
        numpy ndarray: updated array of gravity centers
    """

    # Decrementing for N to be user friendly
    N-=1

    int_map = int_map.reshape(-1, 3)
    
    lattice_parameter = np.array(lattice_parameter)
    new_centers = np.empty((len(cage_centers) * (N + 1), 3), dtype=cage_centers.dtype)
    new_int_map = np.empty((len(int_map) * (N + 1), 3), dtype=int_map.dtype)

    # Updating cages list
    new_cages = cages * (N + 1)
    size = np.max(int_map, axis=0)+1
    # Updating integer map
    for i in range(N + 1):
        new_centers[i * len(cage_centers):(i + 1) * len(cage_centers), :] = cage_centers + lattice_parameter[axis] * i * np.eye(3)[axis]
        new_int_map[i * len(int_map):(i + 1) * len(int_map), :] = int_map + size[axis] * i * np.eye(3)[axis]

    return new_int_map, new_cages, new_centers



#_____________________________________________________________________________________
def get_neighbors(guest_amount):
    """Gets neighbors of each cage in the system.

    Args:
        guest_amount (int): size of the unit cell in the integer coordinate system.

    Returns:
        numpy ndarray: Array containing the neighbors of each cage.
    """

    if structure == "sI":
        script_dir = os.path.dirname(__file__)
        neighbors = np.loadtxt(f'{script_dir}' + '/static/relatives_sI.dat', dtype=int)
        neighbors_map = np.zeros(guest_amount, dtype='object')
        edges_large = 14
        edges_small = 12
        delta_cage = guest_amount - 6

    elif structure == "sII":
        script_dir = os.path.dirname(__file__)
        neighbors = np.loadtxt(f'{script_dir}' + '/input/stats/relatives_sII.dat', dtype=int)
        neighbors_map = np.zeros(guest_amount, dtype='object')
        edges_large = 16
        edges_small = 12
        delta_cage = guest_amount - 8

    k = 0
    for i in range(guest_amount - delta_cage):

        neighbors_map[i] = neighbors[k:k + edges_large, :]
        k += edges_large

    for i in range(guest_amount - delta_cage, guest_amount):
        neighbors_map[i] = neighbors[k:k + edges_small, :]
        k += edges_small

    return neighbors_map



#_____________________________________________________________________________________
def get_neighbors_cage_type(neighbors_map, catalog, guest_amount):
    """Function giving type of cage for each site neighbor

    Args:
        neighbors_map (numpy ndarray): Array containing every neighbor for every site
        catalog (numpy ndarray): Array containing every site information
        guest_amount (int): size of the unit cell in the integer coordinate system

    Returns:
        numpy ndarray: Array containing 0 (1) for large (small) cages for the unit cell
    """

    size = np.max(catalog[:guest_amount,1:4], axis=0)+1
    cages_type = np.zeros(guest_amount, dtype='object')

    for i in range(len(neighbors_map)):
        # Applying PBC

        temp_map = (neighbors_map[i]+catalog[i,1:4])%(size)
        temp_cages = np.zeros(len(temp_map), dtype=int)

        for j in range(len(temp_map)):

            idx = np.where(temp_map[j,0]==catalog[:,1])[0]
            idy = np.where(temp_map[j,1]==catalog[:,2])[0]
            idz = np.where(temp_map[j,2]==catalog[:,3])[0]

            intersect_xy = np.intersect1d(idx, idy)
            intersect_xyz = np.intersect1d(idz, intersect_xy)
            temp_cages[j] = intersect_xyz[0]

        cages_type[i] = catalog[temp_cages, 4]

        del intersect_xy, intersect_xyz, temp_cages

    return cages_type



#_____________________________________________________________________________________
def get_neighbors_face_type(neighbors_map, catalog, guest_amount):
    """Function getting face type of transition

    Args:
        neighbors_map (numpy ndarray): Array containing every neighbor for every site
        catalog (numpy ndarray): Array containing every site information
        pattern_size (int): size of the unit cell in the new coordinate system

    Returns:
        numpy ndarray: transitions for the unit cell: 0: S5L, 1: L5S, 2: L5L, 3:L6L
    """
    unit_size = np.max(catalog[:guest_amount,1:4], axis=0)+1

    if structure == 'sI':
        faces_type = np.zeros(guest_amount, dtype='object')

        for i in range(guest_amount):

            temp_map = (neighbors_map[i]+catalog[i,1:4])%unit_size
            temp_face_type = np.zeros(len(temp_map))

            for j in range(len(temp_map)):

                # Checking if departure of large cage and arrival to small cage (L5S)
                if i<6 and (np.array_equal(temp_map[j,:],catalog[6,1:4]) or\
                    np.array_equal(temp_map[j,:],catalog[7,1:4])):
                    temp_face_type[j] = 1

                # Checking if departure of large cage arrival to large cage (L6L)
                elif (temp_map[j,0] == catalog[i,1] and temp_map[j,1] == catalog[i,2]) or \
            (temp_map[j,1] == catalog[i,2] and temp_map[j,2] == catalog[i,3]) or \
            (temp_map[j,0] == catalog[i,1] and temp_map[j,2] == catalog[i,3]):
                    temp_face_type[j] = 3

                # If not any condition but it is a small cage (S5L)
                elif i>=6:
                    temp_face_type[j] = 0

                # Else it can only be (L5L)
                else:
                    temp_face_type[j] = 2

            faces_type[i] = temp_face_type

            del temp_map, temp_face_type

    elif structure == 'sII':
        faces_type = np.zeros(guest_amount, dtype='object')

        for i in range(guest_amount):
            temp_map = (neighbors_map[i] + catalog[i,1:4])%unit_size
            temp_face_type = np.zeros(len(temp_map))

            for j in range(len(temp_map)):

                for k in range(guest_amount):
                    # L6L
                    if np.array_equal(temp_map[j,:], catalog[k,1:4]) and i<8 and k<8:
                        temp_face_type[j] = 3

                    # L5S
                    elif np.array_equal(temp_map[j,:], catalog[k,1:4]) and i<8 and k>=8:
                        temp_face_type[j] = 1

                    # S5L
                    elif np.array_equal(temp_map[j,:], catalog[k,1:4]) and i>=8 and k<8:
                        temp_face_type[j] = 0

                    # S5S
                    elif np.array_equal(temp_map[j,:], catalog[k,1:4]) and i>=8 and k>=8:
                        temp_face_type[j] = 2

            faces_type[i] = temp_face_type

    return faces_type


def move_molecule(catalog, neighbors_map, candidates, index_to_transitions, selected_transition, molecule_locations):
    """Function moving a molecule from the hopping site to the chosen one

    Args:
        catalog (numpy ndarray (int)): contains site states
        neighbors_map (numpy ndarray (object)): contains neighbors states for each site
        candidates (numpy ndarray (int)): contains cages from which a molecule can perform a specific transition
        index_to_transitions (dict): contains id as keys and transitions as values
        selected_transition (int): id of the selected transition
        molecule_locations (sp lil matrix): current molecules locations

    Returns:
        int: id of hopping site
        int: id of chosen site
        sp lil matrix: updated molecules locations
    """

    # Filtering which molecules respond to the given selected transition
    possible_transitions = candidates[selected_transition][candidates[selected_transition] > -1]

    # Picking one randomly
    hopping_site_index = int(random.random()*len(possible_transitions))
    hopping_site = possible_transitions[hopping_site_index]

    departure, arrival, channel = index_to_transitions[selected_transition]

    # Gathering neighbors of selected particle
    neighboring_sites = neighbors_map[hopping_site][neighbors_map[hopping_site][:,-1]==channel]

    # Filtering occupied neighbors
    available_sites_mask = catalog[neighboring_sites[:,0],5] == arrival
    available_sites = neighboring_sites[available_sites_mask, 0]

    # Choosing an empty neighbor randomly
    chosen_site_index = int(random.random()*len(available_sites))
    chosen_site = available_sites[chosen_site_index]

    # Updating departure and arrival sites occupancies
    catalog[hopping_site, 5] += 1
    catalog[chosen_site, 5] -= 1

    # Updating molecule location in site
    if departure == 0:
        if arrival == 1:
            molecule_locations[catalog[hopping_site, 8]] = catalog[chosen_site, 0]
            catalog[chosen_site, 8] = catalog[hopping_site, 8]
            catalog[hopping_site, 8] = -1
        else:
            molecule_locations[catalog[hopping_site, 8]] = catalog[chosen_site, 0]
            catalog[chosen_site, 7] = catalog[hopping_site, 8]
            catalog[hopping_site, 8] = -1

    elif departure == 1:
        if arrival == 1:
            molecule_locations[catalog[hopping_site, 7]] = catalog[chosen_site, 0]
            catalog[chosen_site, 8] = catalog[hopping_site, 7]
            catalog[hopping_site, 7] = -1
        else:
            molecule_locations[catalog[hopping_site, 7]] = catalog[chosen_site, 0]
            catalog[chosen_site, 7] = catalog[hopping_site, 7]
            catalog[hopping_site, 7] = -1

    return hopping_site, chosen_site, molecule_locations



#_____________________________________________________________________________________
def unwrap_trajectories(positions, times, new_time, lattice_parameter):
    """Function removing PBC for a single molecule trajectory recorded along the KMC simulation
    and recalibrating them to evenly spaced times

    Args:
        positions (numpy ndarray): positions of a given molecule along the simulation
        times (numpy ndarray): randomly incremented KMC times
        new_time (numpy ndarray): evenly spaced times scaled on simulation time
        lattice_parameter (numpy ndarray): Well, well, well...

    Returns:
        numpy ndarray: unwrapped positions
    """

    criteria = np.max(positions, axis=0)

    for coord in range(3):
        criterion = criteria[coord]
        diffs = np.diff(positions[:,coord])

        places = np.where(np.abs(diffs) >= criterion*0.5)[0]
        corrections = np.zeros_like(diffs)
        corrections[places] = - np.sign(diffs[places])*(lattice_parameter[coord]*system_size*1e-10)
        positions[1:,coord] += np.cumsum(corrections)

        interpolated_function = interp1d(times, positions[:,coord], kind='linear')
        positions[:,coord] = interpolated_function(new_time)

    return positions