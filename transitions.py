import numpy as np
import os
from parameters import *

# Contain functions which initialize and refresh transitions

#_____________________________________________________________________________________
def load_pathways(T):
    """Weeeell, it loads pathways ?

    Returns:
        numpy ndarray: contains pathways (s^-1)
    """

    rattling_freqs = [rattling_small, rattling_large]

    if structure == 'sI':
        pre_exp = 4*[0] + 12*[1]
    elif structure == 'sII':
        pre_exp = 4*[0] + 4*[1] + 4*[0] + 4*[1]

    script_dir = os.path.dirname(__file__)
    pathways = np.loadtxt(f'{script_dir}' + f'/input/stats/{input_pathways}', usecols=1, dtype=float)

    for current_pathway in range(len(pathways)):
        if pathways[current_pathway] == -1:
            pathways[current_pathway] = 0
        else:
            pathways[current_pathway] = rattling_freqs[pre_exp[current_pathway]]*np.exp(-(1.6e-19*6.02e23*pathways[current_pathway])/(8.314*T))

    return pathways



#_____________________________________________________________________________________
def build_transition_inventory(transitions):
    """Function mapping 

    Args:
        transitions (numpy ndarray): contains pathways

    Returns:
        object: dict: ids to transitions (integer values to corresponding transition)
                dict: transitions to ids (transitions to corresponding integer values)
                list: forbidden transitions
                list: forbidden transitions ids
                !! Refer to documentation in order to know the structure of transitions
    """

    ID_to_transition = {}
    transition_to_ID = {}

    forbidden_transitions = []
    forbidden_transitions_ids = []

    current_transition, current_id = 0, 0

    # SO/DO IDs (2: empty, 1: single occupancy, 0: full)
    occupancies = [[1,2], [1,1], [0,2], [0,1]]

    # For 'sI/II': (0: S5L, 1: L5S, 2: L5L [resp. S5S sII], 3: L6L)
    pathways = list(range(4))

    for current_pathway in pathways:

        for current_occupancy in occupancies:

            if transitions[current_transition] != 0:
                ID_to_transition[current_id] = current_occupancy + [current_pathway]
                transition_to_ID[tuple(current_occupancy+[current_pathway])] = current_id
                current_id+=1

            current_transition+=1

    current_transition = 0

    for current_pathway in pathways:

        for current_occupancy in occupancies:

            if transitions[current_transition] == 0:
                forbidden_transitions.append(tuple(current_occupancy+[current_pathway]))
                forbidden_transitions_ids.append(current_id)
                current_id += 1

            current_transition+=1

    return [ID_to_transition, transition_to_ID], [forbidden_transitions, forbidden_transitions_ids]



#_____________________________________________________________________________________
def clear_departure_and_arrival_transitions(candidates, hopping_site, chosen_site):
    """Function removing the hopping site and chosen site ids from candidates

    Args:
        candidates (numpy ndarray (int)): contains cages from which a molecule can perform a specific transition
        hopping_site (int): id of departure site
        chosen_site (int): id of arrival site

    Returns:
        numpy ndarray: updated candidates
    """

    for pathway in range(len(candidates)):
        locations = np.where(np.logical_or(candidates[pathway] == chosen_site, candidates[pathway] == hopping_site))[0]
        candidates[pathway][locations] = -1

    return candidates



#_____________________________________________________________________________________
def add_transition(candidates, transition_id, site):
    """Function adding a specific transition to a site which can now perform it

    Args:
        candidates (numpy ndarray (int)): contains cages from which a molecule can perform a specific transition
        old_transition_id (int): id of the transition
        site (int): id of the site

    Returns:
        numpy ndarray: updated candidates array
    """

    idx = np.argmax(candidates[transition_id] == -1)
    candidates[transition_id][idx] = site

    return candidates



#_____________________________________________________________________________________
def remove_transition(candidates, old_transition_id, site):
    """Function removing a site having a specific possible transition no longer available

    Args:
        candidates (numpy ndarray (int)): contains cages from which a molecule can perform a specific transition
        old_transition_id (int): id of the transition
        site (int): id of the site

    Returns:
        numpy ndarray: updated candidates array
    """

    idx = np.argmax(candidates[old_transition_id] == site)
    candidates[old_transition_id][idx] = -1

    return candidates



#_____________________________________________________________________________________
def invert_L5S_S5L_channels(current_channel):
    """Function inverting L5S/S5L channel
        (channel from a site to a neighbor is L5S means that
        channel from the same neighbor to the same site is S5L)

    Args:
        current_channel (int): S5L (0), L5S (1), L5L/S5S (2), L6L (3)

    Returns:
        int: new current channel
    """

    if current_channel == 0:
        return 1
    elif current_channel == 1:
        return 0
    else:
        return current_channel



#_____________________________________________________________________________________
def refresh_departure_site(selected_transition, catalog, neighbors_map, hopping_site, chosen_site, transitions_catalog, candidates):
    """_summary_

    Args:
        selected_transition (int): id of selected transition
        catalog (numpy ndarray (int)): contains site states
        neighbors_map (numpy ndarray (object)): contains neighbors states for each site
        hopping_site (int): id of departure site
        chosen_site (int): id of arrival site
        transitions_catalog (object): contains 4 dictionaries allowing to browse for allowed transitions (resp. ids) 
        candidates (numpy ndarray (int)): contains cages from which a molecule can perform a specific transition

    Returns:
        numpy ndarray: refreshed candidates array
    """

    id_to_transitions, transitions_to_id = transitions_catalog[0]
    departure, _, _ = id_to_transitions[selected_transition]
    forbidden_transitions = transitions_catalog[1][0]
    channels_to_neighbors = neighbors_map[hopping_site][:,-1]
    ids_of_neighbors = neighbors_map[hopping_site][:,0]
    neighbor_occupancies = catalog[ids_of_neighbors][:,5]

    for i in range(len(neighbor_occupancies)):
        current_arrival_occupancy = neighbor_occupancies[i]
        current_channel = channels_to_neighbors[i]
        current_neighbor_id = ids_of_neighbors[i]

        # Adding site to neighbor transitions
        test_transition = (departure+1, current_arrival_occupancy, current_channel)

        if current_arrival_occupancy > 0 and departure+1 < 2 and not test_transition in forbidden_transitions:
            transition_id = transitions_to_id[test_transition]
            candidates = add_transition(candidates, transition_id, hopping_site)

        # Checking if arrival site is occupied
        if current_arrival_occupancy < 2:

            # Inverting L5S (resp. S5L) transitions
            neighbor_to_site_channel = invert_L5S_S5L_channels(current_channel)

            # Removing old neighbor to site transitions
            test_transition = (current_arrival_occupancy, departure, neighbor_to_site_channel)

            if departure==1 and not current_neighbor_id == chosen_site and not test_transition in forbidden_transitions:
                old_transition_id = transitions_to_id[test_transition]
                candidates = remove_transition(candidates, old_transition_id, current_neighbor_id)

                # Adding it twice if doubly occupied
                if current_arrival_occupancy == 0:
                    old_transition_id = transitions_to_id[test_transition]
                    candidates = remove_transition(candidates, old_transition_id, current_neighbor_id)

            # Adding new neighbor to site transitions
            test_transition = (current_arrival_occupancy, departure+1, neighbor_to_site_channel)

            if not test_transition in forbidden_transitions:
                transition_id = transitions_to_id[test_transition]
                candidates = add_transition(candidates, transition_id, current_neighbor_id)

                # Adding it twice if doubly occupied
                if current_arrival_occupancy == 0:
                    transition_id = transitions_to_id[test_transition]
                    candidates = add_transition(candidates, transition_id, current_neighbor_id)

    return candidates



#_____________________________________________________________________________________
def refresh_arrival_site(selected_transition, catalog, neighbors_map, hopping_site, chosen_site, transitions_catalog, candidates):
    """Function refreshing arrival site states and interactions with its neighbors

    Args:
        selected_transition (int): id of selected transition
        catalog (numpy ndarray (int)): contains site states
        neighbors_map (numpy ndarray (object)): contains neighbors states for each site
        hopping_site (int): id of departure site
        chosen_site (int): id of arrival site
        transitions_catalog (object): contains 4 dictionaries allowing to browse for allowed transitions (resp. ids) 
        candidates (numpy ndarray (int)): contains cages from which a molecule can perform a specific transition

    Returns:
        numpy ndarray: refreshed candidates array
    """

    id_to_transitions, transitions_to_id = transitions_catalog[0]
    _, departure, _ = id_to_transitions[selected_transition]
    channels_to_neighbors = neighbors_map[chosen_site][:,-1]

    forbidden_transitions = transitions_catalog[1][0]
    ids_of_neighbors = neighbors_map[chosen_site][:,0]
    neighbor_occupancies = catalog[ids_of_neighbors][:,5]

    for j in range(len(neighbor_occupancies)):
        current_arrival_occupancy = neighbor_occupancies[j]
        current_channel = channels_to_neighbors[j]
        current_neighbor_id = ids_of_neighbors[j]

        # Adding new site to neighbor transitions
        test_transition = (departure-1, current_arrival_occupancy , current_channel)

        if current_arrival_occupancy > 0 and current_neighbor_id != hopping_site and not test_transition in forbidden_transitions:
            transition_id = transitions_to_id[test_transition]
            candidates = add_transition(candidates, transition_id, chosen_site)

            # Adding it twice if doubly occupied
            if departure-1 == 0:
                test_transition = (departure-1, current_arrival_occupancy , current_channel)
                if current_arrival_occupancy > 0 and current_neighbor_id != hopping_site and not test_transition in forbidden_transitions:
                    transition_id = transitions_to_id[test_transition]
                    candidates = add_transition(candidates, transition_id, chosen_site)

        # Checking if arrival site is occupied and different from the hopping site
        if current_arrival_occupancy < 2 and current_neighbor_id != hopping_site:

            # Inverting L5S (resp. S5L) transitions
            neighbor_to_site_channel = invert_L5S_S5L_channels(current_channel)

            # Removing old neighbor to site transitions
            test_transition = (current_arrival_occupancy, departure, neighbor_to_site_channel)

            if not test_transition in forbidden_transitions and not current_neighbor_id == chosen_site:
                old_transition_id = transitions_to_id[test_transition]
                candidates = remove_transition(candidates, old_transition_id, current_neighbor_id)

                # Removing it twice if neighbor is doubly occupied
                if current_arrival_occupancy == 0:
                    old_transition_id = transitions_to_id[test_transition]
                    candidates = remove_transition(candidates, old_transition_id, current_neighbor_id)

            # Adding new neighbor to site transitions
            test_transition = (current_arrival_occupancy, departure-1, neighbor_to_site_channel)

            if departure - 1 != 0 and not test_transition in forbidden_transitions:
                transition_id = transitions_to_id[test_transition]
                candidates = add_transition(candidates, transition_id, current_neighbor_id)

                # Adding it twice if neighbor is doubly occupied
                if current_arrival_occupancy == 0:
                    transition_id = transitions_to_id[test_transition]
                    candidates = add_transition(candidates, transition_id, current_neighbor_id)

    return candidates