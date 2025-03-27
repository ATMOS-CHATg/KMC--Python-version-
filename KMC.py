import numpy as np
import gauge_management
import spatial
import transitions
import initialization as init
import probes_management
import utils
from parameters import *
import time
import scipy.sparse as sp
import debug

# Contain functions handling KMC moves

#_____________________________________________________________________________________
def init_BKL(catalog, neighbors_map, rate_constants, transitions_catalog):
    """Function initializating the Bortz Kalos Lebowitz algorithm
    (https://cmsr.rutgers.edu/images/people/lebowitz_joel/publications/1975jcp-bortz_kalos_113.pdf)

    Args:
        catalog (numpy ndarray (int)): contains site states
        neighbors_map (numpy ndarray (object)): contains neighbors states for each site
        rate_constants (numpy ndarray (float)): contains rate constants computed with the Arrhenius law
        transitions_catalog (object): contains 4 dictionaries allowing to browse for allowed transitions (resp. ids) 
                                        resp. forbidden transitions (resp. forbidden ids)

    Returns:
        numpy ndarray (int): candidates: contains cages from which a molecule can perform a specific transition
        numpy ndarray (float): gauge: contains the cumulative sum of rate constants ponderated by the amount of candidates 
    """

    forbidden_transitions = transitions_catalog[1][0]
    events_amount, candidates = init.init_events(catalog, neighbors_map, rate_constants, forbidden_transitions)
    gauge = gauge_management.build_gauge(events_amount, rate_constants)

    return candidates, gauge



#_____________________________________________________________________________________
def increment_KMC_time(gauge):
    """Randomly increment KMC times with respect to the KMC theory
    (https://en.wikipedia.org/wiki/Kinetic_Monte_Carlo#Rejection-free_KMC)

    Args:
        gauge (numpy ndarray (float)): contains the cumulative sum of rate constants ponderated by the amount of candidates

    Returns:
        float: computed time
    """

    t = -np.log(np.random.rand())/(gauge[-1])

    return t



#_____________________________________________________________________________________
def recompute_occupancy(catalog):
    """Recompute cages occupancy at a given time by checking all cage states

    Args:
        catalog (numpy ndarray (int)): contains site states

    Returns:
        numpy ndarray (int): current system occupancy with respect to cage types
    """

    # index 4 refers to the cage type in the catalog
    small_cages = catalog[catalog[:,4]==1]
    large_cages = catalog[catalog[:,4]==0]

    # index 5 refers to the current occupancy (2: free, 1: simple, 0: full)
    # ss: small simple, ls: large simple, ld: large double
    ss_occupancy = int(len(small_cages[small_cages[:,5]==1]))
    ls_occupancy = int(len(large_cages[large_cages[:,5]==1]))
    ld_occupancy = int(len(large_cages[large_cages[:,5]==0]))

    current_occupancy = np.array([ss_occupancy, ls_occupancy, ld_occupancy])

    return current_occupancy



#_____________________________________________________________________________________
def KMC_core(iterations, neighbors_map, catalog, gauge, candidates, rate_constants, transitions_catalog, cage_centers):
    """Simulation loop

    Args:
        iterations (int): maximum amount of iterations
        neighbors_map (numpy ndarray (object)): contains neighbors states for each site
        catalog (numpy ndarray (int)): contains site states
        gauge (numpy ndarray (float)): contains the cumulative sum of rate constants ponderated by the amount of candidates
        candidates (numpy ndarray (int)): contains cages from which a molecule can perform a specific transition
        rate_constants (numpy ndarray (float)): contains rate constants computed with the Arrhenius law
        transitions_catalog (object): contains 4 dictionaries allowing to browse for allowed transitions (resp. ids) 
        cage_centers (numpy ndarray): contains all cage centers 
            (cage_centers is not used to compute any physical properties, 
            only to get the 'probe_size' nearest neighbors of a cage in order to set up the probes)
        MP (bool, optional): if ran in parallel. Defaults to False.

    Returns:
        list: contains occupancies, probe statistics, times, molecule trajectories and events along the whole simulation
    """

    # Initializing observables
    times = np.zeros(iterations)
    rate_constants = rate_constants[rate_constants != 0]
    total_events = np.zeros((iterations, len(rate_constants)), dtype=int)
    occupancies, occupancies0 = init.init_occupancies(catalog)
    probe_stats, probes = probes_management.init_probes(catalog, cage_centers)
    guest_locations, current_guest_locations = init.init_guest_locations(catalog)
    current_guest_locations0 = current_guest_locations.copy()

    bar = utils.progress_bar()

    for KMC_step in range(1,iterations):
        print(50*'_')
        print(catalog)
        print(neighbors_map)
        index_to_transitions, _ = transitions_catalog[0]

        selected_transition = gauge_management.pick_in_gauge(gauge)
        total_events[KMC_step, selected_transition] += 1

        hopping_site, chosen_site, current_guest_locations = spatial.move_molecule(catalog,
                                                                                neighbors_map,
                                                                                candidates,
                                                                                index_to_transitions,
                                                                                selected_transition,
                                                                                current_guest_locations)

        candidates = transitions.clear_departure_and_arrival_transitions(candidates, 
                                                        hopping_site, 
                                                        chosen_site)

        candidates = transitions.refresh_departure_site(selected_transition,
                                            catalog, 
                                            neighbors_map,
                                            hopping_site, 
                                            chosen_site, 
                                            transitions_catalog,
                                            candidates)

        candidates = transitions.refresh_arrival_site(selected_transition,
                                        catalog, 
                                        neighbors_map,
                                        hopping_site, 
                                        chosen_site, 
                                        transitions_catalog,
                                        candidates)

        # Recomputing observables at current iteraiton
        current_occupancy = recompute_occupancy(catalog)
        times[KMC_step] = increment_KMC_time(gauge)
        gauge = gauge_management.recompute_gauge(gauge, candidates, rate_constants)
        probe_stats = probes_management.refresh_probes(probe_stats, KMC_step, catalog, probes)

        occupancies[:,KMC_step] = current_occupancy - occupancies0
        occupancies0 = current_occupancy.copy()

        guest_locations[KMC_step,:] = current_guest_locations - current_guest_locations0
        current_guest_locations0 = current_guest_locations.copy()

        bar.update(KMC_step)
        print(50*'_')
        print(catalog)
        print(neighbors_map)
    guest_locations = guest_locations.tocoo()
    occupancies = occupancies.tocoo()

    return [occupancies, probe_stats, times, guest_locations, total_events]



#_____________________________________________________________________________________
def simulate(T, total_occupancy):
    """Function running the whole simulation

    Returns:
        list: contains occupancies, probe statistics, times, molecule trajectories and events along the whole simulation
    """

    # Loading structure topography and possible pathways (transitions)
    lattice_parameter, positions, _, _, _, guest_amount = init.load_data()
    rate_constants = transitions.load_pathways(T)
    transitions_catalog = transitions.build_transition_inventory(rate_constants)

    # Building system by linking cages together
    cage_centers = spatial.gravity_center(positions)
    sites, cages, cage_centers = init.build_system(cage_centers, lattice_parameter)

    catalog = init.build_cages_catalog(sites, cages, guest_amount)
    neighbors_map = init.build_neighbors_map(catalog, guest_amount)

    # Filling system with respect the the mode and total occupancy
    catalog = init.fill_system(catalog, total_occupancy)
    # Initializing the Bortz Kalos Lebowitz algorithm
    candidates, gauge = init_BKL(catalog, neighbors_map, rate_constants, transitions_catalog)

    print(50*'_')
    print(gauge)
    # Running simulation
    func_args = [neighbors_map, catalog, gauge, candidates, rate_constants, transitions_catalog, cage_centers]
    data = KMC_core(iterations, *func_args)

    return data



#_____________________________________________________________________________________
def run():

    parameter_dictionary = utils.copy_parameters()
    tinit = time.time()

    def update(*a):
        pbar.update()

    total_occupancies, temperatures, scale = utils.get_params()

    if preset == 'parallel':
        from tqdm import tqdm
        import multiprocessing as mp

        pbar = tqdm(total=len(total_occupancies))
        pool = mp.Pool()
        results = []
        for i in range(len(total_occupancies)):
            result = pool.apply_async(simulate, args=([temperatures[i], total_occupancies[i]]), callback=update)
            results.append([result,total_occupancies[i]])

        pool.close()
        pool.join()
        pbar.close()

        for result in range(len(results)):
            occupancies = results[result][0].get()[0] 
            probe_stats = results[result][0].get()[1]
            times = results[result][0].get()[2]
            trajectories = results[result][0].get()[3]
            events = results[result][0].get()[4]
            data = [occupancies, probe_stats, times, trajectories, events]
            parameter_dictionary['Temperature'] = temperatures[result]
            parameter_dictionary['Total_occupancy'] = total_occupancies[result]
            utils.save_results(data, parameter_dictionary, temperatures[result], total_occupancies[result])

        final_time = np.round(time.time()-tinit,1)

        return print(f'Computational time of {final_time}s for {iterations*scale} iterations.')

    elif preset == 'cluster' or preset == 'local':
        data = simulate(T, total_occupancy)
        utils.save_results(data, parameter_dictionary)

        final_time = np.round(time.time()-tinit,1)

        return print(f'Computational time of {final_time}s for {iterations} iterations.')

    elif preset == 'debug':
        debug.run()
