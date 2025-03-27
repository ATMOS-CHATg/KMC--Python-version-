import initialization as init
import transitions
import spatial
from parameters import *
import numpy as np
import visualisation as visual
import gauge_management
import probes_management
import KMC
# from mayavi import mlab
import utils

# Contains the debugging mode

def run():

    KMC_step = 1
    keep_running = True
    system_size = 3
    figure = mlab.figure()
    pathway_dicts = {0: 'S5L', 1: 'L5S', 2: 'L5L (S5S)', 3: 'L6L'}

    # Loading structure topography and possible pathways (transitions)
    lattice_parameter, positions, oxygen_positions, _, oxygens, guest_amount = init.load_data()
    rate_constants = transitions.load_pathways(T)
    transitions_catalog = transitions.build_transition_inventory(rate_constants)

    # Building system by linking cages together
    cage_centers = spatial.gravity_center(positions)
    sites, cages, cage_centers = init.build_system(cage_centers, lattice_parameter,system_size)
    catalog = init.build_cages_catalog(sites, cages, guest_amount,system_size)
    neighbors_map = init.build_neighbors_map(catalog, guest_amount, system_size)

    positions = cage_centers.copy()

    # Filling system with respect the the mode and total occupancy
    catalog = init.fill_system(catalog, total_occupancy)

    # Building gas hydrate for visualisation
    oxygen_positions = visual.check_PBC(oxygen_positions, lattice_parameter)
    oxygen_positions = visual.add_patternO(oxygen_positions,0,lattice_parameter,system_size)
    oxygen_positions = visual.add_patternO(oxygen_positions,1,lattice_parameter,system_size)
    oxygen_positions = visual.add_patternO(oxygen_positions,2,lattice_parameter,system_size)
    connections_O = visual.connectionsO(oxygen_positions, oxygens, lattice_parameter)
    visual.unit_cell(lattice_parameter[0])

    guests_plot = visual.render_guests(catalog, positions)
    visual.render_cages(oxygen_positions, connections_O)
    figure.scene.disable_render = True
    visual.label_cages(catalog, positions)
    figure.scene.disable_render = False
    k = 0

    # Initializing the Bortz Kalos Lebowitz algorithm
    candidates, gauge = KMC.init_BKL(catalog, neighbors_map, rate_constants, transitions_catalog)

    while keep_running:

        times = np.zeros(iterations)
        rate_constants = rate_constants[rate_constants != 0]
        total_events = np.zeros((iterations, len(rate_constants)), dtype=int)
        occupancies, occupancies0 = init.init_occupancies(catalog)
        probe_stats, probes = probes_management.init_probes(catalog, cage_centers)
        guest_locations, current_guest_locations = init.init_guest_locations(catalog)
        current_guest_locations0 = current_guest_locations.copy()

        index_to_transitions, _ = transitions_catalog[0]

        selected_transition = gauge_management.pick_in_gauge(gauge)
        total_events[KMC_step, selected_transition] += 1

        old_guests_locations = catalog[catalog[:,5]==1][:,0]

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
        current_occupancy = KMC.recompute_occupancy(catalog)
        times[KMC_step] = KMC.increment_KMC_time(gauge)
        gauge = gauge_management.recompute_gauge(gauge, candidates, rate_constants)
        probe_stats = probes_management.refresh_probes(probe_stats, KMC_step, catalog, probes)

        occupancies[:,KMC_step] = current_occupancy - occupancies0
        occupancies0 = current_occupancy.copy()

        guest_locations[KMC_step,:] = current_guest_locations - current_guest_locations0
        current_guest_locations0 = current_guest_locations.copy()

        KMC_step += 1

        new_guests_locations = catalog[catalog[:,5]==1][:,0]
        arrival = np.setdiff1d(new_guests_locations, old_guests_locations)[0]
        chosen = np.setdiff1d(old_guests_locations, new_guests_locations)[0]
        transition = neighbors_map[chosen][neighbors_map[chosen][:,0]==arrival]

        if len(transition[:,0]) > 1:
            transition = transition[0,-1]
        else:
            transition = transition[0][-1]
        print(f'Molecule {catalog[arrival,7]} has moved from site {chosen} to site {arrival} ({pathway_dicts[transition]})')
        guests_plot.mlab_source.trait_set(x=positions[new_guests_locations,0],y=positions[new_guests_locations,1],z=positions[new_guests_locations,2], scale_factor=1, color=(0,1,0))
        if k == 0:
            plot2 = mlab.points3d(positions[chosen,0],positions[chosen,1],positions[chosen,2], color=(0,1,0), opacity=0.2, scale_factor=1.25)
            plot3 = mlab.points3d(positions[arrival,0], positions[arrival,1], positions[arrival,2], color=(0,1,1), opacity=1, scale_factor=1.25)
            k = 1
        else:
            plot2.mlab_source.trait_set(x=positions[chosen,0],y=positions[chosen,1],z=positions[chosen,2], color=(0,1,0))
            plot3.mlab_source.trait_set(x=positions[arrival,0],y=positions[arrival,1],z=positions[arrival,2], color=(0,1,0))

        utils.display_results_after_iteration(candidates, index_to_transitions, pathway_dicts, gauge, rate_constants)
        break_sim = input('Proceed to next step ? (y/n) : ')
        if break_sim == 'n':
            keep_running = False