import numpy as np
import time
import scipy.sparse as sp
from parameters import *
import utils
import msd_tools

# Script computing observables: times, events, occupancies, probes, msd of molecules and msd of center of mass

simulation_folder = 'sI_SO_270_0.1_vdWDF2'
gravity_centers_file = 'gravity_centers_N2_vdWDF2_8x8x8.dat'

parameters = utils.read_parameters(simulation_folder)
iterations = int(parameters['Iterations'])

_ = utils.manage_folder(float(parameters['Temperature']), float(parameters['Total_occupancy']), simulation_folder)

_, _, lattice_parameter, _ = utils.extract_molecule_coordinates(input_structure_file, guest_molecule, num_molecules)
small, large = utils.get_cages_amount(float(parameters['System_size']))

t_init = time.time()
gravity_centers = np.loadtxt(f'output/gravity_centers/{gravity_centers_file}')*1e-10
times = np.loadtxt(f'output/KMC_simulations/{simulation_folder}/times.dat')
times = np.cumsum(times)

events = np.loadtxt(f'output/KMC_simulations/{simulation_folder}/events.dat')
avg_events = np.sum(events, axis=0)/float(parameters['Iterations'])*100
events = utils.unwrap_events(events, iterations)

probes = np.loadtxt(f'output/KMC_simulations/{simulation_folder}/probes.dat')
avg_probes = np.mean(probes, axis=0)

trajectories = sp.load_npz(f'output/KMC_simulations/{simulation_folder}/trajectories.npz')

occupancies = sp.load_npz(f'output/KMC_simulations/{simulation_folder}/occupancies.npz')
occupancies = utils.unwrap_occupancies(occupancies, iterations)
avg_occupancies = np.sum(occupancies, axis=1)

D_self_msd, r_tot, evenly_spaced_times = msd_tools.compute_msd(gravity_centers, trajectories, times, lattice_parameter, Compute_D_self)
D_jump_msd = msd_tools.msd_fft(r_tot)

np.savetxt(f'output/diffusion/{simulation_folder}/msd.dat', np.array([evenly_spaced_times, times, D_self_msd, D_jump_msd]))
np.savetxt(f'output/diffusion/{simulation_folder}/occupancies.dat', occupancies)
np.savetxt(f'output/diffusion/{simulation_folder}/events.dat', events)
np.savetxt(f'output/diffusion/{simulation_folder}/avg_occupancies.dat', avg_occupancies)
np.savetxt(f'output/diffusion/{simulation_folder}/avg_events.dat', avg_events)
np.savetxt(f'output/diffusion/{simulation_folder}/probes.dat', probes)
np.savetxt(f'output/diffusion/{simulation_folder}/avg_probes.dat', avg_probes)
print(f'Done in {np.round(time.time()-t_init,3)}s.')