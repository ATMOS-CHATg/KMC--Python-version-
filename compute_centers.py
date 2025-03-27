import utils
import numpy as np
import visualisation as visual
from mayavi import mlab
from parameters import *

# Script computing gravity centers of guest for any specified molecule
# Returns an error if not enough species are found

atomic_masses = utils.read_atomic_masses()
molecules, oxygen_positions, lattice_parameter, oxygen_amount = utils.extract_molecule_coordinates(input_structure_file, guest_molecule, num_molecules)

gravity_centers = utils.compute_guest_gravity_center(molecules, lattice_parameter)
gravity_centers = utils.map_centers_to_system_size(gravity_centers, num_molecules, lattice_parameter, system_size)

print(gravity_centers)
np.savetxt(f'output/gravity_centers/gravity_centers_{guest_molecule}_{functional}_{system_size}x{system_size}x{system_size}.dat', gravity_centers)

figure = mlab.figure()
oxygen_positions = visual.check_PBC(oxygen_positions, lattice_parameter)
connections_O = visual.connectionsO(oxygen_positions, oxygen_amount, lattice_parameter)
visual.render_cages(oxygen_positions, connections_O)


for i, x in enumerate(gravity_centers[:num_molecules, :]):
    mlab.text3d(x[0]+0.1, x[1]+0.1, x[2]+0.6, f'{i+1}', scale=(.25, .25, .25))
    mlab.points3d(gravity_centers[i, 0],gravity_centers[i, 1],gravity_centers[i, 2], scale_factor=1.2, color=(0,0,1))

mlab.show()