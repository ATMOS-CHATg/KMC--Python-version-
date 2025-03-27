
# from mayavi import mlab
import numpy as np
from numpy.linalg import norm

# Visualisation tools

#_____________________________________________________________________________________
# Checking for boundaries (Oxygen connections purpose)
def check_PBC(coord,lattice_param, oxygen=False):
    updated_coords = coord.copy()
    for i in range(len(coord[:,0])):
        if updated_coords[i,0] == 0:
            updated_coords[i,0] += lattice_param[0]

        if updated_coords[i,1] == 0:
            updated_coords[i,1] += lattice_param[1]

        if updated_coords[i,2] == 0:
            updated_coords[i,2] += lattice_param[2]

    A = np.argwhere(coord == 0)[:,0]
    B = np.append(updated_coords,coord[A,:])

    if oxygen:
        flag = len(A)
        B = B.reshape(int(len(B)/3),3)
        return np.unique(B, axis=0), flag

    B = B.reshape(int(len(B)/3),3)

    return np.unique(B, axis=0)



#_____________________________________________________________________________________
# Adding N O unit cells along axis
def add_patternO(H2O_pos, axis, lattice_param, N=1):
    N-=1
    q = H2O_pos

    for i in range(1,N+1):
        H2O_pos = np.append(H2O_pos,q,axis=0)
        H2O_pos[i*len(q):,axis] += i*lattice_param[axis]

    return H2O_pos



#_____________________________________________________________________________________
# Connecting Oxygen (USE ONLY WHEN FULL rO IS BUILT)
def connectionsO(H20_pos, N_H20, lattice_param):
    connectionsO = np.array([[0,0]])
    added_atoms = check_PBC(H20_pos,lattice_param,True)[1]
    for i in range(0,len(H20_pos[:,0])):

        B = norm([H20_pos[i,0]-H20_pos[i:i+added_atoms+N_H20,0],
                H20_pos[i,1]-H20_pos[i:i+added_atoms+N_H20,1],
                H20_pos[i,2]-H20_pos[i:i+added_atoms+N_H20,2]], 
                axis=0)

        A = np.where(B<3)[0]+i

        C = np.array([A[1:],[i]*len(A[1:])]).transpose()

        connectionsO = np.append(connectionsO,C,axis=0)

    connectionsO = np.unique(connectionsO, axis=0)

    return connectionsO



#_____________________________________________________________________________________
# Drawing unit cell 
def unit_cell(lattice_param):
    mlab.plot3d([lattice_param, 2*lattice_param], [lattice_param, lattice_param], [lattice_param, lattice_param], color=(1,1,1),line_width=10)
    mlab.plot3d([lattice_param, lattice_param], [lattice_param, 2*lattice_param], [lattice_param, lattice_param], color=(1,1,1),line_width=10)
    mlab.plot3d([lattice_param, lattice_param], [lattice_param, lattice_param], [lattice_param, 2*lattice_param], color=(1,1,1),line_width=10)
    mlab.plot3d([2*lattice_param, 2*lattice_param], [2*lattice_param, 2*lattice_param], [lattice_param, 2*lattice_param], color=(1,1,1),line_width=10)
    mlab.plot3d([2*lattice_param, 2*lattice_param], [lattice_param, 2*lattice_param], [2*lattice_param, 2*lattice_param], color=(1,1,1),line_width=10)
    mlab.plot3d([lattice_param, 2*lattice_param], [2*lattice_param, 2*lattice_param], [2*lattice_param, 2*lattice_param],  color=(1,1,1),line_width=10)
    mlab.plot3d([lattice_param, lattice_param], [2*lattice_param, 2*lattice_param], [lattice_param, 2*lattice_param],  color=(1,1,1),line_width=10)
    mlab.plot3d([2*lattice_param, 2*lattice_param], [lattice_param, lattice_param], [lattice_param, 2*lattice_param],  color=(1,1,1),line_width=10)
    mlab.plot3d([lattice_param, lattice_param], [lattice_param, 2*lattice_param], [2*lattice_param, 2*lattice_param],   color=(1,1,1),line_width=10)
    mlab.plot3d([2*lattice_param, 2*lattice_param], [lattice_param, 2*lattice_param], [lattice_param, lattice_param],   color=(1,1,1),line_width=10)
    mlab.plot3d([lattice_param, 2*lattice_param], [lattice_param, lattice_param], [2*lattice_param, 2*lattice_param],   color=(1,1,1),line_width=10)
    mlab.plot3d([lattice_param, 2*lattice_param],[2*lattice_param, 2*lattice_param], [lattice_param, lattice_param],    color=(1,1,1),line_width=10)



#_____________________________________________________________________________________
def render_cages(oxygen_positions, connections_O):
    ptsO = mlab.points3d(oxygen_positions[:,0],oxygen_positions[:,1],oxygen_positions[:,2], scale_factor=.50, color=(1,0,0))
    ptsO.mlab_source.dataset.lines = connections_O
    lines = mlab.pipeline.stripper(ptsO)
    mlab.pipeline.surface(lines, color=(1,0,0), line_width=5, opacity=.4)



#_____________________________________________________________________________________
def render_guests(catalog, guest_positions):
    occupied_cages = catalog[catalog[:,5]==1][:,0]
    guests_plot = mlab.points3d(guest_positions[occupied_cages,0],guest_positions[occupied_cages,1],guest_positions[occupied_cages,2], scale_factor=1, color=(0,0,1))
    return guests_plot



#_____________________________________________________________________________________
def label_cages(catalog, positions):
    cages = catalog[:,4]
    cages_idx = {0: 'L', 1: 'S'}
    for i, x in enumerate(positions):
        mlab.text3d(x[0]+0.1, x[1]+0.1, x[2]+0.6, f'{i}{cages_idx[cages[i]]}', scale=(.25, .25, .25))