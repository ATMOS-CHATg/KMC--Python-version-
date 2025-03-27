# Parameters file

# Structure to be simulated (sI/sII)
structure = 'sI'

# How is it filled ? (SO/DO)
mode = 'DO'

# Are there vacancies ? (nominative)
vacancies = True

# Functional ? (nominative)
functional = 'vdWDF2'

# Which mode to run the KMC in (local, cluster, parallel, debug)
preset = 'cluster'

# Input transitions file (to be put in input/transitions)
input_pathways = 'transitions.dat'

# Total occupancy of the system (0<x<1)
total_occupancy = 0.9

# Temperature (K)
T = 270

# System size (it is cubed after)
system_size = 1

# Amount of KMC steps
iterations = 2

# Probe configurations (probe_size cannot exceed system_size**3)
probe_size = 1
probe_amount = 1

# Rattling frequencies in cages (Hz)
rattling_small = 3e12
rattling_large = 1.5e12

# If parallel mode, which parameter to scale the simulations on (Density/Temperature)
parameter = 'Density'

# Minimum value of the scaled parameter (use 5 for 5% density, not 0.05)
min_value = 5
# Maximum value
max_value = 90
# Step between values
step = 10

# Structure topology to compute cage centers from (msd purpose)
input_structure_file = 'input/structure/sI'

# Guest type (to be recognized by the compute_centers.py script)
guest_molecule = 'N2'

# Amount of guest molecules in the unit cell (sI:8, sII:24)
num_molecules = 8

# If D_self should also be computed (extensive)
Compute_D_self = False
