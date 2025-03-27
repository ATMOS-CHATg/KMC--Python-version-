import numpy as np
import matplotlib.pyplot as plt
import utils

# Script fitting the msd to obtain diffusion coefficients

folder_name = "sI_SO_270_0.1_vdWDF2"

parameters = utils.read_parameters(folder_name)

msd = np.loadtxt(f'output/diffusion/{folder_name}/msd.dat')

total_occupancy = float(parameters['Total_occupancy'])
system_size = int(parameters['System_size'])
structure = parameters['Structure']
mode = parameters['Mode']

if structure == 'sI':
    sites = 8
    if mode == 'DO':
        sites = 14

if structure == 'sII':
    sites = 24
    if mode == 'DO':
        sites = 32

N = int(sites*total_occupancy*system_size**3)

# Default values for 1e6 iterations: 50000 and 25000
N_fit_self = 5000
N_fit_jump = 2500

times = msd[0]
D_self = msd[2]
D_jump = msd[3]

t_fit = times[..., np.newaxis]
slope = np.linalg.lstsq(t_fit[:N_fit_self], D_self[:N_fit_self], rcond=None)[0][0]
D_self_coeff = slope / (6*int(total_occupancy*N))


slope = np.linalg.lstsq(t_fit[:N_fit_jump], D_jump[:N_fit_jump], rcond=None)[0][0]
D_jump_coeff = slope / (6*int(total_occupancy*N))

print(f'{D_jump_coeff=}', f'{D_self_coeff=}', '(m²/s)')

plt.plot(msd[0], msd[2], label='D_self')
plt.vlines(times[N_fit_self], np.amin([D_self, D_jump]), np.amax([D_self, D_jump]), label='Fit interval for D_self')
plt.plot(msd[0], msd[3], label='D_jump')
plt.vlines(times[N_fit_jump], np.amin([D_self, D_jump]), np.amax([D_self, D_jump]), label='Fit interval for D_jump', linestyles='-.')
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('MSD (m²)')
plt.grid()
plt.show()
