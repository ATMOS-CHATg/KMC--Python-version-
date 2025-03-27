from parameters import probe_amount, probe_size, iterations
import numpy as np
from numpy.linalg import norm

# Contain functions which track probes

def init_probes(catalog, cage_centers):
    """Function spreading probes in the system

    Args:
        catalog (numpy ndarray (int)): contains site states
        cage_centers (numpy ndarray (float)): contains cage centers in the xyz system

    Returns:
        numpy ndarray (int): amount of molecules in probes
        numpy ndarray (int): probes locations
    """

    probe_stats = np.zeros((iterations, probe_amount), dtype=int)
    probes = np.zeros((probe_size, probe_amount), dtype=int)

    for i in range(probe_amount):
        dist = norm(cage_centers[np.random.randint(0,len(catalog)),:] - cage_centers, axis=1)
        probes[:,i] = np.argsort(dist)[1:probe_size+1]
        probe_stats[0,i] = np.sum(catalog[probes[:,i],5]==1) + 2*np.sum(catalog[probes[:,i],5]==0)

    return probe_stats, probes

def refresh_probes(probe_stats, KMC_step, catalog, probes):
    """Function updating the amount of molecules in probes at a given time

    Args:
        probe_stats (numpy ndarray (int)): amount of molecules in probes
        KMC_step (int): current iteration
        catalog (numpy ndarray (int)): contains site states
        probes (numpy ndarray (int)): probes locations

    Returns:
        numpy ndarray (int): updated probe stats
    """

    for i in range(probe_amount):
        probe_stats[KMC_step,i] = np.sum(catalog[probes[:,i],5]==1) + 2*np.sum(catalog[probes[:,i],5]==0)

    return probe_stats