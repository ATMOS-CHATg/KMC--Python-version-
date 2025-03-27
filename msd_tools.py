import numpy as np
import spatial

# Contains functions which compute the MSD

def autocorrelation_FFT(x):
    """Computes the autocorrelation of a 1D signal using FFT.

    Args:
        x (numpy ndarray): contains molecule trajectories along a given coordinate

    Returns:
        numpy ndarray: The normalized autocorrelation of x.
    """

    N = len(x)
    F = np.fft.fft(x, n=2 * N)

    # Compute the power spectral density (PSD)
    PSD = F * F.conjugate()

    # Compute the inverse FFT of the PSD to obtain autocorrelation
    autocorr = np.fft.ifft(PSD)
    autocorr = autocorr[:N].real   

    # Normalization factors to account for the decreasing number of terms
    normalization_factors = N * np.ones(N) - np.arange(0, N)

    return autocorr / normalization_factors 


def msd_fft(r):
    """Computes the mean squared displacement (MSD) using FFT-based autocorrelation.

    Args:
        r (2D array): Molecule trajectories along time

    Returns:
        array: The mean squared displacement (MSD) for each time lag.
    """

    N = len(r)

    # Calculate the squared distances of each position vector
    squared_distances = np.square(r).sum(axis=1)
    squared_distances = np.append(squared_distances, 0)

    # Autocorrelations for each axis
    sum_autocorrelations = sum([autocorrelation_FFT(r[:, i]) for i in range(r.shape[1])])

    cumulative_displacement_sum = 2 * squared_distances.sum()
    msd = np.zeros(N)

    # Compute MSD for each time lag m
    for m in range(N):
        cumulative_displacement_sum -= squared_distances[m - 1] + squared_distances[N - m]
        msd[m] = cumulative_displacement_sum / (N - m)

    return msd - 2 * sum_autocorrelations


def compute_msd(centers, trajectories, times, lattice_parameter, Compute_D_self = False):
    """Function computing the mean squared displacement of molecules/system center of mass

    Args:
        centers (numpy ndarray): contains the guest gravity centers
        trajectories (scipy lil matrix): contains the guest locations
        times (numpy ndarray): cumulated KMC times
        lattice_parameter (numpy ndarray): ...
        Compute_D_self (bool, optional): If D_self is also computed. Defaults to False.

    Returns:
        numpy ndarray: msd for D_self
        numpy ndarray: system gravity center
        numpy ndarray: evenly spaced times
    """

    N = trajectories.shape[1]
    evenly_spaced_times = np.zeros(len(times))
    dt = times[-1]/len(times)
    for j in range(len(evenly_spaced_times)):
        evenly_spaced_times[j] = j*dt

    msd = 0
    r_tot = 0
    for i in range(N):
        r = centers[np.cumsum(trajectories.tocsr()[:,i].toarray()).astype(int)]
        r = spatial.unwrap_trajectories(r, times, evenly_spaced_times, lattice_parameter)
        r_tot += r
        if Compute_D_self:
            D_self_single = msd_fft(r)
            msd += D_self_single

    return msd, r_tot, evenly_spaced_times