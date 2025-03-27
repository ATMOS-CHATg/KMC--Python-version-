import numpy as np

# Contains functions which handle the gauge along the simulation

#_____________________________________________________________________________________
def build_gauge(event_amount, rate_constants):
    """Function building BKL gauge based on possible events

    Args:
        event_amount (list): Amount of candidates for each transition at first iteration
        rate_constants (numpy ndarray (float)): contains rate constants computed with the Arrhenius law

    Returns:
        numpy ndarray: gauge
    """

    rate_constants = rate_constants[rate_constants != 0]

    gauge = np.zeros(len(rate_constants)+1)

    for i in range(len(rate_constants)):
        gauge[i+1] = gauge[i] + event_amount[i]*rate_constants[i]
    print(gauge)
    return gauge



#_____________________________________________________________________________________
def recompute_gauge(gauge, candidates, rate_constants):
    """Function recomputing the gauge after a move

    Args:
        gauge (numpy ndarray (float)): contains the cumulative sum of rate constants ponderated by the amount of candidates
        candidates (numpy ndarray (int)): contains cages from which a molecule can perform a specific transition
        rate_constants (numpy ndarray (float)): contains rate constants computed with the Arrhenius law

    Returns:
        _type_: _description_
    """
    gauge = np.zeros(len(gauge))

    for i in range(1,len(candidates)+1):
        gauge[i] = len(candidates[i-1][candidates[i-1]>-1])*rate_constants[i-1] + gauge[i-1]

    return gauge



#_____________________________________________________________________________________
def pick_in_gauge(gauge):
    """Function picking transition from probablity gauge

    Args:
        gauge (np.ndarray): probability gauge

    Returns:
        int: Index to be picked in transitions
    """

    rho1 = np.random.uniform(low=0.0, high=np.max(gauge), size=(1,))
    pathway = np.searchsorted(gauge, rho1, side="right") - 1

    return pathway[0]