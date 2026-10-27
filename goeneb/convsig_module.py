import numpy as np

# functions for calculation of various signals of NEB convergence.
# New signals of convergence functions should be added in this module.


def NEB_RMSF(nebgrads):
    return np.max([RMS(grad) for grad in nebgrads])


def NEB_ABSF(nebgrads):
    return np.max([MaxAbs(grad) for grad in nebgrads])


def NEB_AbsDeltaE(current_ens, prev_ens):
    # should return nan if there are any Nones among the
    # values to signify there are still failed images in the path
    return np.max([Abs_DeltaE(cur, prev) for cur, prev
                  in zip(current_ens, prev_ens)])


def NEB_ImgRMSD(current_pvecs, prev_pvecs):
    return np.max([RMSD(cur, prev) for cur, prev
                  in zip(current_pvecs, prev_pvecs)])


# helper functions for various convergence signal functions
def RMS(vector):
    return np.sqrt(np.average(vector * vector))


def RMSD(vector1, vector2):
    return RMS(vector1 - vector2)


def MaxAbs(vector):
    return np.max(np.abs(vector))


def DeltaE(E1, E2):
    # failed images' energies are given as 'None',
    # so we must treat these values with special care
    try:
        result = E1 - E2

    except TypeError:
        result = np.nan

    return result


def Abs_DeltaE(E1, E2):
    return abs(DeltaE(E1, E2))

def yesno(condition):
    if condition:
        return 'YES'
    else:
        return 'NO'