import numpy as np
import logging

from helper import normalize

logger = logging.getLogger(__name__)

# this module contains functions for calculating the tangent vectors
# of an NEB path. New/changed methods of tangent calculation should
# be added in this module.

def simple_tans(full_path_pvecs, internals=[None]):
    """
    This function calculates the simple tangent according to the definition
    in the original NEB implementation.\n
    Jónsson, H.; Mills, G.; Jacobsen, K. W. Nudged Elastic Band Method for Finding Minimum Energy Paths of 
    Transitions. In Classical and Quantum Dynamics in Condensed Phase Simulations; WORLD SCIENTIFIC: LERICI, 
    Villa Marigola, 1998; pp 385–404. https://doi.org/10.1142/9789812839664_0016.
    """
    if internals[0] is None:
        tanvecs = [normalize(full_path_pvecs[i+1] - full_path_pvecs[i-1])
                   for i in range(1, len(full_path_pvecs)-1)]
    else:
        # Internals are given, we must translate
        tanvecs = [internals[i-1].get_Internal_Values(full_path_pvecs[i+1]) 
                   - internals[i-1].get_Internal_Values(full_path_pvecs[i-1]) 
                   for i in range(1, len(full_path_pvecs)-1)]
    return np.array(tanvecs)


def henkjon_tan_single(pos, prevpos, nextpos, en, preven, nexten):
    """
    This function calculates the Henkelman-Jonsson tangent for one image
    """
    if en is None or preven is None or nexten is None:
        # henkelman-jonsson tangents cant be computed in this case.
        # fall back on simple tangent for this image.
        return normalize(nextpos - prevpos)

    tau_plus = nextpos - pos
    tau_minus = pos - prevpos
    delta_e_max = np.max([np.abs(nexten - en), np.abs(preven - en)])
    delta_e_min = np.min([np.abs(nexten - en), np.abs(preven - en)])

    if nexten > en and en > preven:
        tanvec = tau_plus
    elif nexten < en and en < preven:
        tanvec = tau_minus
    elif (nexten > en and en < preven) or (nexten < en and en > preven):
        if nexten > preven:
            tanvec = tau_plus * delta_e_max + tau_minus * delta_e_min
        else:
            tanvec = tau_minus * delta_e_max + tau_plus * delta_e_min
    else:
        # nexten == en and/or en == preven
        tanvec = tau_plus + tau_minus
    return normalize(tanvec)


def henkjon_tans(full_path_pvecs, full_img_energies, internals=[None]):
    """
    This function calculates the Henkelman-Jonsson tangent according to:\n
    Henkelman, G.; Jónsson, H. Improved Tangent Estimate in the Nudged Elastic Band 
    Method for Finding Minimum Energy Paths and Saddle Points. J. Chem. Phys. 2000, 113 (22), 
    9978–9985. https://doi.org/10.1063/1.1323224.
    """
    # expects len(img_pvecs) == len(img_energies)
    tanvecs = []

    for i in range(1, len(full_path_pvecs)-1):
        if internals[0] is None:
            prev_pvec = full_path_pvecs[i-1]
            pvec = full_path_pvecs[i]
            next_pvec = full_path_pvecs[i+1]
        else:
            # Internals are given
            prev_pvec = internals[i-1].get_Internal_Values(full_path_pvecs[i-1])
            pvec = internals[i-1].get_Internal_Values(full_path_pvecs[i])
            next_pvec = internals[i-1].get_Internal_Values(full_path_pvecs[i+1])

        tanvec = henkjon_tan_single(pvec,
                                    prev_pvec,
                                    next_pvec,
                                    full_img_energies[i],
                                    full_img_energies[i-1],
                                    full_img_energies[i+1])
        tanvecs.append(tanvec)
    return np.array(tanvecs)
