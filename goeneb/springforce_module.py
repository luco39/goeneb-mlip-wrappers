import numpy as np
import neb_exceptions as nex

# This module provides functions for calculating the spring forces
# between NEB images, as well as spring constants for variable k.
# New/altered methods of spring force calculation should be added
# in this module.

# basic implementation of springforce gradients for series of ideal springs
# -------------------------------------------------------------------------

def delta_springgrads(full_path_pvecs, img_pair_ks):
    """
    This function calculates the spring gradients based on the 
    distance between the images. Returns a list of floats.\\
    Expects len(img_pair_ks) == len(full_path_pvecs)-1
    """
    springgrads = []

    for i in range(1, len(full_path_pvecs)-1):
        img = full_path_pvecs[i]
        left_nbor = full_path_pvecs[i-1]
        right_nbor = full_path_pvecs[i+1]

        left_k = img_pair_ks[i-1]
        right_k = img_pair_ks[i]

        sprgrad_left = np.linalg.norm(img - left_nbor) * left_k
        sprgrad_right = np.linalg.norm(img - right_nbor) * right_k

        sprgrad = sprgrad_left - sprgrad_right
        springgrads.append(sprgrad)
    return np.array(springgrads)

def full_springgrads(full_path_pvecs, img_pair_ks, internals=[None]):
    """
    Calculates the full spring gradients. Returns vectors.\\
    Expects len(img_pair_ks) == len(full_path_pvecs)-1
    """
    springgrads = []

    for i in range(1, len(full_path_pvecs)-1):
        img = full_path_pvecs[i]
        left_nbor = full_path_pvecs[i-1]
        right_nbor = full_path_pvecs[i+1]

        left_k = img_pair_ks[i-1]
        right_k = img_pair_ks[i]

        sprgrad_left = (img - left_nbor) * left_k
        sprgrad_right = (img - right_nbor) * right_k
        sprgrad = sprgrad_left + sprgrad_right
        springgrads.append(sprgrad)
    return np.array(springgrads)

# implementation of the analytical position scheme and its helper functions
# -------------------------------------------------------------------------


def compute_ideal_posns(cur_ks, cur_posns):
    """
    This function computes positions for massless points 
    in a chain connected by ideal springs with varying spring 
    constants (the kvals), such that the net spring force acting 
    on them is zero.
    """
    cur_posns = np.array(cur_posns)
    cur_ks = np.array(cur_ks)
    assert(len(cur_posns) == len(cur_ks) + 1)
    n_images = len(cur_posns) - 2

    # construct the condition matrix
    M = build_spring_matrix(cur_ks)

    # solution vector
    y_vec = np.zeros(n_images)

    # the first and the last image are effected by the start and end position
    # This must be included in the solution vector
    y_vec[-1] = cur_ks[-1] * cur_posns[-1]
    y_vec[0] = cur_ks[0] * cur_posns[0] # cur_posns[0] != 0 when starting at CI

    # now, compute the ideal x positions -> solve linear equations
    ideal_posns = np.linalg.solve(M, y_vec)
    return ideal_posns

def build_spring_matrix(cur_ks):
    """
    From the spring constants this function builds the tridiagonal
    matrix which includes:
    - the sum of the neighboring spring constants for the main diagonal entrys
    with a positive sign
    - the spring constant between two neighboring images on the off-diagonal
    entry of the corresponding images (two times) with a negative sign

    returns: the symmetric Hessian for the spring forces
    """
    n_images = len(cur_ks) - 1
    M = np.zeros((n_images, n_images))
    cur_ks = np.array(cur_ks)

    main_diag = cur_ks[:n_images] + cur_ks[1:n_images+1]
    off_diag = -cur_ks[1:n_images]

    np.fill_diagonal(M, main_diag)
    np.fill_diagonal(M[1:,:], off_diag)
    np.fill_diagonal(M[:,1:], off_diag)

    return M


def image_distances(full_path_pvecs):
    """
    This function computes the '1D' position 
    of images along the neb path, by summing up the 
    euclidean distances between neighboring images.
    """
    distances = [0.0] # the first image is at the start of the chain
    distance = 0.0
    for i in range(1, len(full_path_pvecs)):
        delta = full_path_pvecs[i] - full_path_pvecs[i-1]
        distance += np.linalg.norm(delta)
        distances.append(distance)
    return np.array(distances)


def calc_analytic_position_deltas(full_path_pvecs, img_pair_ks, img_tanvecs):
    """
    This function solves for the 1D analytical positions of the 
    neb images, and produces a step vector for each image 
    to move it there in the actual coordinate space, by moving 
    them the appropriate amount along their tangent vectors
    """
    # first, we need to calculate the positions of the images along the neb path
    current_posns_1d = image_distances(full_path_pvecs)
    # now we solve the 1D system of coupled springs
    ideal_posns_1d = compute_ideal_posns(img_pair_ks, current_posns_1d)
    # now compute the difference between current (sans ends) and ideal
    delta_posns_1d = ideal_posns_1d - current_posns_1d[1:-1]
    # now compute the appropriate steps along the tangent vectors
    spring_steps = [tanvec * delta for tanvec, delta in
                    zip(img_tanvecs, delta_posns_1d)]
    return np.array(spring_steps)


def calc_analytic_springsteps(full_path_pvecs,
                              img_pair_ks,
                              img_tanvecs,
                              ci_index=None):                        
    """
    This function is supposed to be used to calculate 
    # analytical position scheme in the NEB. It can 
    # calculate them both for ci and non-ci calculations.
    """
    if ci_index is None:
        return calc_analytic_position_deltas(full_path_pvecs,
                                             img_pair_ks,
                                             img_tanvecs)
    else:
        # the climbing image cannot be moved by the spring forces of its neighbors. Ideal posns have to be
        # computed as two partial chains: left end to CI, then CI to right end
        abs_ci_ind = ci_index + 1
        l_pvecs = full_path_pvecs[:abs_ci_ind+1]
        r_pvecs = full_path_pvecs[abs_ci_ind:]
        # len(img_pair_ks) = len(full_path_pvecs)-1
        l_ks = img_pair_ks[:abs_ci_ind]
        r_ks = img_pair_ks[abs_ci_ind:]
        # len(img_tanvecs) = len(full_path_pvecs)-2
        l_tans = img_tanvecs[:abs_ci_ind-1]
        r_tans = img_tanvecs[abs_ci_ind:]
        l_steps = calc_analytic_position_deltas(l_pvecs, l_ks, l_tans)
        r_steps = calc_analytic_position_deltas(r_pvecs, r_ks, r_tans)

        # the step for the ci is consequently zero
        ci_step = np.zeros_like(full_path_pvecs[ci_index])
        all_steps = np.vstack((l_steps, ci_step))
        all_steps = np.concatenate((all_steps, r_steps))
        return all_steps


def compute_pairwise_ks(full_path_energies, kmax, kmin):
    """
    Method for calculating spring constants in variable k climbing image 
    it uses the improved variable k scheme.
    """
    if (full_path_energies == None).any():
        print('Variable k cannot be used this iteration due to failed' +
              ' images. Switching to regular k.')
        return np.zeros(len(full_path_energies) - 1) + kmax

    # Find the HEI and delta k
    HEI_index, E_max = np.argmax(full_path_energies), np.max(full_path_energies)
    delta_k = kmax - kmin

    pairwise_ks = []
    # find k for pair of images i, i+1
    for i in range(len(full_path_energies) - 1):
        # Get higher energy of the pair
        E_i = np.max(full_path_energies[i:i+2])
        e_index = np.argmax(full_path_energies[i:i+2]) + i

        # find E_ref. It depends on which side of the ci image you are on.
        left_end = full_path_energies[0]
        right_end = full_path_energies[-1]
        E_ref = left_end if e_index <= HEI_index else right_end

        if E_max <= E_ref:
            raise nex.NEBError('Climbing image is at or below the energy of' +
                               'one of the ends. Something went horribly ' +
                               'wrong.')

        fac = (E_max - E_i) / (E_max - E_ref)
        kval = kmax - delta_k * fac
        pairwise_ks.append(kval)
    return np.array(pairwise_ks)

