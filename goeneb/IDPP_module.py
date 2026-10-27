import numpy as np
import sys
import logging
import warnings

from neb_path import NEBPath
from basic_neb import BasicNEB
import springforce_module as sfm
import tangent_module as tgm
import nebgrad_module as ngm
import convsig_module as cm
import step_pred_module as spm

from helper import calculate_distmat, interpolate_linear

if not sys.warnoptions:
    warnings.simplefilter("ignore")

logger = logging.getLogger(__name__)

# Functions for calculating the IDPP S value (stand-in for energy),
# and its gradient, used for structure preoptimization in IDPP.
# ------------------------------------------------------------------

def calc_IDPP_engrads(img_pvecs, ideal_dmats, w_exp=4.0):
    """Calculate the IDPP S value (stand in for energy) and its gradient.
    expects the pvecs (M,3N,), the ideal distance matrices (M,N,N) and the 
    exponent for the weighting factor omega.
    """
    gradvecs = [IDPP_gradS(pvec, ideal_dmat, w_exp=w_exp)
                for pvec, ideal_dmat in zip(img_pvecs, ideal_dmats)]

    energies = [IDPP_S(pvec, ideal_dmat, w_exp=w_exp)
                for pvec, ideal_dmat in zip(img_pvecs, ideal_dmats)]

    return np.array(energies), np.array(gradvecs)

def IDPP_S(pvec, ideal_dmat, w_exp=4.0):
    """
    IDPP energy calculation. Expects the pvec in shape (3N,) and ideal_dmat
    in shape (N,N).
    """
    # Weighting factors omega are 1/d^4 of the real distances
    dist_mat = calculate_distmat(pvec)
    np.fill_diagonal(dist_mat, 1) # avoid division by 0
    omega = np.power(dist_mat, -w_exp)

    # distance of atom to itself should be 0
    np.fill_diagonal(omega, 0)

    delta_mat = ideal_dmat - dist_mat
    s_vals = omega.flatten() * delta_mat.flatten()**2

    # The sum counts all atom combinations twice
    S =  np.sum(s_vals) * 0.5
    return S

def IDPP_gradS(pvec, ideal_dmat, w_exp=4.0):
    """
    IDPP gradient calculation. Expects the pvec in shape (3N,) and ideal_dmat
    in shape (N,N).
    """
    # Get the real distance matrix and the connecting vectors
    dist_mat, rvec_mat = calculate_distmat(pvec, rvec=True)

    # This is d_real - d_ideal in a (N,N) matrix
    delta_mat = dist_mat - ideal_dmat

    # avoid division by 0
    np.fill_diagonal(dist_mat, 1)

    # The IDPP gradient is calculated by multiplying the 
    # factor with a new axis (N,N,1) with the connecting vectors (N,N,3)
    factor = -2 * delta_mat * np.power(dist_mat, -w_exp-1) + 4 * np.power(delta_mat, 2) * np.power(dist_mat, -w_exp-2)
    grad_vec = -np.sum(factor[:,:,np.newaxis] * rvec_mat, axis=0).flatten()

    return grad_vec

# The following is an implementation of an IDPP preoptimizer
# function for NEB starting paths. It uses the BasicNEB class to perform its own NEB optimization
# using the IDPP gradient, as described in the paper introducing IDPP.
# Smidstrup, S.; Pedersen, A.; Stokbro, K.; Jónsson, H. Improved Initial Guess 
# for Minimum Energy Path Calculations. J. Chem. Phys. 2014, 140 (21),
# 214106. https://doi.org/10.1063/1.4878664.
# ----------------------------------------------------------------------

class IDPP(BasicNEB):
    def __init__(self, NEBPath:NEBPath, settings=None):
        super().__init__(NEBPath, settings)

    def calc_springgrads(self):
        all_img_pvecs = self.path.get_img_pvecs(include_ends=True)
        tanvecs = self.path.get_tanvecs()

        full_springgrads = sfm.full_springgrads(all_img_pvecs, self.path.img_pair_ks)
        springgrads = [ngm.project(full_springgrad, tanvec) 
                    for full_springgrad, tanvec in zip(full_springgrads, tanvecs)]
        return springgrads

    def calc_tanvecs(self):
        all_img_pvecs = self.path.get_img_pvecs(include_ends=True)
        return tgm.simple_tans(all_img_pvecs)

    def calc_nebgrads(self):
        engrads = self.path.get_engrads()
        sprgrads = self.path.get_springgrads()
        tanvecs = self.path.get_tanvecs()

        nebgrads = ngm.calculate_nebgrads(engrads,
                                        sprgrads,
                                        tanvecs)

        # project out translations to make sure
        nebgrads = ngm.sanitize_stepvecs(nebgrads,
                                    reject_transl=True,
                                    reject_rot=False)

        # zero the gradients corresponding to frozen atoms,
        # so they don't skew the signals of convergence thresholds
        for i in range(len(nebgrads)):
            nebgrads[i] = ngm.freeze_atom_indices(nebgrads[i], self.frozen_atom_indices)

        orth_grads = [ngm.reject(nebgrad, tanvec) for nebgrad, tanvec in zip(nebgrads, tanvecs)]
        return nebgrads, orth_grads

    def predict_steps(self):
        # there should be no failed images in IDPP, so we do not need to take any precautions
        # for the case of empty nebgrads of failed images
        nebgrads = self.path.get_nebgrads()

        # for some reason, it seems gradient descent performs better
        steps = [- self.stepsize * nebgrad for nebgrad in nebgrads]

        # Enforce maxstep
        if self.max_step is not None:
            steps = spm.enforce_maxsteps(steps, self.max_step)
        return steps

    def conv_checker_func(self, steps):
        IDPP_neb_grads = self.path.get_nebgrads()

        rmsf = cm.NEB_RMSF(IDPP_neb_grads)
        absf = cm.NEB_ABSF(IDPP_neb_grads)

        logger.debug('IDPP Max. RMSF: %f Tol.: %f', rmsf, self.IDPP_max_RMSF)
        logger.debug('IDPP Max. Abs. F: %f Tol.: %f', absf, self.IDPP_max_AbsF)

        if rmsf > self.IDPP_max_RMSF:
            return False
        if absf > self.IDPP_max_AbsF:
            return False
        return True
    
class S_IDPP(IDPP):
    def __init__(self, NEBPath:NEBPath, settings=None):
        super().__init__(NEBPath, settings)

    def initialize_path(self, engrad_calc_func, engrad_calc_kwargs={}):
        """
        Set up a sequential path according to the SIDPP algorithm.

        Schmerwitz, Y. L. A.; Ásgeirsson, V.; Jónsson, H. Improved Initialization of Optimal Path Calculations 
        Using Sequential Traversal over the Image-Dependent Pair Potential Surface. J. Chem. Theory Comput. 2024, 20 (1), 
        155-163. https://doi.org/10.1021/acs.jctc.3c01111.
        """
        self.left_end = 1
        self.right_end = self.n_images
        self.ideal_dmats = engrad_calc_kwargs['ideal_dmats']
        while self.images < self.n_images and self.iteration <= self.maxiter:
            self.iteration += 1
            logger.debug(f"\nIteration {self.iteration}")

            # calculate engrads (only select relevant dmats)
            engrad_calc_kwargs['ideal_dmats'] = np.vstack([self.ideal_dmats[:self.left_end], 
                                                           self.ideal_dmats[self.right_end-1:]])
            energies, engrads = engrad_calc_func(self.path.get_img_pvecs(),
                                                 **engrad_calc_kwargs)
            self.path.set_energies(energies)
            self.path.set_engrads(engrads)

            # calculate tangents and springs
            tanvecs = self.calc_tanvecs()
            self.path.set_tanvecs(tanvecs)

            self.recalculate_img_pair_ks()
            springgrads = self.calc_springgrads()
            self.path.set_springgrads(springgrads)

            # calculate NEB grads
            nebgrads, orth_grads = self.calc_nebgrads()
            self.path.set_nebgrads(nebgrads)
            self.path.set_orth_grads(orth_grads)

            # calculate and apply steps
            steps = self.predict_steps()
            self.path.set_img_pvecs([pvec + step for pvec, step 
                                     in zip(self.path.get_img_pvecs(), steps)])

            # Add images
            path_pvecs = self.path.get_img_pvecs(include_ends=True)
            images_to_add = self.right_end - self.left_end - 1
            logger.debug(f"Images to still add: {images_to_add}")
            interpolated_ends = interpolate_linear(path_pvecs[self.left_end],
                                                 path_pvecs[self.left_end + 1],
                                                 images_to_add)

            if self.conv_check_image(self.left_end + 1) :
                # add right image
                path_pvecs = self.path.get_img_pvecs(include_ends=True)
                left_part = path_pvecs[:self.left_end+1]
                first_interp = interpolated_ends[-2]
                right_part = path_pvecs[self.left_end+1:]
                new_path = np.vstack([left_part, first_interp, right_part])
                self.path.set_img_pvecs(new_path[1:-1])
                self.right_end -= 1
                logger.debug('Image added to the right.')

            if self.conv_check_image(self.left_end) and self.images < self.n_images:
                # add left image
                path_pvecs = self.path.get_img_pvecs(include_ends=True)
                left_part = path_pvecs[:self.left_end+1]
                first_interp = interpolated_ends[1]
                right_part = path_pvecs[self.left_end+1:]
                new_path = np.vstack([left_part, first_interp, right_part])
                self.path.set_img_pvecs(new_path[1:-1])
                self.left_end += 1
                logger.debug('Image added to the left.')
        # The SIDPP did not add all images
        if self.images < self.n_images:
            images_to_add = self.n_images - self.images
            logger.warning("The SIDPP did not converge correctly. %i images are still missing.", images_to_add)
            logger.warning("These images are intepolated linear.")
            interpolated_path = interpolate_linear(path_pvecs[self.left_end],
                                                 path_pvecs[self.left_end + 1],
                                                 images_to_add)
            path_pvecs = self.path.get_img_pvecs(include_ends=True)
            left_part = path_pvecs[:self.left_end+1]
            right_part = path_pvecs[self.left_end+1:]
            new_path = np.vstack([left_part, interpolated_path[1:-1], right_part])
            self.path.set_img_pvecs(new_path[1:-1])
        # complete ideal_dmats and image_pair_ks before returning
        engrad_calc_kwargs['ideal_dmats'] = self.ideal_dmats
        self.path.set_img_pair_ks(np.zeros(self.n_images + 1) + self.k_const)
        return self.path, self.iteration

    def recalculate_img_pair_ks(self):
        """
        Calculate the spring constants for each pair of images
        Includes a lower spring constant for the distance between the two border images
        based on the distance between the ends
        """
        path_pvecs = self.path.get_img_pvecs(include_ends=True)
        path_length = self.path.get_path_length()
        dist_ideal = path_length / (self.n_images + 1)
        dist_between_ends = np.linalg.norm(path_pvecs[self.left_end] 
                                           - path_pvecs[self.left_end + 1])
        new_k = (dist_ideal / dist_between_ends) * self.k_const
        if new_k == np.inf:
            new_k = 1
        new_ks = np.zeros(len(path_pvecs) - 1) + self.k_const
        new_ks[self.left_end] = new_k
        logger.debug(f"The new spring constant is {new_k}")
        self.path.set_img_pair_ks(new_ks)

    def conv_check_image(self, img_index):
        IDPP_neb_grads = self.path.get_nebgrads()

        rmsf_image = cm.NEB_RMSF(IDPP_neb_grads[img_index-1])
        absf_image = cm.NEB_ABSF(IDPP_neb_grads[img_index-1])
        logger.debug(f"Image {img_index}: RMSF: {rmsf_image}, AbsF: {absf_image}")

        # Thresholds 10 times more relaxed than normal IDPP
        if rmsf_image > self.IDPP_max_RMSF * 10:
            return False
        if absf_image > self.IDPP_max_AbsF * 10:
            return False
        return True
    
    def calc_tanvecs(self):
        all_img_pvecs = self.path.get_img_pvecs(include_ends=True)
        energies = self.path.get_energies(include_ends=True)
        return tgm.henkjon_tans(all_img_pvecs, energies)


# The following is the function for optimizing NEB starting paths
# one must be careful to set the IDPP convergence thresholds not too tight.
# ----------------------------------------------------------------------

def do_IDPP_opt_pass(path_pvecs,
                     settings=None):
    # All the info needed to be passed on
    idpp_settings = settings.get_idpp_settings()

    # calculate 'ideal' distance matrices
    start_dmat = calculate_distmat(path_pvecs[0])
    stop_dmat = calculate_distmat(path_pvecs[-1])

    # we want len(path_pvecs)-2 interpolations, because path_pvecs contains the ends,
    # which arent counted in interpolate_linear
    ideal_dmats = interpolate_linear(start_dmat,
                                     stop_dmat,
                                     len(path_pvecs)-2)

    # set up NEBPath object for the neb optimizer function
    # separate distmats of the ends from the rest of the path
    start_dmat = ideal_dmats[0]
    img_dmats = ideal_dmats[1:-1]
    end_dmat = ideal_dmats[-1]

    # atomic labels dont matter in IDPP
    labels = []
    
    # Energy function
    engrad_calc_func = calc_IDPP_engrads
    engrad_calc_kwargs = {'ideal_dmats' : [start_dmat, end_dmat],
                          'w_exp' : 4.0}

    # Now, we have collected all data needed to generate
    # the NEBPath object that can be fed into the NEB optimizer
    # loop function from the neb_optimizer module.
    IDPP_Path = NEBPath(labels,
                        path_pvecs,
                        engrad_calc_func,
                        engrad_calc_kwargs,
                        idpp_settings)

    # feed all the collected variables into the NEB optimizer function
    # to perform the IDPP optimization
    # New energy keywords are needed for the images
    engrad_calc_kwargs = {'ideal_dmats' : img_dmats,
                          'w_exp' : 4.0}
    optimizer = IDPP(IDPP_Path, idpp_settings)
    IDPP_Path, return_state, iterations = optimizer.do_opt_loop(engrad_calc_func, engrad_calc_kwargs, silent_mode=True)

    if return_state == 'FAILED':
        logger.warning('Warning: IDPP did not converge! Be careful with the results! ' +
                       'Study the messages above to find the cause.')

    else:
        logger.info('IDPP pass is converged after %d iterations!\n', iterations)

    # recover the optimized geometries from the NEBPath object,
    # in the form of the 1D position vectors, and return them,
    opt_path_pvecs = IDPP_Path.get_img_pvecs(include_ends=True)
    return opt_path_pvecs

def do_SIDPP_opt_pass(start_pvec,
                      end_pvec,
                      settings=None):
    """
    This function starts the SIDPP algorithm and builds an initial path
    with the given parameters. However during the SIDPP the convergence thresholds
    are relaxed by factor 10. Does SIDPP and then IDPP with normal convergence threshholds.
    """
    # All the info needed to be passed on
    idpp_settings = settings.get_idpp_settings()

    # calculate 'ideal' distance matrices
    start_dmat = calculate_distmat(start_pvec)
    stop_dmat = calculate_distmat(end_pvec)

    # we want len(path_pvecs)-2 interpolations, because path_pvecs contains the ends,
    # which arent counted in interpolate_linear
    ideal_dmats = interpolate_linear(start_dmat,
                                     stop_dmat,
                                     idpp_settings.n_images)

    # set up NEBPath object for the neb optimizer function
    # separate distmats of the ends from the rest of the path
    start_dmat = ideal_dmats[0]
    img_dmats = ideal_dmats[1:-1]
    end_dmat = ideal_dmats[-1]

    # atomic labels dont matter in IDPP
    labels = []
    
    # Energy function
    engrad_calc_func = calc_IDPP_engrads
    engrad_calc_kwargs = {'ideal_dmats' : [start_dmat, end_dmat],
                          'w_exp' : 4.0}

    # Now, we have collected all data needed to generate
    # the NEBPath object that can be fed into the NEB optimizer
    # loop function from the neb_optimizer module.
    cart_interpolation = interpolate_linear(start_pvec, end_pvec, idpp_settings.n_images)
    starting_path_pvecs = np.vstack([cart_interpolation[:2], cart_interpolation[-2:]])
    S_IDPP_Path = NEBPath(labels, 
                          starting_path_pvecs, 
                          engrad_calc_func, 
                          engrad_calc_kwargs, 
                          idpp_settings)

    # feed all the collected variables into the NEB optimizer function
    # to perform the IDPP optimization
    # New energy keywords are needed for the images
    engrad_calc_kwargs = {'ideal_dmats' : img_dmats,
                          'w_exp' : 4.0}
    optimizer = S_IDPP(S_IDPP_Path, idpp_settings)
    S_IDPP_Path, iterations = optimizer.initialize_path(engrad_calc_func, engrad_calc_kwargs)
    logger.info('SIDPP path completed after %i iterations.', iterations)

    # normal IDPP
    IDPP_Path, return_state, iterations = optimizer.do_opt_loop(engrad_calc_func, engrad_calc_kwargs, silent_mode=True)

    if return_state == 'FAILED':
        logger.warning('Warning: IDPP did not converge! Be careful with the results! ' +
                       'Study the messages above to find the cause.')

    else:
        logger.info('IDPP pass is converged after %d iterations!\n', iterations)

    # recover the optimized geometries from the NEBPath object,
    # in the form of the 1D position vectors, and return them,
    opt_path_pvecs = IDPP_Path.get_img_pvecs(include_ends=True)
    return opt_path_pvecs
