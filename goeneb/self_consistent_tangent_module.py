import numpy as np
import logging
import copy

from neb_path import NEBPath
import neb_optimizer as nopt

logger = logging.getLogger(__name__)

def do_harmonic_opt(NEBPath:NEBPath,
                    settings,
                    hessians):
    """
    Do the NEB optimization on the harmonic approximation in the SCT method. 
    Requires the hessians for all images.
    """
    # gather information for NEBPath object
    new_Path = copy.deepcopy(NEBPath)
    optimizer = nopt.SCT_Optimizer(new_Path, settings)

    img_pvecs = new_Path.get_img_pvecs(include_ends=False)
    gradvecs = new_Path.get_engrads()
    energies = new_Path.get_energies(include_ends=False)

    engrad_calc_func = calc_harmonic_engrads
    engrad_calc_kwargs = {'img_pvecs' : img_pvecs,
                          'img_grads' : gradvecs,
                          'hessians' : hessians,
                          'energies' : energies}

    new_Path, ret_state, iterations = optimizer.do_opt_loop(engrad_calc_func, engrad_calc_kwargs, silent_mode=True)

    if ret_state == 'FAILED':
        logger.warning('Warning: Harmonic NEB did not converge!')
    else:
        logger.info('Harmonic NEB is converged after %d iterations!', iterations)

    opt_path_pvecs = new_Path.get_img_pvecs(include_ends=False)
    steps = [opt_pvec - img_pvec for opt_pvec, img_pvec in zip(opt_path_pvecs, img_pvecs)]
    if iterations == 1:
        steps = None

    return steps, ret_state


# Functions for calculating harmonic energy and gradient
# --------------------------------------------------------------------------------

def calc_harmonic_engrads(pvecs, img_pvecs, img_grads, hessians, energies):
    """
    Calculate the harmonic energy and gradients based on the hessians of each image.
    """
    new_energies = [calc_harmonic_energy(pvec, pvec_i, grad_i, hessian.get_hessian(), energy) 
                for pvec, pvec_i, grad_i, hessian, energy
                in zip(pvecs, img_pvecs, img_grads, hessians, energies)]

    new_gradvecs = [calc_harmonic_gradient(pvec, pvec_i, grad_i, hessian.get_hessian()) 
                for pvec, pvec_i, grad_i, hessian 
                in zip(pvecs, img_pvecs, img_grads, hessians)]
    logger.debug('New energies: %s', new_energies)
    return new_energies, new_gradvecs

def calc_harmonic_energy(pvec,
                         pvec_i,
                         grad_i,
                         hessian,
                         energy_i):
    energy = (0.5 * np.dot((pvec - pvec_i),
              np.dot((hessian), (pvec - pvec_i))) 
              + np.dot(grad_i, (pvec - pvec_i)) + energy_i)
    return energy

def calc_harmonic_gradient(pvec,
                           pvec_i,
                           grad_i,
                           hessian):
    gradvec = np.dot(hessian, (pvec - pvec_i)) + grad_i
    return gradvec