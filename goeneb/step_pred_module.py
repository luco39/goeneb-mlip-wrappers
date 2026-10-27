import numpy as np
import logging

logger = logging.getLogger(__name__)

import self_consistent_tangent_module as sct
from hessian_module import hessian


# ------------------------------------------------------------------------    
# Helper Functions

def enforce_maxstep(stepvec, max_step_norm):
    """
    This function rescales a step vector to the maximum step length if it is to big.
    """
    norm = np.linalg.norm(stepvec)
    if norm > max_step_norm:
        logger.debug('Norm of the suggested step is %s, step is scaled', norm)
        stepvec /= norm
        stepvec *= max_step_norm
        logger.debug('New norm: %s', np.linalg.norm(stepvec))
    else:
        logger.debug('Norm of the suggested step is %s', norm)
    return stepvec

def enforce_maxsteps(stepvecs, max_step_norm):
    """
    this function is meant to check the norm of a series
    of step vectors, and scale them down to meet the
    maximum step norm if they exceed it
    """
    for i in range(len(stepvecs)):
        stepvecs[i] = enforce_maxstep(stepvecs[i], max_step_norm)
    return stepvecs

def rms(x):
    return np.sqrt(np.mean(x**2))


# ------------------------------------------------------------------------    
# Steepest Descent

class SD:
    def __init__(self, stepsize_fac=0.2):
        self.stepsize_fac = stepsize_fac

    def reset(self):
        pass

    def predict(self, gradvec):
        logger.debug('Doing gradient descent')
        step = - self.stepsize_fac * gradvec
        return step

    def update(self, pvec):
        pass

# ------------------------------------------------------------------------    
# Adaptive Momentum Gradient Descent (by Björn)

class AMGD:
    def __init__(self, stepsize_fac=0.2, max_gamma=0.9):
        self.cart_pvecs = []
        self.prev_step = None
        self.stepsize_fac = stepsize_fac
        self.max_gamma = max_gamma

    def reset(self):
        self.cart_pvecs = []
        self.prev_step = None

    def predict(self, gradvec):
        if self.prev_step is None:
            # It's the first iteration, just do gradient descent
            logger.debug('Doing gradient descent')
            step = - self.stepsize_fac * gradvec

        else:
            # do full step calculation
            logger.debug('Doing AMGD')

            # determine angle
            gradnorm = np.linalg.norm(gradvec)
            prev_stepnorm = np.linalg.norm(self.prev_step)
            angle_denom = gradnorm * prev_stepnorm

            # determine r value
            if np.abs(angle_denom) < 1e-8:
                angle_rval = 0.5

            else:
                angle = np.dot(gradvec, self.prev_step) / angle_denom
                angle_rval = 0.5 + 0.5 * angle

            angle_rval = min(angle_rval, 1.0)
            angle_rval = max(1.0 - self.max_gamma, angle_rval)

            # compute optimization step
            step = (1.0 - angle_rval) * self.prev_step - self.stepsize_fac * gradvec
            logger.debug('Beta is %f', (1 - angle_rval))

        return step

    def update(self, cart_pvec):
        # save pvec copy for next iteration and 
        # translate last step to internals
        self.cart_pvecs.append(cart_pvec)
        if len(self.cart_pvecs) >= 2:
            self.prev_step = self.cart_pvecs[-1] - self.cart_pvecs[-2]


# ------------------------------------------------------------------------    
# Global BFGS with Newton-Raphson or Rational Function Optimization

class BFGS:
    def __init__(self, images=11, BFGS_start=5, NR_start=10, stepsize_fac=0.2, max_gamma=0.9):
        self.NR_start = NR_start
        self.BFGS_start = BFGS_start
        self.stepsize_fac = stepsize_fac
        self.max_gamma = max_gamma
        self.iteration = 0
        self.do_AMGD = True
        self.images = images

        # Setup AMGD
        self.AMGD_objects = [AMGD(stepsize_fac, max_gamma) for _ in range(images)]

    def predict(self, pvecs, gradvecs, energies, hessian):
        shape = np.array(pvecs).shape

        if len(shape) == 1:
            # only one structure given
            pvecs = np.array([pvecs])
            gradvecs = np.array([gradvecs])
            energies = np.array([energies])

        self.do_AMGD = False
        if None in energies or hessian.get_hessian() is None:
            self.do_AMGD = True

        if (self.iteration == 1) or (self.iteration < self.BFGS_start):
            # Its the first iteration(s)
            self.do_AMGD = True

        if (self.iteration < self.NR_start) or self.do_AMGD:
            # Do AMGD if it wasn't NR_start iterations yet or
            # a calculation failed in this or the last iteration

            AMGD_steps = [obj.predict(gradvec) for obj, gradvec in zip(self.AMGD_objects, gradvecs)]
            # zero step and reset for energy None
            return AMGD_steps

        else:
            gradvec = np.array(gradvecs).ravel()
            # Do a Step
            stepvec = self.step(hessian, gradvec) 
            return stepvec.reshape(shape)

    def step(self, hessian, gradvec):
        """
        Should be overwritten by child classes
        """
        pass

    def update(self, cart_pvec, cart_grad, energies, hessian:hessian):
        """
        Update function for the global hessian in the appropriate coordinate system.
        """
        self.iteration += 1
        # Update the memorized previous pvecs and gradvecs
        hessian.cart_pvecs.append(np.array(cart_pvec).ravel())
        hessian.cart_grads.append(np.array(cart_grad).ravel())

        if not (isinstance(energies, list) or isinstance(energies, np.ndarray)):
            energies = [energies]

        if None in energies:
            # The calculation failed
            # The image is gonna be moved over the PES by interpolation
            # better to reset hessian
            hessian.reset()
            pass

        elif (len(hessian.cart_pvecs) >= 2) and (self.iteration >= self.BFGS_start):
            # Its not the first iteration(s)
            # update the hessian objects
            hessian.update()

        # Also update the AMGD objects in cartesian coordinates
        self._update_AMGD(cart_pvec)

    def _update_AMGD(self, pvecs):
        for obj, pvec in zip(self.AMGD_objects, pvecs):
            obj.update(pvec)


class NewtonRaphson(BFGS):
    def step(self, hessian, gradvec):
        logger.info("Doing Newton-Raphson step")
        # Do a Newton-Raphson Step
        stepvec = - np.dot(hessian.get_inv_hessian(), gradvec) * self.stepsize_fac 
        return stepvec


class RationalFunction(BFGS):
    def step(self, hessian:hessian, gradvec):
        logger.info("Doing Rational Function step")
        lastrow = np.append(gradvec, 0)
        hessmat = hessian.get_hessian()
        aughess = np.vstack((np.column_stack((hessmat, gradvec)), lastrow))

        # Diagonalize the augmented hessian
        eigenvals, eigenvecs = np.linalg.eigh(aughess)

        # The step is now the eigenvector (last element scaled to 1) of the lowest eigenvalue
        lowestev = np.argmin(eigenvals)
        logger.debug('%s', eigenvals[lowestev])
        if eigenvals[lowestev] > 1.0E-4:
            raise ValueError('ERROR: I don\'t want to go up the PES!')
        stepvec = eigenvecs[:, lowestev]

        # Do a Rational Function Step
        if np.abs(stepvec[-1]) < 1e-8:
            scaledstepvec = stepvec[:-1]
            logger.warning('WARNING: Step should be devided by 0! No guarantee for the step size!')
        else:
            scaledstepvec = stepvec[:-1]/stepvec[-1]
            logger.debug('Scaling eigenvector of RFO matrix by last element: %s', stepvec[-1])
        logger.debug('Length of RFO step vector: %s', np.linalg.norm(stepvec[:-1]))
        return scaledstepvec


class LocalRF(RationalFunction):
    def update(self, cart_pvecs, cart_grads, energies, hessians):
        self.iteration += 1
        for pvec, grad, hessian in zip(cart_pvecs, cart_grads, hessians):

        # Update the memorized previous pvecs and gradvecs
            hessian.cart_pvecs.append(pvec)
            hessian.cart_grads.append(grad)

            if (len(hessian.cart_pvecs) >= 2) and (self.iteration >= self.BFGS_start):
                # Its not the first iteration(s)
                # update the hessian objects   
                hessian.update()

        # Also update the AMGD objects in cartesian coordinates
        self._update_AMGD(cart_pvecs)

    def predict(self, pvecs, gradvecs, energies, hessians):
        self.do_AMGD = False
        if None in energies:
            self.do_AMGD = True
        if (self.iteration == 0) or (self.iteration < self.BFGS_start):
            # Its the first iteration(s)
            self.do_AMGD = True
        if (self.iteration < self.NR_start) or self.do_AMGD:
            # Do AMGD if it wasn't NR_start iterations yet or
            # a calculation failed in this or the last iteration
            AMGD_steps = [obj.predict(gradvec) for obj, gradvec in zip(self.AMGD_objects, gradvecs)]
            # zero step and reset for energy None
            return AMGD_steps
        else:
            # Do a Step
            stepvecs = [self.step(hessian, gradvec) for hessian, gradvec in zip(hessians, gradvecs)]
            return stepvecs


class LocalNR(NewtonRaphson):
    def update(self, cart_pvecs, cart_grads, energies, hessians):
        self.iteration += 1
        for pvec, grad, hessian in zip(cart_pvecs, cart_grads, hessians):
        # Update the memorized previous pvecs and gradvecs
            hessian.cart_pvecs.append(pvec)
            hessian.cart_grads.append(grad)

            if (len(hessian.cart_pvecs) >= 2) and (self.iteration >= self.BFGS_start):
                # Its not the first iteration(s)
                # update the hessian objects   
                hessian.update()
        # Also update the AMGD objects in cartesian coordinates
        self._update_AMGD(cart_pvecs)

    def predict(self, pvecs, gradvecs, energies, hessians):
        self.do_AMGD = False
        if None in energies:
            self.do_AMGD = True
        if (self.iteration == 0) or (self.iteration < self.BFGS_start):
            # Its the first iteration(s)
            self.do_AMGD = True
        if (self.iteration < self.NR_start) or self.do_AMGD:
            # Do AMGD if it wasn't NR_start iterations yet or
            # a calculation failed in this or the last iteration
            AMGD_steps = [obj.predict(gradvec) for obj, gradvec in zip(self.AMGD_objects, gradvecs)]
            # zero step and reset for energy None
            return AMGD_steps
        else:
            # Do a Step
            stepvecs = [self.step(hessian, gradvec) for hessian, gradvec in zip(hessians, gradvecs)]
            return stepvecs


# ------------------------------------------------------------------------
# Self consistent tangent method

class self_consistent_tangents:
    def __init__(self, images=11, BFGS_start=5, NR_start=10, stepsize_fac=0.2, max_gamma=0.9, soft_reset=True, soft_reset_memory=5):
        self.NR_start = NR_start
        self.BFGS_start = BFGS_start
        self.stepsize_fac = stepsize_fac
        self.max_gamma = max_gamma
        self.iteration = 0
        self.do_AMGD = True
        self.images = images
        self.failed_sct_count = 0
        self.soft_reset = soft_reset
        self.soft_reset_memory = soft_reset_memory

        # Setup AMGD
        self.AMGD_objects = [AMGD(stepsize_fac, max_gamma) for _ in range(images)]

    def predict(self, nebgradvecs, hessians, energies, settings=None, path=None):
        self.do_AMGD = False
        if None in energies or None in hessians:
            self.do_AMGD = True
        if (self.iteration == 0) or (self.iteration < self.BFGS_start):
            # Its the first iteration(s)
            self.do_AMGD = True
        if False in [len(hessian.cart_pvecs) >= 2 for hessian in hessians]:
            # Last iteration was a failed calculation
            self.do_AMGD = True
        # Prepare AMGD - WITH NEB-GRADIENTS!
        AMGD_steps = [obj.predict(gradvec) for obj, gradvec 
                      in zip(self.AMGD_objects, nebgradvecs)]

        # if it wasn't NR_start iterations yet or
        # a calculation failed in this or the last iteration
        if (self.iteration < self.NR_start) or self.do_AMGD:
            return AMGD_steps

        else:
            # do harmonic NEB, but only if the optimization didn't fail
            # With real energy gradient
            SCT_steps, ret_state = sct.do_harmonic_opt(path, settings, hessians)
            if ret_state == 'FAILED':
                logger.warning('Doing AMGD, since the SCT did not converge')
                self.failed_sct_count += 1
                return AMGD_steps
            elif SCT_steps is not None:
                logger.info('Doing SCT-NEB')
                return SCT_steps
            else:
                logger.info('Harmonic NEB already converged, doing scaled AMGD')
                return np.array(AMGD_steps) * 0.1

    def update(self, cart_pvecs, cart_grads, energies, hessians):
        """
        Update function for the local hessians of the images.
        """
        self.iteration += 1
        # soft reset if SCT did not converge in the last 5 iterations
        if self.failed_sct_count >= 5 and self.soft_reset:
            for hessian in hessians:
                hessian.soft_reset(self.soft_reset_memory)
            self.failed_sct_count = 0

        for pvec, grad, energy, hessian in zip(cart_pvecs, cart_grads, energies, hessians):
            # Update the memorized previous pvecs and gradvecs
            hessian.cart_pvecs.append(pvec)
            hessian.cart_grads.append(grad)
            if energy is None:
                # The calculation failed The image is gonna be moved over 
                # the PES by interpolation better to reset hessian
                hessian.reset()
            elif (len(hessian.cart_pvecs) >= 2) and (self.iteration >= self.BFGS_start):
                # Its not the first iteration(s), update the hessian objects
                hessian.update()
        # Update AMGD objects also
        self._update_AMGD(cart_pvecs)

    def _update_AMGD(self, pvecs):
        for obj, pvec in zip(self.AMGD_objects, pvecs):
            obj.update(pvec)