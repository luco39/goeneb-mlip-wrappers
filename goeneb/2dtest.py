import numpy as np
import matplotlib.pyplot as plt
import logging
import os
import matplotlib.colors as mcolors
from scipy.linalg import block_diag
import argparse

import nebgrad_module as ngr
import convsig_module as con 
from neb_path import NEBPath
from neb_optimizer import NEB_Optimizer
from logging_module import setup_logger
from neb_configparse import factory_settings as config_dict
from helper import interpolate_linear


def update_config_from_args(config_dict):
    parser = argparse.ArgumentParser(description="Set 2D NEB optimizer parameters.")

    parser.add_argument('-s', '--stepsize_fac', type=float, default=0.005,
                        help='Step size factor')
    parser.add_argument('-m', '--step_pred_method', type=str, default='AMGD',
                        choices=['AMGD', 'SD', 'RFO', 'NR', 'SCT', 'L-NR','L-RFO'],
                        help='Step predictor method')
    parser.add_argument('-k', '--k_const', type=float, default=0.003,
                        help='Spring constant')
    parser.add_argument('-t', '--tangents', type=str, default='henkjon',
                        choices=['henkjon', 'simple'],
                        help='Tangents calculation method')
    parser.add_argument('-i', '--maxiter', type=int, default=20,
                        help='Maximum number of iterations')
    parser.add_argument('-l', '--max_step', type=float, default=1,
                        help='Maximum step length')
    parser.add_argument('-n', '--NR_start', type=int, default=10,
                        help='Iteration when to start Newton-Raphson')
    parser.add_argument('-b', '--BFGS_start', type=int, default=5,
                        help='Iteration when to start BFGS')
    parser.add_argument('-d', '--draw_every_iteration', action='store_true',
                        help='Draw after every optimization iteration')
    parser.add_argument('-H', '--show_hessian', action='store_true',
                        help='Show hessian matrix')
    parser.add_argument('-c', '--caption', type=str, default=None,
                        help='Title of the output graph')
    parser.add_argument('-f', '--filename', type=str, default=None,
                        help='Filename of the produced image')
    parser.add_argument('-ff', '--fileformat', type=str, default='pdf',
                        choices=['pdf', 'png', 'svg', 'jpg', 'jpeg'],
                        help='File format of the produced image')
    args = parser.parse_args()

    # Update config_dict only with user-settable arguments
    for key in ['stepsize_fac', 'step_pred_method', 'k_const', 'tangents', 'maxiter',
                'max_step', 'NR_start', 'BFGS_start', 'draw_every_iteration', 'show_hessian', 'filename', 'fileformat', 'caption']:
        config_dict[key] = getattr(args, key)


class TwoD_Optimizer(NEB_Optimizer):
    def __init__(self, nebpath, dict={}):
        super().__init__(nebpath, dict)
        self.atoms = 2/3
        self.step_collection = []

    def calc_nebgrads(self):
        """
        Overwrite real NEB gradient function. Dont have translations and rotations here
        """
        engrads = self.path.get_engrads()
        springgrads = self.path.get_springgrads()
        tangents = self.path.get_tanvecs()

        nebgrads = ngr.calculate_nebgrads(engrads,
                                        springgrads,
                                        tangents,
                                        ci_index=None)

        orth_grads = [ngr.reject(nebgrad, tanvec) 
                    for nebgrad, tanvec in zip(nebgrads, tangents)]

        return nebgrads, orth_grads

    def do_iter_printout(self):
        """
        Helper function for printing some NEB stats to the console
        """
        logger = logging.getLogger(__name__)
        energies = self.path.get_energies(include_ends=True)
        energies_np = np.array(energies, dtype=float)  # forces None -> np.nan
        max_energy = np.nanmax(energies_np)

        left_barrier_kJmol = (max_energy - energies[0])
        right_barrier_kJmol = (max_energy - energies[-1])

        logger.info('There are %d failed images in the path.',  self.path.n_failed_images())

        ci_index = self.current_CI_index
        if ci_index is not None:
            logger.info('Image at index %d is now the climbing image.', ci_index)

        logger.info('Approx. Barrier with respect to left end: %f kJ/mol', left_barrier_kJmol)
        logger.info('Approx. Barrier with respect to right end: %f kJ/mol', right_barrier_kJmol)

    def conv_checker_func(self, steps):
        """
        This function should check for signals of convergence,
        and return True if NEB convergence has been reached, False otherwise.
        It should also take care of logging information about
        the progress of the NEB optimization, if that is desired.
        Here, we wont do any logging, but will do the visualization
        for the current iteration here.
        """
        logger = logging.getLogger(__name__)
        # now perform the check for the signals of convergence
        nebgrads_o = self.path.get_orth_grads()
        nebgrads = self.path.get_nebgrads()
        max_rmsf_o = con.NEB_RMSF(nebgrads_o)
        max_absf_o = con.NEB_ABSF(nebgrads_o)
        max_rmsf = con.NEB_RMSF(nebgrads)
        max_absf = con.NEB_ABSF(nebgrads)

        # compile rmsf, absf etc. and put them into the log
        values_dict = {'RMSF' : max_rmsf,
                       'AbsF' : max_absf,
                       'RMSFo': max_rmsf_o,
                       'AbsFo': max_absf_o}

        if self.logger is not None:
            self.logger.write_to_log(self.path, values_dict)

        Max_RMSF_tol = self.max_rmsf
        Max_AbsF_tol = self.max_absf

        logger.info('%-30s %f Tol. %f, %s', 'Current max. RMSF:', max_rmsf, Max_RMSF_tol, con.yesno(max_rmsf<Max_RMSF_tol))
        logger.info('%-30s %f Tol. %f, %s', 'Current max. AbsF:', max_absf, Max_AbsF_tol, con.yesno(max_absf<Max_AbsF_tol))

        logger.info('%-30s %f Tol. %f', 'Current max. orthogonal RMSF:', max_rmsf_o, Max_RMSF_tol)
        logger.info('%-30s %f Tol. %f', 'Current max. orthogonal AbsF:', max_absf_o, Max_AbsF_tol)

        is_converged = ((max_rmsf <= Max_RMSF_tol) and (max_absf <= Max_AbsF_tol))
        self.do_iter_printout()
        self.draw_output(is_converged)

        # now return result
        return is_converged

    def draw_output(self, is_converged):
        """
        This function draws the model surface and the images as 
        well as their path across the PES. Handles when to draw this and when
        to skip drawing.
        This function also draws the hessian if one exists and the user chose to draw the
        hessian.
        """
        # do the visualization
        self.step_collection.append(self.path.get_img_pvecs())

        if self.draw_every_iteration or is_converged or (self.iteration==self.maxiter):
            # draw the hessian for global bfgs
            if self.hessian is not None and self.hessian.get_hessian() is not None:
                hessian = self.hessian.get_hessian()
            # and for local
            elif self.hessians[0] is not None and self.hessians[0].get_inv_hessian() is not None:
                hessian = block_diag(*[obj.get_hessian() for obj in self.hessians])
            else:
                hessian = None
            if hessian is not None and self.show_hessian:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,4))
                ax2.matshow(hessian)
                ax2.set_xticks([])
                ax2.set_yticks([])
            # only one plot when no hessian
            else:
                fig, ax1 = plt.subplots(figsize=(4,4))
                if self.caption is not None:
                    ax1.set_title(self.caption, loc='left')

            draw_func(xmin=-4, xmax=4, ymin=-4, ymax=4, res=1000, ax=ax1, n_blocks=10)
            full_path_pvecs = self.path.get_img_pvecs(include_ends=True)
            draw_path(full_path_pvecs, ax=ax1)
            draw_trace(self.step_collection, ax=ax1)
            ax1.set_xlim(-4,4)
            ax1.set_ylim(-4,4)
            plt.tight_layout()
            if self.filename is not None:
                filename = self.filename + '.' + self.fileformat
                filepath = os.path.join(self.tmpdir, filename)
                plt.savefig(filepath)
            plt.show()


def main():
    workdir = os.getcwd()
    tmpdir = os.path.join(workdir, '2dtest')
    os.makedirs(tmpdir, exist_ok=True)
    logger = setup_logger(tmpdir)

    update_config_from_args(config_dict)
    config_dict['logfile_path'] = os.path.join(tmpdir, 'optlog.csv')
    config_dict['tmpdir'] = tmpdir

    # first, we have to gather the two minima,
    # and generate a starting path via interpolation.
    min1 = np.array([-2.805118, 3.131312])
    min2 = np.array([3.584428, -1.848126])
    path_pvecs = interpolate_linear(min1, min2, 11)

    # set up the neb path object that will store images,
    # engrads, tanvecs etc.
    nebpath = NEBPath(labels=[],
                      starting_path=path_pvecs,
                      engrfunc=calc_hblau_engrads,
                      engrfunc_kwargs={},
                      dict=config_dict)

    # perform the NEB optimization
    optimizer = TwoD_Optimizer(nebpath, config_dict)
    opt_nebpath, return_state, iterations =\
        optimizer.do_opt_loop(engrad_calc_func=calc_hblau_engrads,
                              engrad_calc_kwargs={})

    logger.info(return_state)
    logger.info('Ended at iteration: %i', iterations)
    print(f'Ended at iteration: {iterations}')
    logger.info(opt_nebpath.get_img_pvecs(include_ends=True))


# Helper functions for Himmelblaus function
# --------------------------------------------------

def calc_hblau_engrads(img_pvecs):
    """
    This function calculates the energies and gradients of the images
    according to Himmelblaus function.
    """
    energies = [hblau(img) for img in img_pvecs]
    gradients = [hblau_grad(img) for img in img_pvecs]
    return np.array(energies), np.array(gradients)


def hblau(pvec):
    """
    The 'energy' calculated with Himmelblaus function.
    """
    [x, y] = pvec
    part1 = x*x + y - 11.0
    part2 = x + y*y - 7.0
    return part1**2 + part2**2


def numgrad(func, x, numstep=1e-8, func_kwargs={}):
    """
    Numerically computes the gradient for arbitrary
    scalar function 'func'.
    """
    gradvals = []
    for i in range(len(x)):
        # step forward
        new_x = x.copy()
        new_x[i] += numstep
        val_fwd = func(new_x, **func_kwargs)

        #step backward
        new_x = x.copy()
        new_x[i] -= numstep
        val_bwd = func(new_x, **func_kwargs)
        grad = (val_fwd - val_bwd) / (2.0 * numstep)
        gradvals.append(grad)
    return np.array(gradvals)


def hblau_grad(pvec):
    """
    The gradient of Himmelblaus function.
    """
    return numgrad(hblau, pvec)


# helper functions for plotting the 'PES' and the NEB 
# and the NEB trajectory on top of it
# --------------------------------------------------


def draw_func(xmin, xmax, ymin, ymax, ax, res=1000, n_blocks=10, show_axes=False):
    """
    Draw the 2D function using matplolib.
    """
    x = np.linspace(xmin, xmax, num=res)
    y = np.linspace(ymin, ymax, num=res)

    XX, YY = np.meshgrid(x, y)
    Z = (XX ** 2 + YY - 11) ** 2 + (XX + YY ** 2 - 7) ** 2

    # Number of color blocks
    levels = np.linspace(np.min(Z), np.max(Z), n_blocks + 1)
    f = lambda x: x ** 2  # or x**1.5, for less strong compression
    levels = np.min(Z) + (np.max(Z) - np.min(Z)) * f(np.linspace(0, 1, n_blocks + 1))

    # Create a discrete viridis colormap
    viridis = plt.colormaps['viridis'].resampled(n_blocks)
    colors = viridis(np.arange(n_blocks))
    discrete_viridis = mcolors.ListedColormap(colors)

    ax.contourf(XX, YY, Z, levels=levels, cmap=discrete_viridis, alpha=0.7)
    ax.contour(XX, YY, Z, levels=levels, colors='black', linestyles='solid', linewidths=0.5)

    if not show_axes:
        ax.set_xticks([])
        ax.set_yticks([])


def draw_path(pvecs, ax):
    """
    Draw the 2D path of the images (Current NEB path).
    """
    xvals = pvecs[:, 0]
    yvals = pvecs[:, 1]
    ax.plot(xvals, yvals, 'o-', color='black')


def draw_trace(step_collection, ax):
    """
    Draw the movement of each image across the 2D surface.
    """
    for i in range(np.array(step_collection).shape[1]):
        xvals = np.array(step_collection)[:,i,0]
        yvals = np.array(step_collection)[:,i,1]
        ax.plot(xvals, yvals, '-', color='black', markersize=2, lw=0.7)


if __name__ == '__main__':
    main()
