import numpy as np
from datetime import timedelta
import time
import logging

import springforce_module as sfm
import step_pred_module as spm
import nebgrad_module as ngm
import tangent_module as tgm
import path_interpolator_module as pim
import struct_aligner_module as sam
import failed_img_recalculator as fir
import convsig_module as csm
import file_sys_io as io
from neb_path import NEBPath
import hessian_module as hm
import logging_module as lgm
from basic_neb import BasicNEB
from helper import project, normalize

logger = logging.getLogger(__name__)

Hartree_in_kJmol = 2625.49963948

# the following section contains the NEB Optimizer
# --------------------------------------------------------------------

class NEB_Optimizer(BasicNEB):
    def __init__(self, NEBPath:NEBPath, settings, log=True):
        # Parent class
        super().__init__(NEBPath, settings)

        # Begin with relaxed NEB
        self.relaxed = True
        self.ci_user_setting = self.climbing_image
        self.climbing_image = False
        self.max_rmsf = self.Relaxed_Max_RMSF_tol
        self.max_absf = self.Relaxed_Max_AbsF_tol

        # Set ups
        # Step predictor
        self.hessian = None
        self.hessians = [None] * self.images

        if self.step_pred_method == 'amgd':
            self.predictor = [spm.AMGD(self.stepsize_fac, self.AMGD_max_gamma) for _ in range(self.images)]

        elif self.step_pred_method == 'sd':
            self.predictor = [spm.SD(self.stepsize_fac) for _ in range(self.images)]

        elif self.step_pred_method in ('nr', 'rfo'):
            if self.step_pred_method == 'nr':
                self.predictor = spm.NewtonRaphson(self.images, self.BFGS_start, self.NR_start, self.stepsize_fac, self.AMGD_max_gamma)
            elif self.step_pred_method == 'rfo':
                self.predictor = spm.RationalFunction(self.images, self.BFGS_start, self.NR_start, self.stepsize_fac, self.AMGD_max_gamma)
            # Also set up global hessian
            self.hessian = hm.hessian(mode=self.initial_hessian, start=self.BFGS_start, labels=self.labels) 

        elif self.step_pred_method in ('sct', 'l-rfo', 'l-nr'):
            if self.step_pred_method == 'sct':
                self.predictor = spm.self_consistent_tangents(self.images, self.BFGS_start, self.NR_start, self.stepsize_fac, self.AMGD_max_gamma, self.soft_reset, self.soft_reset_memory)
            elif self.step_pred_method == 'l-rfo':
                self.predictor = spm.LocalRF(self.images, self.BFGS_start, self.NR_start, self.stepsize_fac, self.AMGD_max_gamma)
            elif self.step_pred_method == 'l-nr':
                self.predictor = spm.LocalNR(self.images, self.BFGS_start, self.NR_start, self.stepsize_fac, self.AMGD_max_gamma)
            # Also set up hessians
            self.hessians = [hm.hessian(mode=self.initial_hessian, start = self.BFGS_start, labels=self.labels) for _ in range(self.images)]
        else:
            raise ValueError('Error in with step prediction method. %s is not a valid step predictor mode.',
                               str(settings.step_pred_method))

        # Logging
        if log:
            self.logger = lgm.NEBLogger(settings.logfile_path)
        else:
            self.logger = None

    def calc_springgrads(self):
        """
        Callback function used to calculate springforce gradients
        of the images, as well as to recalculate the variable k
        constants if variable k and CI are active.
        """
        if self.use_vark:
            self.recalculate_varks()

        else:
            # if not, set all ks to be the value set in the settings
            self.path.set_img_k_const(self.k_const)

        # calculate regular springforce gradients. even if analytical positions scheme is active,
        # they are still needed for calculating convergence thresholds.
        full_path_pvecs = self.path.get_img_pvecs(include_ends=True)
        img_pair_ks = self.path.get_img_pair_ks()
        tanvecs = self.path.get_tanvecs()

        if self.spring_gradient == 'difference':
            springs = sfm.delta_springgrads(full_path_pvecs, img_pair_ks)
            springgrads = [spring * tanvec 
                           for spring, tanvec in zip(springs, tanvecs)]

        elif self.spring_gradient == 'projected':
            full_springgrads = sfm.full_springgrads(full_path_pvecs, img_pair_ks)
            springgrads = [ngm.project(full_springgrad, tanvec) 
                           for full_springgrad, tanvec in zip(full_springgrads, tanvecs)]

        elif self.spring_gradient == 'raw':
            springgrads = sfm.full_springgrads(full_path_pvecs, img_pair_ks)

        else:
            raise ValueError('Error with springforce definition. %s is not a valid springforce mode.',
                           str(self.spring_gradient))

        # if ci is active, the springforces acting on the ci are zero
        ci_index = self.current_CI_index
        if ci_index is not None:
            springgrads[ci_index][:] = 0.0

        return springgrads

    def recalculate_varks(self):
        """
        Helper function for the spring gradient function.
        Sets the spring constants for all image pairs according to
        the improved variable k scheme.
        """
        full_path_energies = self.path.get_energies(include_ends=True)
        maxk = self.k_const
        mink = self.k_const * self.vark_min_fac

        pairwise_ks = sfm.compute_pairwise_ks(full_path_energies,
                                              maxk,
                                              mink)
        self.path.set_img_pair_ks(pairwise_ks)

    def calc_nebgrads(self):
        """
        Callback function for calculating neb gradients,
        after engrads, springgrads, tanvecs have all been
        calculated and saved in the NEBPath object.
        """
        ci_index = self.current_CI_index
        engrads = self.path.get_engrads()
        img_pvecs = self.path.get_img_pvecs(include_ends=False)

        raw_nebgrads = ngm.calculate_nebgrads(engrads,
                                              self.path.get_springgrads(),
                                              self.path.get_tanvecs(),
                                              ci_index)

        # project out translation and/or rotation from neb gradients
        # (experimental feature), if selected by user
        nebgrads = ngm.sanitize_stepvecs(raw_nebgrads,
                                         img_pvecs,
                                         self.remove_gradtrans,
                                         self.remove_gradrot)

        # zero the gradients corresponding to frozen atoms,
        # so they don't skew the signals of convergence thresholds
        for i in range(len(nebgrads)):
            nebgrads[i] = ngm.freeze_atom_indices(nebgrads[i], self.frozen_atom_indices)

        # calculate the orthogonal gradients
        # project out the spring contribution, which is parallel to the tangent
        tanvecs = self.path.get_tanvecs()
        orth_grads = np.zeros_like(nebgrads)
        for i in range(len(nebgrads)):
            # except if there is a climbing image, which has a special NEB gradient.
            if i != ci_index:
                orth_grads[i] = ngm.reject(nebgrads[i], tanvecs[i])

        return nebgrads, orth_grads

    def calc_tanvecs(self):
        """
        Callback function for computing image tangent vectors.
        we want it to be able to both return normal tangents,
        or apply smoothing for the tangent vectors, depending on
        what the user chose.
        """
        # first, compute unsmoothed tans
        full_path_pvecs = self.path.get_img_pvecs(include_ends=True)
        full_path_energies = self.path.get_energies(include_ends=True)

        # tangent definitions
        if self.tangents == 'henkjon':
            raw_tans = tgm.henkjon_tans(full_path_pvecs,
                                        full_path_energies)
        elif self.tangents == 'simple':
            raw_tans = tgm.simple_tans(full_path_pvecs)
        else:
            raise ValueError('Error with tangent definition. %s is not a valid tangent mode.',
                           str(self.tangents))
        return raw_tans


    # -----------------------------------------------------------------------------------------
    # Step Prediction

    def predict_steps(self):
        """
        Do the step prediction with the saved step predictors
        """
        # first gather a bunch of data from the NEBPath
        full_energies   = self.path.get_energies(include_ends=True) 
        energies        = full_energies[1:-1] 

        nebgrads        = self.path.get_nebgrads()
        engrads         = self.path.get_engrads()
        full_path_pvecs = self.path.get_img_pvecs(include_ends=True)
        img_pvecs       = self.path.get_img_pvecs(include_ends=False)
        ci_index        = self.current_CI_index

        # remove spring contribution from neb gradients
        # if analytical position scheme is active
        if self.use_analytical_springpos:
            nebgrads = self.path.get_orth_grads()

        # Perform the step prediction
        if self.step_pred_method == 'sct':
            # Update the hessian objects by providing cartesian coordinates
            self.predictor.update(img_pvecs, engrads, energies, self.hessians)
            steps = self.predictor.predict(nebgrads, 
                                           self.hessians,
                                           full_energies,
                                           self.settings,
                                           self.path)

        elif self.step_pred_method in ['amgd', 'sd']:
            for object, pvec in zip(self.predictor, img_pvecs):
                object.update(pvec)
            steps = [object.predict(nebgrad) 
                     for object, nebgrad in zip(self.predictor, nebgrads)]

        elif self.step_pred_method in ['nr','rfo']:
            # no checking for failed calculations, is that good?
            self.predictor.update(img_pvecs, nebgrads, energies, self.hessian)
            steps = self.predictor.predict(img_pvecs, nebgrads, energies, self.hessian)

        elif self.step_pred_method in ['l-nr','l-rfo']:
            # no checking for failed calculations, is that good?
            self.predictor.update(img_pvecs, nebgrads, energies, self.hessians)
            steps = self.predictor.predict(img_pvecs, nebgrads, energies, self.hessians)

        steps = np.array(steps)

        # zero step and reset for energy None
        steps[energies==None] = np.zeros(int(self.atoms * 3))

        # Enforce maxstep
        if self.max_step is not None:
            steps = spm.enforce_maxsteps(steps, self.max_step)

        # If the user chose analytical position scheme,
        # we now add in the spring steps from the analytical
        # position scheme 
        if self.use_analytical_springpos:
            if self.step_pred_method == 'sct':
                logger.warning('Analytical spring position scheme cant be used with'
                                + ' self consisten tangents. The NEB will do the SCT'
                                + ' and ignore the use_analytical_springpos')
            else:
                img_pair_ks     = self.path.get_img_pair_ks()
                tanvecs         = self.path.get_tanvecs()
                ap_spring_steps = sfm.calc_analytic_springsteps(full_path_pvecs,
                                                                img_pair_ks,
                                                                tanvecs,
                                                                ci_index)

                # make sure none of the spring steps exceed max stepsize
                ap_spring_steps = spm.enforce_maxsteps(ap_spring_steps, self.max_step)

                # again, we will not compute steps for failed images, so we zero the spring steps in question
                ap_spring_steps[energies==None][:] = 0.0

                # add spring steps to our optimization steps
                steps += ap_spring_steps

        # Atom freezing (if selected)
        # their nebgrads should already be zeroed at this point. to ensure 
        # these atoms are not moved, we zero their step vecs as well.
        steps = np.array([ngm.freeze_atom_indices(step, self.frozen_atom_indices) for step in steps])

        return steps

    def giveup_signal_func(self):
        """
        Callback function that checks if NEB has reached an unrecoverable
        state -- or, more generally, a state an interface has flagged as a
        deliberate early-stop condition -- and should be aborted. Two
        conditions are checked:
        1. Too many failed images (the original check).
        2. An interface-supplied "driver" object (passed in via
           engrad_calc_kwargs, e.g. mace_ensemble_interface's
           EnsembleUncertaintyDriver for active-learning uncertainty
           quantification) has set its own .triggered flag. This is a
           generic hook -- any interface can opt into it by including a
           'driver' key in its engrfunc_kwargs with a .triggered attribute
           and a .trigger_message to log.
        """
        img_energies = self.path.get_energies()
        failed_img_count = 0
        for energy in img_energies:
            if energy is None:
                failed_img_count += 1
        failure_percentage = failed_img_count / len(img_energies)

        if failure_percentage > self.failed_img_tol_percent:
            # signal that the NEB should be aborted
            logger.error('Too many images failed to converge (%f %% failed). The NEB will be aborted. \nNOTE: the NEB ' +
                         'did not converge! Be careful with the results!', failure_percentage * 100.0)
            return True

        driver = getattr(self, 'engrad_calc_kwargs', {}).get('driver')
        if driver is not None and getattr(driver, 'triggered', False):
            # the driver already logged its own trigger_message when it
            # set .triggered -- nothing more to log here.
            #
            # This is a *deliberate*, successful early stop (e.g. an
            # active-learning uncertainty threshold), not a crash -- unlike
            # the too-many-failed-images case above, mark it with a
            # distinct state so neb.py's main() can tell the difference and
            # still write out the final path instead of raising NEBError.
            # This matters especially because NEB_Optimizer always begins
            # in a forced-relaxed pre-stage (see __init__): if the driver
            # triggers during that first stage, self.state would otherwise
            # stay 'FAILED' and main() would incorrectly treat a deliberate
            # stop as "something went wrong".
            self.state = 'DRIVER_STOPPED'
            return True

        # Signal that the NEB should not give up
        return False

    def failed_image_replacer_func(self):
        """
        Callback function for reinterpolating failed images
        """
        full_path_pvecs = self.path.get_img_pvecs(include_ends=True)
        full_path_energies = self.path.get_energies(include_ends=True)

        # our interpolation function is the same we used for generating
        # the initial path
        interp_func =  pim.interpolate_path

        # perform the reinterpolation of failed images
        new_full_path_pvecs = fir.replace_failed_images(full_path_pvecs,
                                                        full_path_energies,
                                                        interp_func,
                                                        self.labels,
                                                        self.interp_mode,
                                                        self.rot_align_mode,
                                                        self.settings)

        # in addition to the image replacing, the images need to
        # be recentered and rotationally realigned every iteration.
        new_full_path_pvecs = sam.align_path(new_full_path_pvecs,
                                                self.rot_align_mode,
                                                self.translate_mode)

        # replace NEBPath images with the now patched path
        patched_images = new_full_path_pvecs[1:-1]
        self.path.set_img_pvecs(patched_images)
        return self.path

    # ----------------------------------------------------------------------------------
    # the following section contains the callback functions for checking
    # NEB convergence. There are several stages of optimization (relaxed NEB,
    # NEB-CI, regular NEB) which have different convergence thresholds.
    # This is implemented by having three slightly different checker functions.

    def conv_checker_func(self, steps):
        if self.relaxed or not self.climbing_image:
            return self.conv_checker_func_noci(steps)
        else:
            return self.conv_checker_func_ci(steps)

    def conv_checker_func_noci(self, steps):
        energies = self.path.get_energies()

        # now perform the check for the signals of convergence
        nebgrads_o = self.path.get_orth_grads()
        nebgrads = self.path.get_nebgrads()

        max_rmsf_o = csm.NEB_RMSF(nebgrads_o)
        max_absf_o = csm.NEB_ABSF(nebgrads_o)
        max_rmsf = csm.NEB_RMSF(nebgrads)
        max_absf = csm.NEB_ABSF(nebgrads)

        # compile rmsf, absf etc. and put them into the log
        values_dict = {'RMSF' : max_rmsf,
                       'AbsF' : max_absf,
                       'RMSF_o': max_rmsf_o,
                       'AbsF_o': max_absf_o}

        if self.logger is not None:
            self.logger.write_to_log(self.path, values_dict)

        # print other information about the current NEB path
        self.do_iter_printout()

        if None in energies:
            logger.warning('There are still failed images left in the path.' +
                           ' There is no point in checking for signals of ' +
                           'convergence yet.')
            return False

        # now check for signals of convergence. get the values from settings
        Max_RMSF_tol = self.max_rmsf
        Max_AbsF_tol = self.max_absf

        logger.info('%-30s %f Tol. %f, %s', 'Current max. RMSF:', max_rmsf, Max_RMSF_tol, csm.yesno(max_rmsf<Max_RMSF_tol))
        logger.info('%-30s %f Tol. %f, %s', 'Current max. AbsF:', max_absf, Max_AbsF_tol, csm.yesno(max_absf<Max_AbsF_tol))

        logger.info('%-30s %f Tol. %f', 'Current max. orthogonal RMSF:', max_rmsf_o, Max_RMSF_tol)
        logger.info('%-30s %f Tol. %f', 'Current max. orthogonal AbsF:', max_absf_o, Max_AbsF_tol)

        # return the signal if converged
        if max_rmsf <= Max_RMSF_tol and max_absf <= Max_AbsF_tol:
            # The relaxed NEB did converge succesfully! However, if the NEB calculation continues after this
            # we still need to apply the optimization steps anyways, so this step doesn't get forgotten
            if self.relaxed and not self.relaxed_neb and (self.iteration < self.maxiter):
                self.relaxed = False
                self.max_rmsf = self.Max_RMSF_tol
                self.max_absf = self.Max_AbsF_tol
                self.climbing_image = self.ci_user_setting
                logger.info('Applying optimization steps.\n')
                self.path.set_img_pvecs(self.path.get_img_pvecs() + steps)
            return True

        else:
            return False

    def conv_checker_func_ci(self, steps):
        energies = self.path.get_energies()

        # now check for signals of convergence. get the values from settings
        Max_RMSF_tol = self.Max_RMSF_tol
        Max_AbsF_tol = self.Max_AbsF_tol
        CI_RMSF_tol = self.CI_RMSF_tol
        CI_AbsF_tol = self.CI_AbsF_tol

        # then, separate the nebgrad of the CI from the rest
        nebgrads = self.path.get_nebgrads()
        nebgrads_o = self.path.get_orth_grads()
        ci_index = self.current_CI_index

        ci_nebgrad = nebgrads[ci_index]
        path_nebgrads = np.concatenate([nebgrads[:ci_index],
                                        nebgrads[ci_index+1:]])
        path_nebgrads_o = np.concatenate([nebgrads_o[:ci_index],
                                          nebgrads_o[ci_index+1:]])

        # now perform the check for the signals of convergence
        max_rmsf_o = csm.NEB_RMSF(path_nebgrads_o)
        max_absf_o = csm.NEB_ABSF(path_nebgrads_o)
        max_rmsf = csm.NEB_RMSF(path_nebgrads)
        max_absf = csm.NEB_ABSF(path_nebgrads)
        ci_rmsf = csm.RMS(ci_nebgrad)
        ci_absf = csm.MaxAbs(ci_nebgrad)

        # print other information about the current NEB path
        self.do_iter_printout()

        # compile rmsf, absf etc. and put them into the log
        values_dict = {'RMSF' : max_rmsf,
                       'AbsF' : max_absf,
                       'RMSF_o': max_rmsf_o,
                       'AbsF_o': max_absf_o,
                       'RMSF_CI' : ci_rmsf,
                       'AbsF_CI' : ci_absf}

        if self.logger is not None:
            self.logger.write_to_log(self.path, values_dict)
        if None in energies:
            logger.warning('There are still failed images left in the path.' +
                           ' There is no point in checking for signals of ' +
                           'convergence yet.')
            return False

        logger.info('%-30s %f Tol. %f, %s', 'Current max. RMSF:', max_rmsf, Max_RMSF_tol, csm.yesno(max_rmsf<Max_RMSF_tol))
        logger.info('%-30s %f Tol. %f, %s', 'Current max. AbsF:', max_absf, Max_AbsF_tol, csm.yesno(max_absf<Max_AbsF_tol))

        logger.info('%-30s %f Tol. %f, %s', 'Current CI RMSF:', ci_rmsf, CI_RMSF_tol, csm.yesno(ci_rmsf<CI_RMSF_tol))
        logger.info('%-30s %f Tol. %f, %s', 'Current CI AbsF:', ci_absf, CI_AbsF_tol, csm.yesno(ci_absf<CI_AbsF_tol))

        # return the signal if converged
        if (max_rmsf <= Max_RMSF_tol and
            max_absf <= Max_AbsF_tol and
            ci_rmsf <= CI_RMSF_tol and
            ci_absf <= CI_AbsF_tol):
            return True

        else:
            return False

    def do_iter_printout(self):
        """
        Helper function for printing some NEB stats to the console
        """
        path_pvecs = self.path.get_img_pvecs(include_ends=True)
        energies = self.path.get_energies(include_ends=True)
        grads = self.path.get_engrads()
        tans = self.path.get_tanvecs()
        par_grads = [project(grad, normalize(tan)) for grad, tan in zip(grads, tans)]
        par_grads = [np.zeros_like(par_grads[0])] + par_grads + [np.zeros_like(par_grads[0])]
        labels = self.labels
        io.write_xyz_traj(labels,
                        path_pvecs,
                        self.resultdir / 'currenttraj.xyz',
                        energies=energies)
        energies_np = np.array(energies, dtype=float)  # forces None -> np.nan

        # HEI
        hei_index = np.nanargmax(energies_np)
        max_energy = energies_np[hei_index]
        io.write_xyz_file(labels, 
                          path_pvecs[hei_index], 
                          self.resultdir / 'HEI_trj.xyz',
                          mode='a',
                          energy=max_energy)

        if not self.ci_user_setting:
            # TS_guess cubic like HJ
            ts_coords_hj = pim.interpolate_TS_cubic(path_pvecs,
                                                    energies_np,
                                                    par_grads)
            io.write_xyz_file(labels, 
                              ts_coords_hj, 
                              self.resultdir / 'TS_guess_cubic_trj.xyz',
                              mode='a')

        left_barrier_kJmol = (max_energy - energies[0]) * Hartree_in_kJmol
        right_barrier_kJmol = (max_energy - energies[-1]) * Hartree_in_kJmol
        time_elapsed = time.time() - self.start_time

        logger.info('Wall time elapsed (h:m:s): %s', str(timedelta(seconds=time_elapsed)))
        logger.info('There are %d failed images in the path.',  self.path.n_failed_images())

        ci_index = self.current_CI_index
        if ci_index is not None:
            logger.info('Image at index %d is now the climbing image.', ci_index)

        logger.info('Approx. Barrier with respect to left end: %f kJ/mol', left_barrier_kJmol)
        logger.info('Approx. Barrier with respect to right end: %f kJ/mol', right_barrier_kJmol)


class SCT_Optimizer(BasicNEB):
    def __init__(self, NEBPath:NEBPath, settings=None):
        super().__init__(NEBPath, settings)

        # Step predictor is AMGD
        self.predictor = [spm.AMGD(self.harmonic_stepsize_fac, self.AMGD_max_gamma) for _ in range(self.images)]
        self.maxiter = 400

    def predict_steps(self):
        """
        Do the step prediction with the saved step predictors
        """
        # first gather a bunch of data from the NEBPath
        nebgrads        = self.path.get_nebgrads()
        img_pvecs       = self.path.get_img_pvecs(include_ends=False)

        # Update the AMGD objects
        for object, pvec in zip(self.predictor, img_pvecs):
            object.update(pvec)

        # Perform the step prediction
        steps = [object.predict(nebgrad) 
                 for object, nebgrad in zip(self.predictor, nebgrads)]

        # Enforce maxstep
        if self.max_step is not None:
            steps = spm.enforce_maxsteps(steps, self.max_step)
        return steps

    def calc_springgrads(self):
        """
        Callback function used to calculate springforce gradients
        of the images, as well as to recalculate the variable k
        constants if variable k and CI are active.
        """
        if self.use_vark:
            self.recalculate_varks()

        else:
            # if not, set all ks to be the value set in the settings, to make sure
            self.path.set_img_k_const(self.k_const)

        # calculate regular springforce gradients.
        full_path_pvecs = self.path.get_img_pvecs(include_ends=True)
        img_pair_ks = self.path.get_img_pair_ks()
        tanvecs = self.path.get_tanvecs()

        if self.spring_gradient == 'difference':
            springs = sfm.delta_springgrads(full_path_pvecs, img_pair_ks)
            springgrads = [spring * tanvec 
                           for spring, tanvec in zip(springs, tanvecs)]

        elif self.spring_gradient == 'projected':
            full_springgrads = sfm.full_springgrads(full_path_pvecs, img_pair_ks)
            springgrads = [ngm.project(full_springgrad, tanvec) 
                           for full_springgrad, tanvec in zip(full_springgrads, tanvecs)]

        elif self.spring_gradient == 'raw':
            springgrads = sfm.full_springgrads(full_path_pvecs, img_pair_ks)

        else:
            raise ValueError('Error with springforce definition. %s is not a valid springforce mode.',
                           str(self.spring_gradient))

        # if ci is active, the springforces acting on the ci are zero
        ci_index = self.current_CI_index
        if ci_index is not None:
            springgrads[ci_index][:] = 0.0

        return springgrads

    def calc_tanvecs(self):
        """
        Callback function for computing image tangent vectors.
        we want it to be able to both return normal tangents,
        or apply smoothing for the tangent vectors, depending on
        what the user chose.
        """
        full_path_pvecs = self.path.get_img_pvecs(include_ends=True)
        full_path_energies = self.path.get_energies(include_ends=True)
        
        # use tangent definition
        if self.tangents == 'henkjon':
            raw_tans = tgm.henkjon_tans(full_path_pvecs,
                                        full_path_energies)
        elif self.tangents == 'simple':
            raw_tans = tgm.simple_tans(full_path_pvecs)
        else:
            raise ValueError('Error with tangent definition. %s is not a valid tangent mode.',
                           str(self.tangents))
        return raw_tans

    def calc_nebgrads(self):
        """
        Callback function for calculating neb gradients,
        after engrads, springgrads, tanvecs have all been
        calculated and saved in the NEBPath object.
        """
        ci_index = self.current_CI_index
        engrads = self.path.get_engrads()
        img_pvecs = self.path.get_img_pvecs(include_ends=False)

        raw_nebgrads = ngm.calculate_nebgrads(engrads,
                                              self.path.get_springgrads(),
                                              self.path.get_tanvecs(),
                                              ci_index)

        # project out translation and/or rotation from neb gradients
        # (experimental feature), if selected by user
        nebgrads = ngm.sanitize_stepvecs(raw_nebgrads,
                                         img_pvecs,
                                         self.remove_gradtrans,
                                         self.remove_gradrot)

        # zero the gradients corresponding to frozen atoms,
        # so they don't skew the signals of convergence thresholds
        for i in range(len(nebgrads)):
            nebgrads[i] = ngm.freeze_atom_indices(nebgrads[i], self.frozen_atom_indices)

        # calculate the orthogonal gradients
        # project out the spring contribution, which is parallel to the tangent
        tanvecs = self.path.get_tanvecs()
        orth_grads = np.zeros_like(nebgrads)
        for i in range(len(nebgrads)):
            # except if there is a climbing image, which has a special NEB gradient.
            if i != ci_index:
                orth_grads[i] = ngm.reject(nebgrads[i], tanvecs[i])
        return nebgrads, orth_grads
    
    def recalculate_varks(self):
        """
        Helper function for the spring gradient function.
        Sets the spring constants for all image pairs according to
        the improved variable k scheme.
        """
        full_path_energies = self.path.get_energies(include_ends=True)
        maxk = self.k_const
        mink = self.k_const * self.vark_min_fac

        pairwise_ks = sfm.compute_pairwise_ks(full_path_energies,
                                              maxk,
                                              mink)
        self.path.set_img_pair_ks(pairwise_ks)

    def conv_checker_func(self, steps):
        nebgrads = self.path.get_nebgrads()

        rmsf = csm.NEB_RMSF(nebgrads)
        absf = csm.NEB_ABSF(nebgrads)

        max_rmsf = self.harmonic_conv_fac * self.Max_RMSF_tol
        max_absf = self.harmonic_conv_fac * self.Max_AbsF_tol

        logger.warning('Harmonic Max. RMSF: %f Tol.: %f', rmsf, max_rmsf)
        logger.warning('Harmonic Max. Abs. F: %f Tol.: %f', absf, max_absf)

        if rmsf < max_rmsf and absf < max_absf:
            return True
        else:
            return False
