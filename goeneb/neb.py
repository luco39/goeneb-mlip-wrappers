from pathlib import Path
import numpy as np
import time
import os
import logging

import neb_path as path
import neb_exceptions as nex
import path_interpolator_module as pim
import neb_configparse as conf
import file_sys_io as io
import logging_module as lgm
import struct_aligner_module as sam
import IDPP_module as idpp
from neb_optimizer import NEB_Optimizer
from logo import logo
from interfaces.engrad_interface import gather_engrfunc
from helper import project, normalize

Hartree_in_kJmol = 2625.49963948


# the following function is what's being executing upon starting
# the program in the terminal
# --------------------------------------------------------------

def main(args):
    logger = logging.getLogger(__name__)
    logger.info(logo)

    tempdir = None
    workdir = None

    try:
        # gather user-given data
        inpfile_path = io.get_full_path(Path(args.input_file))

        # read the input file, overwrite if options are given in commandline
        settings = process_input_path(inpfile_path)
        if args.verbose is not None:
            settings.verbose = args.verbose
        if args.maxiter is not None:
            settings.maxiter = args.maxiter
        if args.images is not None:
            settings.n_images = args.images
        if args.trajtest:
            settings.trajtest = True

        # Adjust logging level
        level, message = lgm.translate_level(settings.verbose)
        logger.info(message)
        logger.setLevel(level)

        logger.debug('Chosen settings:\n%s', settings)

        # find/generate resultdir, where the results should be stored
        workdir, resultdir = find_workdir(inpfile_path)
        settings.resultdir = resultdir
        settings.workdir = workdir

        # find/generate tempdir, where temporary files should be stored
        tempdir = find_tempdir(settings)

        # find/generate the starting path
        settings.get_complete_paths()
        labels, starting_path = produce_starting_path(settings)
        if settings.starttraj is not None:
            # the starttraj fixes the image count, not n_images; keep the two
            # in step, since interfaces may rely on n_images (e.g. the ensemble
            # MACE driver uses it to tell real NEB iterations apart)
            settings.n_images = len(starting_path) - 2

        # save the starting path into the workdir
        io.write_xyz_traj(labels, starting_path, resultdir / 'starttraj.xyz')

        # if trajtest is selected, only the starting path should be printed.
        # in that case, we're already done.
        if settings.trajtest is True:
            io.safe_delete_dir(tempdir)
            return

        # generate NEBPath object
        # ------------------------------
        # first, set up our energy and gradient calculation function
        engrfunc, engrfunc_kwargs = gather_engrfunc(labels,
                                                    settings,
                                                    tempdir)

        # prepare the logger for the progress of the (NEB) optimization
        logfile_path = resultdir / 'optlog.csv'
        settings.logfile_path = logfile_path

        # now we can generate the NEBPath object, which will hold the
        # relevant data for the neb path, like structures, labels,
        # tangent vectors, neb gradients etc.
        nebpath = path.NEBPath(labels,
                               starting_path,
                               engrfunc,
                               engrfunc_kwargs,
                               settings)

        # do the actual NEB optimization
        # ------------------------------
        settings.start_time = time.time()

        optimizer = NEB_Optimizer(nebpath, settings)
        nebpath, return_state, iterations = optimizer.do_opt_loop(engrfunc, engrfunc_kwargs)

        if (settings.relaxed_neb or iterations >= settings.maxiter
                or return_state == 'DRIVER_STOPPED'):
            # if relaxed NEB is active, or the iteration budget is
            # used up, or an interface driver deliberately stopped the
            # NEB early (e.g. an active-learning ensemble-uncertainty
            # threshold was exceeded), we're already done now.
            if return_state == 'SUCCESS':
                logger.info('\n\nRelaxed NEB convergence has been reached after %d iterations!\n\n', iterations)
            elif return_state == 'DRIVER_STOPPED':
                logger.warning('\n\nThe NEB optimization was stopped early by an interface ' +
                               'driver during the relaxed pre-optimization stage (e.g. an ' +
                               'active-learning ensemble-uncertainty threshold was exceeded) ' +
                               'before reaching convergence. The path has not converged -- ' +
                               'this is expected for this mode. See the driver\'s own log ' +
                               'messages/marker file for details.\n\n')
            else:
                logger.warning('\n\nWarning: the NEB did not converge, it just ' +
                                'reached the maximum number of iterations! Be ' +
                                'careful with the results!\n\n')
            do_end_of_opt_printout(optimizer, resultdir)
            return

        # if the previous stage of optimization didn't converge
        # for some reason other than running out of iterations
        # (i.e. it crashed), we cant continue and must raise
        # an error
        if return_state != 'SUCCESS':
            raise nex.NEBError('Error: something went wrong during relaxed ' +
                               'NEB optimization stage. Aborting...')

        # if the user selected climbing image, we now need to
        # do the rest of the optimization with CI active.
        logger.info('Initial stage of NEB optimization complete after %d iterations.', iterations)
        if settings.climbing_image:
            logger.info('Activating climbing image.\n')
        elif iterations < settings.maxiter:
            logger.info('Continueing with main optimization stage.\n')

        nebpath, return_state, iterations = optimizer.do_opt_loop(engrfunc, engrfunc_kwargs)

        # whatever the user had selected is now finished.

        if return_state == 'SUCCESS':
            logger.info('\n\nNEB convergence has been reached after %d iterations!\n\n', iterations)
        elif return_state == 'DRIVER_STOPPED':
            logger.warning('\n\nThe NEB optimization was stopped early by an interface ' +
                           'driver (e.g. an active-learning ensemble-uncertainty threshold ' +
                           'was exceeded) before reaching convergence. The path has not ' +
                           'converged -- this is expected for this mode. See the driver\'s ' +
                           'own log messages/marker file for details.\n\n')
        else:
            logger.warning('\n\nWarning: the NEB did not converge, it just ' +
                            'reached the maximum number of iterations! Be ' +
                            'careful with the results!\n\n')

        do_end_of_opt_printout(optimizer, resultdir)

        return

    except nex.EngradError:
        # if the ends fail to converge, there's probably something
        # wrong with the way the user set up the engrad calculations.
        # copy over the calculation results so they can find out
        # what's wrong.
        if tempdir is not None:
            io.move_dir(tempdir, workdir)
            logger.warning('There might be a problem with the Engrad calculations. ' + 
                           'Contents of tempdir were copied to the job directory.')
        raise

    finally:
        if tempdir is not None:
            # remove tempdir if it wasn't removed already
            io.safe_delete_dir(tempdir, does_not_exist_ok=True)
            logger.info('Temporary files deleted.')


def do_end_of_opt_printout(optimizer, resultdir):
    """Do final printout of the optimization stats. Also save the final structures of the path.
    - Iteration Information
    - Final trajectory
    - Highest Energy Image
    - Interpolated TS guess"""
    # do final printout
    logger = logging.getLogger(__name__)
    logger.info('Final stats of the NEB path:\n')
    optimizer.do_iter_printout()

    # print final trajectory
    nebpath = optimizer.path
    final_path = nebpath.get_img_pvecs(include_ends=True)
    final_energies = np.array(nebpath.get_energies(include_ends=True))
    grads = nebpath.get_engrads()
    tans = nebpath.get_tanvecs()
    par_grads = [project(grad, normalize(tan)) for grad, tan in zip(grads, tans)]
    par_grads = [np.zeros_like(par_grads[0])] + par_grads + [np.zeros_like(par_grads[0])]
    labels = nebpath.get_labels()
    finaltraj_filepath = resultdir / 'finaltraj.xyz'
    io.write_xyz_traj(labels,
                      final_path,
                      finaltraj_filepath,
                      energies=final_energies)

    logger.info('Result trajectory printed under %s', str(finaltraj_filepath))

    # Highest Energy Image
    hei_index = np.nanargmax(final_energies)
    hei_coords = final_path[hei_index]
    hei_energy = final_energies[hei_index]
    hei_filepath = resultdir / 'HEI.xyz'
    io.write_xyz_file(labels,
                      hei_coords,
                      hei_filepath,
                      energy=hei_energy)

    logger.info('Highest Energy Image printed under %s', str(hei_filepath))

    # TS Guess quadratic
    if not optimizer.climbing_image:
        ts_guess_filepath = resultdir / 'TS_guess_cubic.xyz'
        ts_coords = pim.interpolate_TS_cubic(final_path, final_energies, par_grads)
        io.write_xyz_file(labels,
                          ts_coords, 
                          ts_guess_filepath)

        logger.info('Cubic interpolated TS Guess structure printed under %s', str(ts_guess_filepath))


# the following section contains helper functions for gathering starting
# trajectories or gathering end structures and interpolating starting paths
# --------------------------------------------------------------------------


def produce_starting_path(settings):
    """Produces a starting path either by reading and optimizing a given one or by
    interpolating in the user specified manner (cartesian, internal or geodesic).
    Includes the TS_guess structure if given.
    """
    if settings.starttraj is not None:
        # a starting trajectory was given, just gather that one
        labels, starting_path = gather_starting_path(settings)

        # apply structure alignment
        starting_path = sam.align_path(starting_path, settings.rot_align_mode, settings.translate_mode)
        
        # apply IDPP if selected
        if settings.IDPP:
            starting_path = idpp.do_IDPP_opt_pass(starting_path, settings)

    else:
        # two end structures should be given, generate starting path from that
        if settings.TS_guess is None:
            # only the two ends, gather them, make interpolation
            labels, starting_path = generate_from_ends(settings)

        else:
            # a TS guess structure has been given, make interpolation
            # with that structure in the middle of the path
            labels, starting_path = generate_from_ends_and_TS(settings)

    return labels, starting_path


def gather_end_structures(settings):
    """Gather the end structures (xyzs) and the atom labels."""
    start_xyz = settings.start_structure
    end_xyz = settings.end_structure

    if start_xyz is None or end_xyz is None:
        raise nex.NEBError('Either start_structure or end_structure (or both) ' +
                           'have not been set! Please make sure they are located in the working directory, ' +
                           'or correctly specified in the input file.')

    start_labels, start_pvec = io.read_xyz_file(start_xyz)
    end_labels, end_pvec = io.read_xyz_file(end_xyz)

    if start_labels != end_labels:
        raise nex.NEBError('Atomic labels for the two' +\
                           ' end structures do not match!')

    labels = start_labels
    return labels, start_pvec, end_pvec


def gather_starting_path(settings):
    """Gather the complete starting trajectory as well as the atom labels."""
    starttraj = settings.starttraj
    labels, path_pvecs = io.read_xyz_traj(starttraj)

    return labels, path_pvecs


def gather_TS_guess(settings):
    """Gather the TS guess structure as well as the atom labels."""
    tsguess_xyz = settings.TS_guess
    ts_labels, ts_pvec = io.read_xyz_file(tsguess_xyz)

    return ts_labels, ts_pvec


def generate_from_ends(settings):
    """Interpolate the starting path in the user specified method (cartesian, internal, geodesic).
    This function also applies the translational (and rotational) alignement as well 
    as the IDPP optimization. Starts from the two end structures."""
    labels, start_pvec, end_pvec = gather_end_structures(settings)

    # recenter and align the given structures
    end_sequence = np.vstack([start_pvec, end_pvec])
    end_sequence = sam.align_path(end_sequence,
                                  settings.rot_align_mode,
                                  settings.translate_mode)

    start_pvec = end_sequence[0]
    end_pvec = end_sequence[1]

    # generate the starting path 
    startpath = pim.interpolate_path(start_pvec,
                                     end_pvec,
                                     settings.n_images,
                                     labels,
                                     settings.interp_mode,
                                     settings.rot_align_mode,
                                     settings)

    return labels, startpath


def generate_from_ends_and_TS(settings):
    """Interpolate the starting path in the user specified method (cartesian, internal, geodesic).
    This function also applies the translational (and rotational) alignement as well 
    as the IDPP optimization. Starts from the two end structures and a TS guess structure."""
    labels, start_pvec, end_pvec = gather_end_structures(settings)
    ts_labels, ts_pvec = gather_TS_guess(settings)

    if labels != ts_labels:
        raise nex.NEBError('Atomic labels of end structures and TS guess' +
                           ' do not match!')

    # recenter and align the given structures
    end_sequence = np.vstack([start_pvec, ts_pvec, end_pvec])
    end_sequence = sam.align_path(end_sequence,
                                  settings.rot_align_mode,
                                  settings.translate_mode)

    start_pvec = end_sequence[0]
    ts_pvec = end_sequence[1]
    end_pvec = end_sequence[2]

    # generate the starting path
    n_imgs_per_segment= int(settings.n_images / 2)

    kwargs = {'n_new_interps' : n_imgs_per_segment,
              'labels' : labels,
              'interp_mode' : settings.interp_mode,
              'rot_align_mode' : settings.rot_align_mode,
              'IDPP' : settings.IDPP,
              'SIDPP' : settings.SIDPP,
              'IDPP_maxiter' : settings.IDPP_maxiter,
              'IDPP_max_RMSF' : settings.IDPP_max_RMSF,
              'IDPP_max_AbsF': settings.IDPP_max_AbsF,
              'max_step' : settings.max_step}

    segment1 = pim.interpolate_path(start_pvec,
                                    ts_pvec,
                                    **kwargs)

    segment2 = pim.interpolate_path(ts_pvec,
                                    end_pvec,
                                    **kwargs)

    complete_path = np.concatenate([segment1, segment2[1:]])

    return labels, complete_path


# the following section contains helper functions for processing
# the arguments given to the program upon being started from
# console, as well as helper functions for finding/generating
# the result directory and the directory for temporary files.
# -------------------------------------------------------------


def process_input_path(inpfile_path:Path):
    """
    Get the input parameters either from the input file or from the work directory. Returns a settings object.
    """
    logger = logging.getLogger(__name__)

    # check if inpfile_path is an actual file, or if it's a directory
    if inpfile_path.is_file():
        # load the input file normally
        settings = conf.Settings(inpfile_path)

    elif inpfile_path.is_dir():
        # The given path is the workdir
        logger.warning('Warning: directory was given as argument in program call. ' +
                       'This requires the directory to include all nessecary files in the ' +
                       'correctly named manner.')
        settings = process_workdir(inpfile_path)

    else:
        raise FileNotFoundError('Error: argument given in program call is no '
                                'valid file or directory: ' + str(inpfile_path))

    return settings


def process_workdir(workdir:Path):
    """
    Sort all nessecary files given in the work directory and read the input file.
    Returns a settings object.
    """
    logger = logging.getLogger(__name__)

    # try to find ini file
    inifiles = list(workdir.glob("*.ini"))
    if len(inifiles) != 1:
        raise OSError('Error: ' + str(workdir) + ' is not a valid' +
                           ' workdir. It must contain exactly one ini file.')
    inipath = workdir / inifiles[0]

    # try to read the ini, skipping the '[options]' line at the start
    settings = conf.Settings(inipath)

    # if we have a starttraj given, the conf data is already compatible
    if settings.starttraj is not None:
        return settings

    # If start and end structure are explicitly given, there is also no need for further processing
    elif (settings.start_structure is not None 
          and settings.end_structure is not None):
        return settings

    else:
        # all xyz files need to be automatically read and sorted
        logger.info('Structures are not specified in input-file. ' +
                    'The xyz-files are automatically sorted.')
        xyz_files = list(workdir.glob("*.xyz"))

        one = [filepath for filepath in xyz_files if "1" in filepath.name]
        two = [filepath for filepath in xyz_files if "2" in filepath.name]
        ts = [filepath for filepath in xyz_files if "TS" in filepath.name]

        if (len(one) != 1) or (len(two) != 1) or (len(ts) > 1):
            logger.warning('It was not clear what structure is supposed to be ' +
                           'start, end and TS structure. ' +
                           'Now its sorted alphabetically. Have better names.\n' +
                           '(Prefaribly *1.xyz, *2.xyz and *TS.xyz)')
            xyz_files.sort()

            # Remove TS guess from xyz_filepaths
            if settings.TS_guess is not None:
                TS_path = settings.TS_guess
                xyz_files = [filepath for filepath in xyz_files
                             if not filepath.samefile(TS_path)]

            # we should be left with exactly two xyz files in case of NEB
            if len(xyz_files) != 2:
                raise OSError('Error: ' + str(workdir) + ' is not a valid' +
                                ' workdir. There must be two xyz files for' +
                                ' the two end structures.')

            if settings.start_structure is None:
                settings.start_structure = xyz_files[0]
            if (settings.end_structure is None):
                settings.end_structure = xyz_files[1]

        else:
            # Overwrite only if necessary
            if settings.start_structure is None:
                settings.start_structure = one[0]
            if settings.end_structure is None:
                settings.end_structure = two[0]
            if (settings.TS_guess is None) and (len(ts) == 1):
                settings.TS_guess =  ts[0]

        logger.info('Start structure set to: %s', settings.start_structure)
        logger.info('End structure set to: %s', settings.end_structure)
        logger.info('TS structure set to: %s', settings.TS_guess)

        return settings


def find_workdir(inpfile_path:Path):
    """Get the path of the work directory (the results folder)."""
    logger = logging.getLogger(__name__)
    if inpfile_path.is_dir():
        workdir = inpfile_path
    elif inpfile_path.is_file():
        workdir = inpfile_path.parent

    # make sure to get the full path
    workdir = io.get_full_path(workdir)

    # in this directory, create a results folder
    folder_name = find_resultfolder_name(workdir)
    resultdir = workdir / folder_name

    # check if this directory can be created and accessed properly
    try:
        io.safe_create_dir(resultdir)

    except OSError:
        logger.error('Error when trying to access job directory. ' +
                     'See message below. Make sure you '
                     'have writing access to the job directory.')
        raise

    # the jobdir was successfully found and/or created.
    return workdir, resultdir


def find_resultfolder_name(workdir:Path):
    """Gets the lowest number results folder that does not exist yet."""
    # first, find any existing results folders
    if not (workdir / 'results').is_dir():
        return 'results'

    folder_nr = 0
    while True:
        folder_nr += 1
        folder_name = 'results' + str(folder_nr)
        if not (workdir / folder_name).exists():
            return folder_name


def find_tempdir(settings):
    """Reurn the correct path of the temp directory. Either from the input file or the 
    environment variable TMPDIR."""
    logger = logging.getLogger(__name__)
    if settings.tempdir is not None:
        logger.info('Using tempdir specified in input file.')
        tempdir = settings.tempdir

    else:
        tempdir = os.getenv('NEB_TMPDIR')
        if tempdir is None:
            raise nex.MissingEnvironmentVariable('Error: NEB_TMPDIR environment variable not set.' +\
                             'Set this variable before starting, or ' +\
                             'specify a directory under the "tempdir" ' +\
                             'keyword in the input file.')
        tempdir = Path(tempdir)

    # make sure to get the full path
    tempdir = io.get_full_path(tempdir)

    # check if this directory can be accessed properly
    if not io.check_directory(tempdir):
        raise OSError('Error when trying to access temporary ' +
                           'files directory. See message above. ' +
                           'Make sure you set the tempdir setting ' +
                           'in the input file, or the NEB_TMPDIR environment ' +
                           'variable correctly.')

    # generate a folder in the directory for our job
    if any(tempdir.iterdir()):
        logger.warning('The given temp dir is not empty, a new directory will be created inside.')
        job_id = os.getenv('SLURM_JOB_ID')
        if job_id is None:
            job_id = time.strftime('%Y%m%d_%H%M%S')
        tempdir = tempdir / job_id
        io.safe_create_dir(tempdir)

    logger.info('Temporary files stored under: \n' + str(tempdir))
    return tempdir
