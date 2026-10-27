import os
import logging
from pathlib import Path
import numpy as np
import time
import argparse

from logging_module import setup_logger

main_logger = setup_logger(name='test', level='error')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


from logo import logo
from interfaces.engrad_interface import gather_engrfunc
from file_sys_io import read_xyz_file
from struct_aligner_module import align_path
from path_interpolator_module import do_interpolation
from IDPP_module import do_IDPP_opt_pass
from neb_path import NEBPath
from neb_optimizer import NEB_Optimizer
from neb_configparse import Settings
import neb_exceptions as nex

np.set_printoptions(precision=12)

def main():
    logger.info(logo)
    logger.info('\n\n    Test module of the GöNEB started.\n    All relevant functions of the ' +
                'NEB are tested and compared with reference values\n    to ensure ' + 
                'your set up is working as expected.\n\n')

    parser = argparse.ArgumentParser(description="Parameters for testing module")
    parser.add_argument('-v', '--verbose', action='store_true', help='Print energies and gradients.')
    parser.add_argument('-vv', '--veryverbose', action='store_true', help='Activate logging for background processes.')
    parser.add_argument('-s', '--skip', action='store_true', help='Skip the interface testing.')
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)
    if args.veryverbose:
        logger.setLevel(logging.DEBUG)
        main_logger.setLevel(logging.INFO)

    # Get reference structure
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(script_dir, 'testjobs/pentadiene-1.xyz')
    labels, pvec = read_xyz_file(filepath)

    tempdir = Path(os.environ['NEB_TMPDIR'])

    # Test the interfaces
    logger.info('')
    logger.info('INTERFACES')
    if not args.skip:
        test_orca(labels, pvec, tempdir)
        test_molpro(labels, pvec, tempdir)
        test_gaussian(labels, pvec, tempdir)
        test_tblite(labels, pvec, tempdir)

    filepath = os.path.join(script_dir, 'testjobs/pentadiene-2.xyz')
    labels, pvec2 = read_xyz_file(filepath)

    logger.info('')
    logger.info('ALIGNEMENT AND INTERPOLATION')
    test_alignment(pvec, pvec2)
    test_interpolation(pvec, pvec2, labels, 'cartesian')
    test_interpolation(pvec, pvec2, labels, 'internal')
    test_interpolation(pvec, pvec2, labels, 'geodesic')
    test_idpp(pvec, pvec2)

    logger.info('')
    logger.info('NEB GRADIENTS')
    test_nebgrad(labels, pvec, pvec2)

    logger.info('')
    logger.info('STEP PREDICTION')
    test_step_prediction(labels, pvec, pvec2, tempdir)

    logger.info('')
    logger.info('NEB Testing complete.')
    logger.info('')
    logger.info('You should get CHECK for all tests concluded in this test module (except geodesic). ' + 
                'If this is not the case, you are not working with the same ' +
                'functionality as the developers! You need to figure out the reason why.\n\n' +
                '1. Did you set the correct environment variables for all external programs you ' + 
                'want to use (ORCA_EXE, MOL_EXE, GAUSS_EXE and MPI_PATH in the case of ORCA) ?\n' + 
                '2. Are you running GöNEB in the correct python virtual environment?\n' + 
                '3. Did you install all the required python modules (including tblite if you are planning on using xTB)?\n' +
                '4. Run this testing module with the flag -v or --verbose to get the caluclated values.\n' + 
                '5. Run this testing module with the flag -vv or --veryverbose to activate the output for all background activities.\n\n' + 
                'If you can not detect the problem this way consider asking your system admin for help and feel free to contact the developers.\n')



def check_value(value, reference):
    error = value - reference
    close = np.isclose(value, reference)
    return error, close


def check_vector(vector, reference):
    rmsd = np.sqrt(np.mean((vector - reference) ** 2))
    close = np.allclose(vector, reference)
    return rmsd, close


def bool2msg(bool):
    if bool:
        return 'CHECK'
    else:
        return '!!!FAILED!!!'


def test_step_prediction(labels, start_pvec, end_pvec, tempdir):
    logger = logging.getLogger(__name__)
    try:
        path = [start_pvec, cart_interp, end_pvec]
        logpath = tempdir / 'test.csv'
        thistime = time.time()

        # Build settings
        settings = Settings()
        settings.interface = 'dummy'
        settings.n_images = 1
        settings.k_const = 1
        settings.Relaxed_Max_RMSF_tol = 0
        settings.Relaxed_Max_AbsF_tol = 0
        settings.step_pred_method = 'sd'
        settings.maxiter = 2
        settings.logfile_path = logpath
        settings.max_step = 5
        settings.workdir = tempdir
        settings.resultdir = tempdir
        settings.interp_mode = 'cartesian'
        settings.start_time = thistime
        settings.relaxed_neb = True
        settings.failed_img_tol_percent = 0

        logger.info('-----  SD             -----')
        engrfunc, engrfunc_kwargs = gather_engrfunc(labels, settings, None)
        path_obj = NEBPath(labels, path, engrfunc, engrfunc_kwargs, settings)
        optimizer = NEB_Optimizer(path_obj, settings)
        nebpath, return_state, iterations = optimizer.do_opt_loop(engrfunc, engrfunc_kwargs)
        logger.debug(list(nebpath.get_img_pvecs()[0]))
        rmsd, close = check_vector(nebpath.get_img_pvecs()[0], dummy_sd)
        logger.info(f'Steepest descent:                RMSD:  {rmsd:+8.4e}   |   {bool2msg(close)}')
        logger.info('')

        logger.info('-----  AMGD           -----')
        # adjust settings
        settings.step_pred_method = 'amgd'
        settings.maxiter = 3

        engrfunc, engrfunc_kwargs = gather_engrfunc(labels, settings, None)
        path_obj = NEBPath(labels, path, engrfunc, engrfunc_kwargs, settings)
        optimizer = NEB_Optimizer(path_obj, settings)
        nebpath, return_state, iterations = optimizer.do_opt_loop(engrfunc, engrfunc_kwargs)
        logger.debug(list(nebpath.get_img_pvecs()[0]))
        rmsd, close = check_vector(nebpath.get_img_pvecs()[0], dummy_amgd)
        logger.info(f'AMGD:                            RMSD:  {rmsd:+8.4e}   |   {bool2msg(close)}')
        logger.info('')

        logger.info('-----  RFO            -----')
        # adjust settings
        settings.step_pred_method = 'rfo'
        settings.maxiter = 4
        settings.BFGS_start = 2
        settings.NR_start = 3

        engrfunc, engrfunc_kwargs = gather_engrfunc(labels, settings, None)
        path_obj = NEBPath(labels, path, engrfunc, engrfunc_kwargs, settings)
        optimizer = NEB_Optimizer(path_obj, settings)
        nebpath, return_state, iterations = optimizer.do_opt_loop(engrfunc, engrfunc_kwargs)
        logger.debug(list(nebpath.get_img_pvecs()[0]))
        rmsd, close = check_vector(nebpath.get_img_pvecs()[0], dummy_rfo)
        logger.info(f'RFO:                             RMSD:  {rmsd:+8.4e}   |   {bool2msg(close)}')
        logger.info('')

        logger.info('-----  SCT            -----')
        # adjust settings
        settings.step_pred_method = 'sct'

        engrfunc, engrfunc_kwargs = gather_engrfunc(labels, settings, None)
        path_obj = NEBPath(labels, path, engrfunc, engrfunc_kwargs, settings)
        optimizer = NEB_Optimizer(path_obj, settings)
        nebpath, return_state, iterations = optimizer.do_opt_loop(engrfunc, engrfunc_kwargs)
        logger.debug(list(nebpath.get_img_pvecs()[0]))
        rmsd, close = check_vector(nebpath.get_img_pvecs()[0], dummy_sct)
        logger.info(f'SCT:                             RMSD:  {rmsd:+8.4e}   |   {bool2msg(close)}')

    except Exception as e:
        logger.error(f'STEP PREDICTION:                                      |   ERROR')
        logger.error("Step prediction test failed due to unexpected exception", exc_info=e)
    finally:
        logger.info('')

def test_alignment(start_pvec, end_pvec):
    logger = logging.getLogger(__name__)
    logger.info('-----  ALIGNEMENT     -----')
    try:
        end_sequence = np.vstack([start_pvec, end_pvec])
        end_sequence = align_path(end_sequence, 'pairwise')
        start = check_vector(end_sequence[0], al_ref1)
        end = check_vector(end_sequence[1], al_ref2)
        logger.debug(list(end_sequence[0]))
        logger.debug(list(end_sequence[1]))
        logger.info(f'Start vector:                    RMSD:  {start[0]:+8.4e}   |   {bool2msg(start[1])}')
        logger.info(f'End vector:                      RMSD:  {end[0]:+8.4e}   |   {bool2msg(end[1])}')
        logger.info(f'ALIGNMENT:                                            |   {bool2msg((start[1] and end[1]))}')
    except Exception as e:
        logger.error(f'ALIGNEMENT:                                           |   ERROR')
        logger.error("Alignment test failed due to unexpected exception", exc_info=e)
    finally:
        logger.info('')


def test_interpolation(start_pvec, end_pvec, labels, mode):
    logger = logging.getLogger(__name__)
    if mode =='cartesian':
        logger.info('-----  CARTESIAN      -----')
        ref = cart_interp
    elif mode == 'internal':
        logger.info('-----  INTERNAL       -----')
        ref = int_interp
    elif mode == 'geodesic':
        logger.info('-----  GEODESIC       -----')
        ref = geo_interp
    try:
        path = do_interpolation(start_pvec, end_pvec, labels, 1, mode)
        logger.debug(list(path[1]))
        rmsd, check = check_vector(path[1], ref)
        if mode != 'geodesic':
            label = mode + ':'
            logger.info(f'{label.upper():12}                     RMSD:  {rmsd:+8.4e}   |   {bool2msg(check)}')
        else:
            logger.info('Geodesic code is non-deterministic.')
            label = ' '.join([mode, 'rmsd']).upper()
            logger.info(f'{label:12}:                                        |   {rmsd:+8.4e}')
    except (ImportError, AssertionError) as e:
        logger.warning(f'{mode.upper():12}:                                         |   NOT TESTED')
        logger.warning(e)
        logger.warning(f"{mode} test failed due to import problems")
    except Exception as e:
        logger.error(f'{mode.upper():12}:                                         |   ERROR')
        logger.error(f"{mode} test failed due to unexpected exception", exc_info=e)
    finally:
        logger.info('')


def test_idpp(start_pvec, end_pvec):
    logger = logging.getLogger(__name__)
    logger.info('-----  IDPP           -----')
    settings = Settings()
    try:
        path = [start_pvec, cart_interp, end_pvec]
        idpp_path = do_IDPP_opt_pass(path, settings)
        logger.debug(list(idpp_path[1]))
        rmsd, check = check_vector(idpp_path[1], idpp_interp)
        logger.info(f'IDPP:                            RMSD:  {rmsd:+8.4e}   |   {bool2msg(check)}')
    except Exception as e:
        logger.error(f'IDPP:                                                 |   ERROR')
        logger.error("IDPP test failed due to unexpected exception", exc_info=e)
    finally:
        logger.info('')


def test_nebgrad(labels, start_pvec, end_pvec):
    logger = logging.getLogger(__name__)
    try:
        path = [start_pvec, cart_interp, end_pvec]

        # Build settings
        settings = Settings()
        settings.interface = 'dummy'
        settings.k_const = 1
        settings.Relaxed_Max_RMSF_tol = 10
        settings.Relaxed_Max_AbsF_tol = 10
        settings.step_pred_method = 'sd'
        settings.stepsize_fac = 0.1
        settings.n_images = 1

        engrfunc, engrfunc_kwargs = gather_engrfunc(labels, settings, None)
        es, grads = engrfunc(path, **engrfunc_kwargs)
        error, check1 = check_value(es[1], dummy_img_energy)
        rmsd, check2 = check_vector(grads[1], dummy_image_grad)

        logger.info('-----  DUMMY ENGRADS  -----')
        logger.debug(es[1])
        logger.debug(list(grads[1]))
        logger.info(f'Dummy Energy:                    Error: {error:+8.4e}   |   {bool2msg(check1)}')
        logger.info(f'Dummy gradient:                  RMSD:  {rmsd:+8.4e}   |   {bool2msg(check2)}')

        path_obj = NEBPath(labels, path, engrfunc, engrfunc_kwargs, settings)
        path_obj.set_energies([es[1]])
        path_obj.set_engrads([grads[1]])
        optimizer = NEB_Optimizer(path_obj, settings, log=False)
        tanvecs = optimizer.calc_tanvecs()
        optimizer.path.set_tanvecs(tanvecs)

        logger.info('')
        logger.info('-----  TANGENTS       -----')
        logger.debug(list(tanvecs[0]))
        rmsd, check3 = check_vector(tanvecs, dummy_tangent)
        logger.info(f'Tangents:                        RMSD:  {rmsd:+8.4e}   |   {bool2msg(check3)}')

        springgrads = optimizer.calc_springgrads()
        optimizer.path.set_springgrads(springgrads)

        logger.info('')
        logger.info('-----  SPRINGS        -----')
        logger.debug(list(springgrads[0]))
        rmsd, check4 = check_vector(springgrads, dummy_springs)
        logger.info(f'Spring gradients:                RMSD:  {rmsd:+8.4e}   |   {bool2msg(check4)}')

        nebgrads, orth_grads = optimizer.calc_nebgrads()
        logger.info('')
        logger.info('-----  NEBGRADS       -----')
        logger.debug(list(nebgrads[0]))
        rmsd, check5 = check_vector(nebgrads, dummy_nebgrad)
        logger.info(f'NEB gradients:                   RMSD:  {rmsd:+8.4e}   |   {bool2msg(check5)}')
    except Exception as e:
        logger.error(f'NEB gradients:                                        |   ERROR')
        logger.error("NEB gradients test failed due to unexpected exception", exc_info=e)
    finally:
        logger.info('')


def test_orca(atomic_labels, pvec, temp_workdir):
    logger = logging.getLogger(__name__)
    logger.info('-----  ORCA           -----')
    logger.info('Reference energies and gradients calculated on HF-3c with Orca 6.1.1.\n' +
                'If you are using a different version the values might be slightly different.')
    try:
        nodes = os.getenv('SLURM_NTASKS_PER_NODE', os.getenv('NUM_THREADS', 1))
        settings = Settings()
        settings.interface = 'orca'
        settings.orca_keywords = 'HF-3c'
        settings.n_threads = nodes

        engrfunc, engrfunc_kwargs = gather_engrfunc(atomic_labels, 
                                                    settings, 
                                                    temp_workdir)
        energy, engrad = engrfunc([pvec], **engrfunc_kwargs)
        logger.debug(energy)
        logger.debug(list(engrad))
        error, close1 = check_value(energy[0], orca_energy)
        logger.info(f'ORCA Energy:                     Error: {error:+8.4e}   |   {bool2msg(close1)}')
        error, close2 = check_vector(engrad, orca_grad)
        logger.info(f'ORCA Gradient:                   Error: {error:+8.4e}   |   {bool2msg(close2)}')
        logger.info(f'ORCA INTERFACE:                                       |   {bool2msg((close1 and close2))}')
    except nex.MissingEnvironmentVariable:
        logger.warning(f'ORCA INTERFACE:                                       |   NOT TESTED')
        logger.warning('ORCA_EXE environment variable was not set.')
    except Exception as e:
        logger.error(f'ORCA INTERFACE:                                       |   ERROR')
        logger.error("ORCA test failed due to unexpected exception", exc_info=e)
    finally:
        logger.info('')

def test_molpro(atomic_labels, pvec, temp_workdir):
    logger = logging.getLogger(__name__)
    logger.info('-----  MOLPRO         -----')
    logger.info('Reference energies and gradients calculated on hf basis=3-21G with Molpro 2025.2.\n' + 
                'If you are using a different version the values might be slightly different.')
    try:
        nodes = os.getenv('SLURM_NTASKS_PER_NODE', os.getenv('NUM_THREADS', 1))
        settings = Settings()
        settings.interface = 'molpro'
        settings.molpro_keywords = 'hf basis=3-21G'
        settings.memory = 29000
        settings.n_threads = nodes

        engrfunc, engrfunc_kwargs = gather_engrfunc(atomic_labels, 
                                                    settings, 
                                                    temp_workdir)
        energy, engrad = engrfunc([pvec], **engrfunc_kwargs)
        logger.debug(energy)
        logger.debug(list(engrad))
        error, close1 = check_value(energy[0], molpro_energy)
        logger.info(f'Molpro Energy:                   Error: {error:+8.4e}   |   {bool2msg(close1)}')
        error, close2 = check_vector(engrad, molpro_grad)
        logger.info(f'Molpro Gradient:                 Error: {error:+8.4e}   |   {bool2msg(close2)}')
        logger.info(f'MOLPRO INTERFACE:                                     |   {bool2msg((close1 and close2))}')
    except nex.MissingEnvironmentVariable:
        logger.warning(f'MOLPRO INTERFACE:                                     |   NOT TESTED')
        logger.warning('MOL_EXE environment variable was not set.')
    except Exception as e:
        logger.error(f'MOLPRO INTERFACE:                                     |   ERROR')
        logger.error("MOLPRO test failed due to unexpected exception", exc_info=e)
    finally:
        logger.info('')


def test_gaussian(atomic_labels, pvec, temp_workdir):
    logger = logging.getLogger(__name__)
    logger.info('-----  GAUSSIAN       -----')
    logger.info('Reference energies and gradients calculated on HF/3-21G with Gaussian 16.\n' +
                'If you are using a different version the values might be slightly different.')
    try:
        nodes = os.getenv('SLURM_NTASKS_PER_NODE', os.getenv('NUM_THREADS', 1))
        settings = Settings()
        settings.interface = 'gaussian'
        settings.gaussian_keywords = 'HF/3-21G'
        settings.memory = 10000
        settings.n_threads = nodes

        engrfunc, engrfunc_kwargs = gather_engrfunc(atomic_labels, 
                                                    settings, 
                                                    temp_workdir)
        energy, engrad = engrfunc([pvec], **engrfunc_kwargs)
        logger.debug(energy)
        logger.debug(list(engrad))
        error, close1 = check_value(energy[0], gauss_energy)
        logger.info(f'Gaussian Energy:                 Error: {error:+8.4e}   |   {bool2msg(close1)}')
        error, close2 = check_vector(engrad, gauss_grad)
        logger.info(f'Gaussian Gradient:               Error: {error:+8.4e}   |   {bool2msg(close2)}')
        logger.info(f'GAUSSIAN INTERFACE:                                   |   {bool2msg((close1 and close2))}')
    except nex.MissingEnvironmentVariable:
        logger.warning(f'GAUSSIAN INTERFACE:                                   |   NOT TESTED')
        logger.warning('GAUSS_EXE environment variable was not set.')
    except Exception as e:
        logger.error(f'GAUSSIAN INTERFACE:                                   |   ERROR')
        logger.error("GAUSSIAN test failed due to unexpected exception", exc_info=e)
    finally:
        logger.info('')


def test_tblite(atomic_labels, pvec, temp_workdir):
    logger = logging.getLogger(__name__)
    logger.info('-----  xTB            -----')
    logger.info('Reference energies and gradients calculated on GFN2-xTB with tblite 0.4.0.\n' +
                'If you are using a different version the values might be slightly different.')
    try:
        nodes = os.getenv('SLURM_NTASKS_PER_NODE', os.getenv('NUM_THREADS', 1))

        settings = Settings()
        settings.interface = 'tblite'
        settings.tblite_keywords = 'GFN2-xTB'
        settings.n_threads = nodes

        engrfunc, engrfunc_kwargs = gather_engrfunc(atomic_labels, 
                                                    settings, 
                                                    temp_workdir)
        energy, engrad = engrfunc([pvec], **engrfunc_kwargs)
        logger.debug(energy)
        logger.debug(list(engrad))
        error, close1 = check_value(energy[0], tblite_energy)
        logger.info(f'tblite Energy:                   Error: {error:+8.4e}   |   {bool2msg(close1)}')
        error, close2 = check_vector(engrad, tblite_grad)
        logger.info(f'tblite Gradient:                 Error: {error:+8.4e}   |   {bool2msg(close2)}')
        logger.info(f'tblite INTERFACE:                                     |   {bool2msg((close1 and close2))}')
    except ImportError:
        logger.warning(f'tblite INTERFACE:                                     |   NOT TESTED')
        logger.warning('tblite module can not be imported.')
    except Exception as e:
        logger.error(f'tblite INTERFACE:                                     |   ERROR')
        logger.error("tblite test failed due to unexpected exception", exc_info=e)
    finally:
        logger.info('')




orca_energy = -192.689150787829
orca_grad = np.array([-2.772758859414e-05, -3.648387458620e-05,  2.497124540762e-05, 3.457342005607e-05,  7.933960330729e-06, -1.936203371388e-05, -4.187506290780e-05,  5.271933753020e-05,  1.505964700358e-05, -5.266929569271e-05, -4.605280895212e-05, -2.234146473855e-05, 9.133729305664e-05, -6.581567058450e-05, -1.271854089726e-05, -4.134254942434e-05,  1.405311839855e-05, -3.907575679785e-06, 9.893232155834e-06,  1.754604848242e-05,  1.152433603192e-05, -3.324401851462e-05,  3.521047696818e-05,  2.583058891400e-05, 1.142418810624e-06, -2.062539688619e-06, -3.958636079663e-06, 2.873521513004e-05,  1.138200374997e-05, -1.122942348324e-05, 7.649097345577e-06,  6.153063533947e-06, -1.057575776822e-06, 2.275582309609e-05, -7.161587689762e-06,  3.642959220328e-06, 7.720173724564e-07,  1.257847628665e-05, -6.453528097982e-06])

molpro_energy = -193.96674057043
molpro_grad = np.array([ 0.019230185632,  0.006069446932, -0.005572496205, -0.034448310738, -0.005387922875, -0.002975846214,  0.031838234243, -0.017055607861, 0.005786889413,  0.000831383119,  0.041478955525,  0.006256769814, -0.012094549552, -0.027336541898, -0.002804425378,  0.000783401083, 0.005163073812, -0.006612051553, -0.006294239304,  0.004816112537, 0.0032057125  ,  0.003254654517,  0.0035094444  ,  0.00416673839 , 0.002245504862, -0.002429282617,  0.002180834654,  0.000256343238, 0.000730777879, -0.002849113621,  0.002558562561, -0.003009540031, 0.002227278453, -0.003728682867, -0.003635268035, -0.001021090835, -0.004432486795, -0.002913647769, -0.001989201307])

gauss_energy = -192.87417982
gauss_grad = np.array([ 0.015216845761,  0.007079600032, -0.005152447882, -0.021064427508, -0.007336037757,  0.000235465544,  0.01922081826 , -0.013466927245, 0.003309637988, -0.006920719419,  0.01812000555 ,  0.000887848135, -0.005173446519, -0.009896267056,  0.000807051005, -0.001355224649, -0.00049093384 ,  0.001322793169,  0.000909494948, -0.001123060456, -0.001302782859, -0.00448917291 ,  0.002043774708, -0.001331397092, -0.010661016542,  0.001224408358, -0.001732060605,  0.003329121065, -0.010701970686,  0.0046283399  , -0.002146870607,  0.007722855246, -0.004947473069,  0.004235482847,  0.007009593238,  0.001482779273, 0.008899113382, -0.000185038203,  0.001792246492])

tblite_energy = -14.716031316322
tblite_grad = np.array([ 0.013056946638,  0.012220087955, -0.005280427646, -0.045486094608, -0.004415976567, -0.00396533549 ,  0.048171830287, -0.017066224935, 0.008731457911,  0.002643720379,  0.04463665936 ,  0.003902229154, -0.012008229343, -0.027093527389, -0.004254876433, -0.001469152115, -0.001400886207, -0.003490134853, -0.003917228229, -0.000101005297, 0.000828936619,  0.002423618986, -0.002015865917,  0.002440988244, -0.003957982075,  0.000637370233, -0.0011000525  ,  0.002200697216, -0.00447187544 ,  0.000990624562, -0.003151184629, -0.00153786792 , -0.000393965147,  0.002067671539, -0.002709050169,  0.001305264594, -0.000574614048,  0.003318162291,  0.000285290985])

al_ref1 = np.array([-1.7757155656499501, 0.41717269133289314, 0.17266747524715848, -0.8173707808878601, 1.2321597777582431, -0.22073781746393153, 0.63398094133639, 1.0705448794873031, 0.09524143724184846, 1.30326648975502, -0.06546852594889685, 0.14677116123953848, 0.74933453208716, -1.4556998606273366, -0.14194169482621152, 0.48335338422854, -1.9667267082266466, 0.7805349753182684, 1.49687849607354, -2.057645559534187, -0.6491974429248816, -0.13619559134603, -1.3955100024971667, -0.7638001310782916, -2.8114902613586503, 0.5951440197039032, -0.08170130993919154, -1.57684131056852, -0.45239809008876686, 0.7805632944246784, -1.06955360222952, 2.1148665955121233, -0.7948824993588016, 1.1649469943211601, 1.9947117584848932, 0.2795447144571985, 2.35540627423872, -0.031150975356356803, 0.3969378376626185])
al_ref2 = np.array([-1.6279869911685092, -0.0034342140936257474, -0.2282480169856277, -0.6592807481712007, 1.1333731232608644, 0.07512800164110181, 0.6576860526857882, 1.0554950746657201, 0.09599037709017776, 1.4791222895001332, -0.17065146412516238, -0.1365223618263192, 1.1708095727342684, -1.3826836662335489, 0.27962631677630023, 0.2748144419263416, -1.5850615121766884, 0.8466834333476819, 1.8148127389278228, -2.2295574136532226, 0.08651907565051031, -1.1389227162229463, -0.7856657703971301, -0.7969062822928528, -2.474237052368161, 0.3691347462263516, -0.7971755054459009, -2.013438680561835, -0.4401151013879017, 0.6902359768196382, -1.1206095098521114, 2.0925351026811234, 0.26998374956140325, 1.227973573814415, 1.955708975516411, 0.2814706324433776, 2.4092570287559942, -0.009077880283191845, -0.6667853967794906])

cart_interp = np.array([-2.3290553660841153, -0.48197652406706, -0.046441809777379994, -1.370069087462285, 0.49376515086576, -0.07712450429649001, 0.013740789796250001, 0.374952547111185, 0.10617660155606501, 0.7626650724038251, -0.804825386055705, 0.02162127832486499, 0.32780611436666496, -2.10738073672528, 0.078266010677535, -0.259399626794, -2.466394780672805, 0.812094997808105, 1.02640354037189, -2.83073944338433, -0.26693164950701, -1.2576657524666501, -1.777415515168625, -0.7955601590175501, -3.263738049870875, -0.20570502184803002, -0.46664835796135506, -2.432466512553545, -1.13804174994991, 0.710944987140795, -1.729785050275455, 1.413780279402355, -0.26933536901144506, 0.561426921936765, 1.287029445184305, 0.30016760727834, 1.7596270066315451, -0.704648264691855, -0.10721963321446998])
int_interp = np.array([-2.3219757104963974, -0.5591144977663249, 0.018592572454829537, -1.3772173776688736, 0.4993915558085577, -0.06184449001736525, 0.013937864386572804, 0.37523827470971577, 0.10634396090062156, 0.7628945275677343, -0.8130782538084227, 0.028625103623740378, 0.24275281498443435, -2.1319342586600465, 0.12637034093746569, -0.41901534889660674, -2.330688837930741, 0.9610144963705431, 0.8816636858904748, -2.9214123982810167, -0.2509358073733786, -1.2469110678695037, -1.8608133602204413, -0.7392955114049522, -3.2895396352940898, -0.3641817876347593, -0.42832179842578505, -2.2521779923540413, -1.205155110095627, 0.8856577925055975, -1.7575344040278806, 1.4919664927076925, -0.2661750253623387, 0.5646228326093997, 1.2861517343784876, 0.298643708938846, 1.830699403318287, -0.7082109862459172, -0.11417832067107736])
geo_interp = np.array([-1.6453555455267204, 0.13088685566843467, -0.11253942550095329, -0.7239618043991846, 1.2146241709791659, -0.10335261562670689, 0.6665222042859739, 1.0978421849636455, 0.0665996957675367, 1.410097262739083, -0.09462526941289985, -0.009802271081407861, 0.8685972598378455, -1.4070180496934577, -0.049694295695590654, 0.28678208282247697, -1.80702611367538, 0.7734866694598156, 1.5433508869787882, -2.1610794144266134, -0.4382415222134312, -0.6288068792210135, -1.0436375867341223, -0.2877921159723972, -2.6074487115407803, 0.3652779765910123, -0.5566441429876995, -1.7292758923241793, -0.5065226066343178, 0.7663678793137363, -1.1229968543558269, 2.2069200296188027, -0.2812339967766353, 1.2021372549678522, 1.988309153435456, 0.367822993026921, 2.480358735735685, 0.016048669320274586, -0.13497685171318763])
idpp_interp = np.array([-2.3390754432752754, -0.4907632746368999, -0.07376902886106686, -1.3493016103012279, 0.49226193230608783, -0.06955843123127439, 0.019877372299112843, 0.38465430069987955, 0.1107041268543626, 0.7519168567955918, -0.7892744180420642, 0.027552470609187876, 0.3285472828451333, -2.118927847329806, 0.05099826670820064, -0.2906793315873184, -2.4825412534569127, 0.8530061312992532, 1.0403592161675848, -2.861224862870846, -0.2790507829219729, -1.2651359934392747, -1.7907820852036314, -0.803550684611806, -3.2987785096855466, -0.20884898047267692, -0.481419013767816, -2.4356352596725914, -1.1721343138591707, 0.7537135547035368, -1.7364298566888559, 1.4788675740318797, -0.27862359967986566, 0.5660241434278211, 1.2952237693151936, 0.302884126888213, 1.8178011331148614, -0.6841105404810288, -0.11287713598894708])

dummy_img_energy = 0.91
dummy_image_grad = np.array([-0.5822638415210288, -0.120494131016765, -0.011610452444344999, -0.34251727186557124, 0.12344128771644, -0.019281126074122502, 0.0034351974490625004, 0.09373813677779624, 0.026544150389016252, 0.19066626810095627, -0.20120634651392624, 0.0054053195812162474, 0.08195152859166624, -0.52684518418132, 0.01956650266938375, -0.0648499066985, -0.6165986951682012, 0.20302374945202625, 0.2566008850929725, -0.7076848608460825, -0.0667329123767525, -0.31441643811666253, -0.44435387879215626, -0.19889003975438752, -0.8159345124677188, -0.051426255462007506, -0.11666208949033877, -0.6081166281383863, -0.2845104374874775, 0.17773624678519875, -0.43244626256886376, 0.35344506985058877, -0.06733384225286126, 0.14035673048419126, 0.32175736129607624, 0.075041901819585, 0.4399067516578863, -0.17616206617296376, -0.026804908303617495])

dummy_tangent = np.array([0.060743663060232524, -0.16700685533591972, -0.1735291550775079, 0.061251712174913736, -0.03969273512714736, 0.11373921403972001, 0.007760047340178791, -0.0057942534734523165, 0.008661020322145703, 0.07083218401722717, -0.04045480279970871, -0.09911536243852605, 0.16513539694167562, 0.028982660050742567, 0.17440029851488617, -0.08926745939589126, 0.14937347773801848, 0.02499545716428751, 0.1263708087691494, -0.06717378815635719, 0.30274676176090387, -0.38920316400235816, 0.2426388509410496, -0.025152638680458646, 0.14080632676067478, -0.08915526636402245, -0.3048690383708077, -0.17865976739683884, 0.00208487769318545, -0.05513556838796675, -0.023912185333956393, -0.010145368499064467, 0.4162224486627829, 0.021001984552041286, -0.015369254094154435, 0.016333489538309244, 0.027132532731534383, 0.011704537645413756, -0.3992810874849469])
dummy_springs = np.array([1.3487802665908549e-17, -3.708297121283615e-17, -3.8531212682159734e-17, 1.360061223086045e-17, -8.813557689701348e-18, 2.5255178845933204e-17, 1.7230766458495398e-18, -1.28658272334821e-18, 1.92313283567851e-18, 1.5727904316082324e-17, -8.98277070498137e-18, -2.2008031494663804e-17, 3.6667423973052586e-17, 6.435443300643621e-18, 3.8724645382545424e-17, -1.9821357754221948e-17, 3.3167574850616274e-17, 5.5501064109647634e-18, 2.8059956307202463e-17, -1.491557725249608e-17, 6.722328510753248e-17, -8.642046278647578e-17, 5.387664779666892e-17, -5.585007718624501e-18, 3.1265285196518896e-17, -1.9796445896785297e-17, -6.76945251789202e-17, -3.967043746762907e-17, 4.629358437003739e-19, -1.2242555500023123e-17, -5.309571745372475e-18, -2.2527243401936274e-18, 9.241994917425676e-17, 4.663377362499618e-18, -3.4126599533289414e-18, 3.626763231581008e-18, 6.0246325109890325e-18, 2.5989294373060537e-18, -8.86582113246319e-17])
dummy_nebgrad = np.array([-0.42483447660320706, 0.05179626374524236, -0.011380840950301374, -0.18508857974995493, 0.295563082202613, -0.019431939989234048, 0.16093472781444396, 0.2658150399875907, 0.026532490029032046, 0.34808227293418326, -0.029083542832389925, 0.005536385949674117, 0.2392426490233433, -0.354814335542717, 0.01933535614420387, 0.09277811581191922, -0.4447272783745242, 0.202990457630012, 0.4139433409165745, -0.5355266735928437, -0.0671340262949455, -0.15639121502514283, -0.27260597200366987, -0.19885692116742176, -0.6586111733949627, 0.12076104154970098, -0.11625854645975393, -0.4503702246678527, -0.11244396849885072, 0.17780907133887677, -0.27490478911691507, 0.5255277351800312, -0.06788523040678536, 0.2978387247407511, 0.4938469445427992, 0.07502008091791025, 0.5973806273168202, -0.004108336362982051, -0.02627633674126707])

dummy_sd = np.array([-2.307813642253955, -0.4845663372543221, -0.04587276772986492, -1.360814658474787, 0.47898699675562934, -0.07615290729702831, 0.005694053405527803, 0.3616617951118054, 0.10484997705461341, 0.745260958757116, -0.8033712089140854, 0.021344459027381288, 0.3158439819154978, -2.089640019948144, 0.0772992428703248, -0.264038532584596, -2.4441584167540786, 0.8019454749266044, 1.0057063733260614, -2.803963109704688, -0.26357494819226274, -1.249846191715393, -1.7637852165684416, -0.785617312959179, -3.230807491201127, -0.21174307392551509, -0.46083543063836735, -2.4099480013201524, -1.1324195515249675, 0.7020545335738512, -1.7160398108196093, 1.3875038926433536, -0.2659411074911058, 0.5465349856997275, 1.262337097957165, 0.2964166032324445, 1.7297579752657042, -0.7044428478737059, -0.1059058163774066])
dummy_amgd = np.array([-2.283555893598752, -0.48730284232512566, -0.04498035980299559, -1.350294407613844, 0.46210699643917574, -0.07519702140238406, -0.003537820547061839, 0.34643933262817067, 0.10331774416332468, 0.7252186880031496, -0.8016487765814801, 0.02116450224955033, 0.3019073311334596, -2.069350252052586, 0.07594985960308553, -0.26923084608852177, -2.4188836045881037, 0.7902800547090248, 0.981813527173874, -2.7731858476889673, -0.2601476114320614, -1.2403464027247995, -1.748501651791731, -0.7741885121984774, -3.1932658435155163, -0.21853888908390434, -0.4537519244420681, -2.3838955812959526, -1.1259797030811376, 0.6919429097413308, -1.700255392428219, 1.3574066178878887, -0.26262787742917426, 0.5294405431140392, 1.2340622815674287, 0.29209553662323134, 1.6954920983881603, -0.7042236613296236, -0.10384730038238185])
dummy_rfo = np.array([-2.2756222424883665, -0.48824293748030545, -0.04473799245536759, -1.3468438614328733, 0.45658701075143715, -0.07485303516305482, -0.006548467481067263, 0.3414698330529266, 0.10282013706045832, 0.7186978186657891, -0.8010980518623465, 0.021077859157098127, 0.2974055962245197, -2.0627205007548017, 0.07555860248603026, -0.2709505035334277, -2.410593073017903, 0.7864800706444586, 0.9740516594069893, -2.763160571109508, -0.25894383459836356, -1.2373557460909634, -1.7434455090262213, -0.7704657941154356, -3.1809744289876742, -0.22078181951854917, -0.451526074115731, -2.3754436598287496, -1.1238774638948938, 0.6886274417603654, -1.6951108637226806, 1.3475814810354, -0.26142938152774736, 0.5238676672740105, 1.2248304348809298, 0.29068995124987573, 1.6843170319945102, -0.7041488330561598, -0.1032879503825818])
dummy_sct = np.array([-2.2761219570195457, -0.48829830334481594, -0.04487900662399513, -1.3470362253401293, 0.4569365404068239, -0.07479504222933031, -0.006336745335688779, 0.34180597198822665, 0.10286042024746724, 0.7191951292357912, -0.8011647682794047, 0.021012727578571595, 0.2978322574618745, -2.063153701067177, 0.07571043491765964, -0.270896750301733, -2.411053679619444, 0.7867581997911934, 0.9746737690029563, -2.763895226593352, -0.25880920235586496, -1.2378395854690043, -1.7436177673253952, -0.7707387468180648, -3.1817151436571613, -0.22069215405726197, -0.45189707785551403, -2.3761505109851475, -1.124019922169131, 0.6888149407286212, -1.695480286041998, 1.3482469963031944, -0.26121302790376644, 0.5242643355317743, 1.2254515784233857, 0.2907979104187306, 1.6851017129180286, -0.7041455646656367, -0.10361252989570284])


if __name__ == '__main__':
    main()