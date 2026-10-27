# the following section contains the helper 
# function to gather the engrad function and its arguments,
# depending on which interface was chosen by the user.
# --------------------------------------------------------

import os
import logging
from pathlib import Path
import codecs

import neb_exceptions as nex
from . import orca_interface as orc
from . import gaussian_interface as gss
from . import molpro_interface as mol
from . import tblite_interface as tbl
from . import mace_interface as mac
from . import mace_ensemble_interface as mace_ens
from . import uma_interface as uma
from . import dummy_interface as dmm

logger = logging.getLogger(__name__)

def gather_engrfunc(atomic_labels, settings, temp_workdir):
    """Set up the appropriate energy function from the selected 
    interface (orca, gaussian, molpro or dummy interface).\\
    Returns:
    - energy function
    - energy keywords to be used with the function
    """
    interface = settings.interface


    if interface is None:
        # user forgot to set this
        raise nex.MissingKeyword('Error: no interface for engrad calculation '
                                 'selected. Make sure to select an interface '
                                  'option in the input file.')

    elif interface == 'orca':
        engrfunc, engrfunc_kwargs = setup_orca(atomic_labels,
                                               settings,
                                               temp_workdir)

    elif interface == 'gaussian':
        engrfunc, engrfunc_kwargs = setup_gaussian(atomic_labels,
                                                   settings,
                                                   temp_workdir)

    elif interface == 'molpro':
        engrfunc, engrfunc_kwargs = setup_molpro(atomic_labels,
                                                 settings,
                                                 temp_workdir)
        
    elif interface == 'tblite':
        engrfunc, engrfunc_kwargs = setup_tblite(atomic_labels,
                                                 settings)

    elif interface == 'mace':
        engrfunc, engrfunc_kwargs = setup_mace(atomic_labels,
                                               settings)

    elif interface == 'mace_ensemble':
        engrfunc, engrfunc_kwargs = setup_mace_ensemble(atomic_labels,
                                                        settings)

    elif interface == 'uma':
        engrfunc, engrfunc_kwargs = setup_uma(atomic_labels,
                                              settings)

    elif interface == 'dummy':
        number_of_images = settings.n_images + 2
        engrfunc, engrfunc_kwargs = setup_dummy(number_of_images)

    else:
        raise ValueError(f'Error: invalid interface selected in input file: {str(interface)}')

    return engrfunc, engrfunc_kwargs


# helper functions for setting up ORCA interface
def setup_orca(atomic_labels, settings, temp_workdir):
    """Set up the orca energy and gradient calculation function with the specified user input.\\
    Returns: 
    - energy function
    - energy keywords to be used with the function
    """
    logger = logging.getLogger(__name__)
    logger.info('Orca interface for EnGrad calculation selected.')
    logger.info('Temporary files stored at: \n' + str(temp_workdir))

    engrfunc = orc.orca_path_engrads
    orca_keywords = settings.orca_keywords

    if orca_keywords is None:
        raise nex.MissingKeyword('Error: ORCA interface selected, but no '
                                 'ORCA calculation keywords specified. Make '
                                 'sure to specify them in the .ini file under '
                                 'the "orca_keywords" keyword.')

    if settings.orca_keywords2 is None:
        ml2 = None
    else:
        ml2 = codecs.decode(settings.orca_keywords2, 'unicode_escape') 

    engrfunc_kwargs = {'labels' : atomic_labels,
                       'npal' : settings.n_threads,
                       'workingdir' : temp_workdir,
                       'orca' : find_orcapath(settings),
                       'method_keywords' : orca_keywords,
                       'charge' : settings.charge,
                       'spin' : settings.spin,
                       'series_name' : 'image',
                       'method_line2' : ml2}

    return engrfunc, engrfunc_kwargs


def find_orcapath(settings):
    """Get the ORCA Path. Either from the ini file or from the environment variable ORCA_EXE."""
    logger = logging.getLogger(__name__)
    if settings.orca_path is not None:
        logger.info('Using orca path from .ini file.')
        orcapath = settings.orca_path

    else:
        orcapath = os.getenv('ORCA_EXE')

        if orcapath is None or orcapath == '':
            raise nex.MissingEnvironmentVariable(
            "ORCA_EXE environment variable is not set.\n"
            "Please set the ORCA_EXE environment variable before running this program,\n"
            "or specify the 'orca_path' option in the configuration file."
            )


    return orcapath


# helper functions for setting up the Gaussian interface
def setup_gaussian(atomic_labels, settings, temp_workdir):
    """Set up the gaussian energy and gradient calculation function with the specified user input.\\
    Returns: 
    - energy function
    - energy keywords to be used with the function
    """
    logger = logging.getLogger(__name__)
    logger.info('Gaussian interface for EnGrad calculation selected.')
    logger.info('Temporary files stored at: \n' + str(temp_workdir))
    logger.info('Memory allocated: ' + str(settings.memory))

    engrfunc = gss.gaussian_path_engrads
    gauss_keywords = settings.gaussian_keywords

    if gauss_keywords is None:
        raise nex.MissingKeyword('Error: Gaussian interface selected, but no '
                                 'Gaussian calculation keywords specified. Make '
                                 'sure to specify them in the .ini file under '
                                 'the "gaussian_keywords" keyword.')

    engrfunc_kwargs = {'labels' : atomic_labels,
                       'workingdir' : temp_workdir,
                       'series_name' : 'image',
                       'gaussian' : find_gausspath(settings),
                       'method_keywords' : gauss_keywords,
                       'charge' : settings.charge,
                       'spin' : settings.spin,
                       'nprocs' : settings.n_threads,
                       'mem' : settings.memory,
                       'title' : 'Title Card Required'}

    return engrfunc, engrfunc_kwargs


def find_gausspath(settings):
    """Get the gaussian Path. Either from the ini file or from the environment variable GAUSS_EXE."""
    logger = logging.getLogger(__name__)
    if settings.gaussian_path is not None:
        logger.info('Using gaussian path from .ini file.')
        gaussianpath = settings.gaussian_path

    else:
        gaussianpath = os.getenv('GAUSS_EXE')
        print('path:')
        print(gaussianpath)

        if gaussianpath is None or gaussianpath == '':
            raise nex.MissingEnvironmentVariable(
            "GAUSS_EXE environment variable is not set.\n"
            "Please set the GAUSS_EXE environment variable before running this program,\n"
            "or specify the 'gaussian_path' option in the configuration file."
            )

    return gaussianpath


# helper functions for setting up MolPro interface 

def setup_molpro(atomic_labels, settings, temp_workdir):
    """Set up the Molpro energy and gradient calculation function with the specified user input.\\
    Returns: 
    - energy function
    - energy keywords to be used with the function
    """
    logger = logging.getLogger(__name__)
    os.environ['TMPDIR'] = str(temp_workdir)
    os.environ['TMPDIR4'] = str(temp_workdir)

    logger.info('Molpro interface for EnGrad calculation selected.')
    logger.info('Temporary files stored at: \n' + str(temp_workdir))
    logger.info('Memory allocated: ' + str(settings.memory))

    engrfunc = mol.molpro_path_engrads
    molpro_keywords = settings.molpro_keywords

    if molpro_keywords is None:
        raise nex.MissingKeyword('Error: Molpro interface selected, but no '
                                 'Molpro calculation keywords specified. Make '
                                 'sure to specify them in the .ini file under '
                                 'the "molpro_keywords" keyword.')

    engrfunc_kwargs = {'labels' : atomic_labels,
                       'workingdir' : temp_workdir,
                       'series_name' : 'image',
                       'molpro' : find_molpro_path(settings),
                       'method_keywords' : molpro_keywords,
                       'charge' : settings.charge,
                       'spin' : settings.spin,
                       'nprocs' : settings.n_threads,
                       'mem' : settings.memory}
                       #'errordir' : find_workdir()}

    return engrfunc, engrfunc_kwargs


def find_molpro_path(settings):
    """Get the Molpro Path. Either from the ini file or from the environment variable MOL_EXE."""
    logger = logging.getLogger(__name__)
    if settings.molpro_path is not None:
        logger.info('Using molpro path from .ini file.')
        molpath = Path(settings.molpro_path)

    else:
        molpath = os.getenv('MOL_EXE')
        if molpath is None or molpath == '':
            raise nex.MissingEnvironmentVariable(
            "MOL_EXE environment variable is not set.\n"
            "Please set the MOL_EXE environment variable before running this program,\n"
            "or specify the 'molpro_path' option in the configuration file."
            )

        molpath = Path(molpath)
    return molpath


# helper functions for setting up tblite interface
def setup_tblite(atomic_labels, settings):
    """Set up the tblite energy and gradient calculation function with the specified user input.\\
    Returns: 
    - energy function
    - energy keywords to be used with the function
    """
    logger = logging.getLogger(__name__)
    logger.info('Tblite interface for EnGrad calculation selected.')

    npal = settings.n_threads
    os.environ["OMP_NUM_THREADS"] = str(npal)

    engrfunc = tbl.tblite_path_engrads
    tblite_keywords = settings.tblite_keywords

    if tblite_keywords is None:
        raise nex.MissingKeyword('Error: tblite interface selected, but no '
                                 'tblite calculation keywords specified. Make '
                                 'sure to specify them in the .ini file under '
                                 'the "tblite_keywords" keyword.')

    engrfunc_kwargs = {'labels' : atomic_labels,
                       'method_keywords' : tblite_keywords,
                       'charge' : settings.charge,
                       'spin' : settings.spin,
                       'series_name' : 'image'}

    return engrfunc, engrfunc_kwargs


# helper functions for setting up MACE interface
def setup_mace(atomic_labels, settings):
    """Set up the MACE energy and gradient calculation function with the specified user input.
    Supports both MACE foundation models (mace-off, mace-mp) and custom models loaded
    from a local checkpoint path. The calculator is instantiated once here and reused
    for every image and every iteration of the NEB run.\\
    Returns:
    - energy function
    - energy keywords to be used with the function
    """
    logger = logging.getLogger(__name__)
    logger.info('MACE interface for EnGrad calculation selected.')

    if settings.charge != 0 or settings.spin != 1:
        logger.warning('MACE interface selected, but charge/spin were set to '
                       'non-default values (charge=%s, spin=%s). These are '
                       'not used by MACE foundation or custom models and will '
                       'be ignored.', settings.charge, settings.spin)

    calculator = build_mace_calculator(settings)

    engrfunc = mac.mace_path_engrads
    engrfunc_kwargs = {'labels' : atomic_labels,
                       'series_name' : 'image',
                       'calculator' : calculator}

    return engrfunc, engrfunc_kwargs


def build_mace_calculator(settings):
    """Instantiate the MACE ASE calculator, either from a custom model path
    (takes priority if set) or from a foundation model family."""
    logger = logging.getLogger(__name__)

    if settings.mace_model_path is not None:
        logger.info('Loading custom MACE model from: %s', settings.mace_model_path)
        from mace.calculators import MACECalculator
        calculator = MACECalculator(model_paths=str(settings.mace_model_path),
                                    device=settings.mace_device,
                                    default_dtype=settings.mace_dtype)

    elif settings.mace_foundation is not None:
        logger.info('Loading MACE foundation model: %s (size=%s, device=%s)',
                    settings.mace_foundation, settings.mace_size, settings.mace_device)

        if settings.mace_foundation == 'mace_off':
            from mace.calculators import mace_off
            calculator = mace_off(model=settings.mace_size,
                                  device=settings.mace_device,
                                  default_dtype=settings.mace_dtype)

        elif settings.mace_foundation == 'mace_mp':
            from mace.calculators import mace_mp
            calculator = mace_mp(model=settings.mace_size,
                                 device=settings.mace_device,
                                 default_dtype=settings.mace_dtype)

        else:
            raise ValueError(f'Error: invalid mace_foundation selected: '
                             f'{str(settings.mace_foundation)}. Must be '
                             f"'mace_off' or 'mace_mp'.")

    else:
        raise nex.MissingKeyword('Error: MACE interface selected, but neither '
                                 'mace_model_path nor mace_foundation was set. '
                                 'Specify one of them in the .ini file.')

    return calculator


def setup_mace_ensemble(atomic_labels, settings):
    """Set up the ensemble-MACE energy and gradient calculation function,
    for uncertainty-quantification-driven active learning. N MACE
    calculators are instantiated once here and reused for every image and
    iteration; the NEB path itself is driven by the *mean* energy/force
    across all N models (see mace_ensemble_interface.py's module docstring
    for why a single shared path rather than N independent ones), and the
    run stops early -- via goeneb's existing giveup_signal_func() hook,
    the same mechanism used for too-many-failed-images -- as soon as the
    per-image energy disagreement across models exceeds
    mace_ensemble_uncertainty_threshold.\\
    Returns:
    - energy function
    - energy keywords to be used with the function
    """
    logger = logging.getLogger(__name__)
    logger.info('Ensemble MACE interface for EnGrad calculation selected '
               '(active-learning uncertainty quantification mode).')

    if settings.charge != 0 or settings.spin != 1:
        logger.warning('Ensemble MACE interface selected, but charge/spin were set to '
                       'non-default values (charge=%s, spin=%s). These are '
                       'not used by MACE models and will be ignored.',
                       settings.charge, settings.spin)

    if settings.mace_ensemble_model_paths is None:
        raise nex.MissingKeyword('Error: mace_ensemble interface selected, but '
                                 'mace_ensemble_model_paths was not set. Give a '
                                 'comma-separated list of checkpoint paths in the .ini file.')

    if settings.mace_ensemble_uncertainty_threshold is None:
        raise nex.MissingKeyword('Error: mace_ensemble interface selected, but '
                                 'mace_ensemble_uncertainty_threshold was not set '
                                 '(in eV). Specify it in the .ini file.')

    model_paths = [p.strip() for p in settings.mace_ensemble_model_paths.split(',') if p.strip()]
    if len(model_paths) < 2:
        raise nex.MissingKeyword('Error: mace_ensemble_model_paths must list at least 2 '
                                 f'checkpoint paths to compute a meaningful disagreement '
                                 f'(got {len(model_paths)}).')

    logger.info('Loading %d ensemble MACE models.', len(model_paths))
    calculators = mace_ens.build_mace_ensemble_calculators(
        model_paths, device=settings.mace_ensemble_device, dtype=settings.mace_ensemble_dtype)

    if settings.n_images == 2:
        logger.warning('n_images = 2 with the ensemble MACE interface: NEBPath\'s '
                       'one-off sanity check of the two end structures also calls '
                       'this interface with exactly 2 images, which is otherwise '
                       'used to distinguish that pre-flight call from a real NEB '
                       'iteration. With n_images = 2 they are indistinguishable, so '
                       'the uncertainty check will run one extra time (on the fixed '
                       'endpoints) before the first real iteration.')

    batch_images = getattr(settings, 'mace_ensemble_batch_images', False)
    max_batch_atoms = getattr(settings, 'mace_ensemble_max_batch_atoms', 0) or None
    if batch_images:
        logger.info('Ensemble MACE images will be batched into one forward pass per '
                    'model per iteration%s.',
                    f' (max {max_batch_atoms} atoms per batch)' if max_batch_atoms else '')

    metric = getattr(settings, 'mace_ensemble_uncertainty_metric', 'energy')
    logger.info('Ensemble uncertainty metric: %s (threshold %.4f %s).', metric,
                settings.mace_ensemble_uncertainty_threshold,
                'eV' if metric == 'energy' else 'eV/Angstrom')

    resultdir = getattr(settings, 'resultdir', None)
    driver = mace_ens.EnsembleUncertaintyDriver(
        calculators, settings.mace_ensemble_uncertainty_threshold, resultdir=resultdir,
        n_images=settings.n_images, batch_images=batch_images,
        max_batch_atoms=max_batch_atoms, uncertainty_metric=metric)

    engrfunc = mace_ens.ensemble_mace_path_engrads
    engrfunc_kwargs = {'labels' : atomic_labels,
                       'series_name' : 'image',
                       'driver' : driver}

    return engrfunc, engrfunc_kwargs


# helper functions for setting up UMA interface
def setup_uma(atomic_labels, settings):
    """Set up the UMA (fairchem-core) energy and gradient calculation function
    with the specified user input. The predictor is instantiated once here
    and reused for every image and every iteration of the NEB run.\\
    Returns:
    - energy function
    - energy keywords to be used with the function
    """
    logger = logging.getLogger(__name__)
    logger.info('UMA interface for EnGrad calculation selected.')
    logger.info('Loading UMA model: %s (task=%s, device=%s)',
               settings.uma_model, settings.uma_task, settings.uma_device)

    from fairchem.core import pretrained_mlip
    predictor = pretrained_mlip.get_predict_unit(settings.uma_model, device=settings.uma_device)

    if settings.uma_batch_images:
        logger.info('UMA images will be batched into a single predictor call per iteration.')

    engrfunc = uma.uma_path_engrads
    engrfunc_kwargs = {'labels' : atomic_labels,
                       'series_name' : 'image',
                       'predictor' : predictor,
                       'task_name' : settings.uma_task,
                       'charge' : settings.charge,
                       'spin' : settings.spin,
                       'batch' : settings.uma_batch_images}

    return engrfunc, engrfunc_kwargs


# helper function for setting up the dummy interface
def setup_dummy(number_of_images):
    engrfunc = dmm.dummy_path_engrads
    engrfunc_kwargs = {'iteration' : dmm.IterationState(), 
                       'number_of_images' : number_of_images}
    return engrfunc, engrfunc_kwargs