import json
import time
import logging

import numpy as np

from . import mace_interface as mac

logger = logging.getLogger(__name__)

eV_in_Eh = 27.211386245988  # Hartree per eV

# Ensemble MACE interface for active-learning uncertainty quantification.
#
# Design (see conversation for the reasoning): a single shared NEB path is
# driven by the *mean* energy/force across all N ensemble models -- there is
# only ever one set of image geometries, evaluated by every model each
# iteration. This avoids the failure mode of running N independent
# per-model NEB paths (which would let climbing image, or even just
# ordinary force differences, drive each model's path to a different
# geometry near the barrier, making per-model energies incomparable).
#
# Per iteration, each model's own per-image energies are de-meaned (each
# model's own path-mean energy subtracted off) before comparing across
# models, so a constant reference-energy offset between independently
# trained MACE models isn't mistaken for genuine predictive disagreement.
# The per-image standard deviation of these de-meaned energies across
# models is the per-image uncertainty; the run stops as soon as the *max*
# over all images exceeds the threshold (most sensitive to a single
# problematic region of the path, which is the point -- catch it early
# rather than average it away).
#
# The actual early-stop is wired in via goeneb's existing
# giveup_signal_func() mechanism (see neb_optimizer.py) -- this module only
# needs to compute the (mean_energies, mean_engrads) that goeneb's optimizer
# will actually use to move the path, and set self.triggered/trigger_message
# on the driver object when the threshold is crossed. basic_neb.py stashes
# engrad_calc_kwargs (which includes this driver) onto the optimizer so
# giveup_signal_func() can see it.
# ---------------------------------------------------------


class EnsembleUncertaintyDriver:
    """Holds the N loaded MACE calculators plus the running uncertainty
    state that giveup_signal_func() checks every iteration.

    - calculators: list of N pre-loaded ASE-compatible MACE calculators.
    - threshold_eV: stop the NEB as soon as the max per-image energy
      disagreement (std across models, after de-meaning each model's own
      path) exceeds this, in eV.
    - resultdir: if given, a small JSON marker is written here
      (ensemble_uncertainty.json) when triggered, so a driving Python
      script can tell *why* a run stopped without parsing the free-text
      log -- mirrors how barrier_from_results() reads optlog.csv rather
      than grepping output.log.
    """

    def __init__(self, calculators, threshold_eV, resultdir=None, n_images=None,
                 batch_images=False, max_batch_atoms=None, uncertainty_metric='energy'):
        self.calculators = calculators
        self.n_models = len(calculators)
        self.threshold_eV = threshold_eV
        self.resultdir = resultdir

        # 'energy' -- std across models of the per-image energy, after each
        # model's own path mean is subtracted. Threshold in eV.
        #
        # 'force'  -- per-atom force disagreement: for each atom, the RMS over
        # models of |F_m - Fbar|, then the worst atom in the image. Threshold in
        # eV/Angstrom.
        #
        # Two reasons the force metric is better suited to a NEB. It needs no
        # de-meaning: forces have no arbitrary per-model reference, so the whole
        # awkward subtract-the-path-mean step disappears along with the
        # possibility of it hiding real disagreement. And it is intensive --
        # a per-atom quantity, so one threshold means the same thing for a
        # 24-atom and a 152-atom complex, where a total-energy threshold quietly
        # gets stricter as the system grows. It is also the quantity the
        # optimiser actually moves the path with.
        if uncertainty_metric not in ('energy', 'force'):
            raise ValueError(f"uncertainty_metric must be 'energy' or 'force', "
                             f"got {uncertainty_metric!r}")
        self.uncertainty_metric = uncertainty_metric
        self.units = 'eV' if uncertainty_metric == 'energy' else 'eV/A'

        # Batch every image of the path into one forward pass per model, so an
        # iteration costs n_models calls instead of n_models x n_images. The
        # models cannot be batched together (different weights), only the images
        # within each. max_batch_atoms splits the path across several forwards
        # if one batch would not fit in memory.
        self.batch_images = batch_images
        self.max_batch_atoms = max_batch_atoms
        # Set on a model whose batched call raised, so the warning is logged
        # once rather than every iteration.
        self._batch_fallback_warned = set()

        # Expected number of *intermediate* path images (settings.n_images).
        # NEBPath.__init__() makes a one-off sanity-check call to the engrad
        # function with exactly the 2 fixed endpoint structures, before the
        # real optimization loop ever starts -- without this, that call
        # looked identical to a genuine (short, 2-image) NEB iteration, and
        # the uncertainty check could trigger on the endpoints alone before
        # any real NEB iteration ran. ensemble_mace_path_engrads() only
        # increments iteration_count / checks the threshold when the number
        # of images passed in matches this (i.e. it's a real do_opt_loop
        # call, not the endpoint sanity check). If n_images is None (or
        # equals 2, an unavoidable ambiguity -- see engrad_interface.py's
        # warning), the check runs unconditionally.
        self.expected_n_images = n_images

        self.triggered = False
        self.trigger_message = None
        self.trigger_info = None
        self.iteration_count = 0

        # Highest disagreement seen so far, and the iteration it happened on.
        # Recorded whether or not the threshold is ever crossed, so a run that
        # finishes at maxiter still says how close it came -- otherwise "did not
        # trigger" is a single bit, and there is no way to calibrate a threshold
        # from a run that never hit one.
        self.max_uncertainty_seen = float('nan')
        self.max_uncertainty_seen_iteration = None

        # last-iteration diagnostics, kept around for inspection/debugging
        self.last_energies_per_model_Eh = None   # (n_models, n_images)
        self.last_uncertainty_per_image_eV = None  # (n_images,)

    def _write_marker(self, extra=None):
        """Write ensemble_uncertainty.json.

        Written whether or not the threshold was crossed, so a driving script
        can read the running maximum off a run that finished at maxiter. The
        'triggered' field, not the file's existence, says which happened --
        older marker files always carried triggered=True, so reading it with a
        default of True stays compatible with them.

        Only written when the running maximum improves (and on the trigger), so
        a 500-iteration NEB does a handful of writes rather than 500.
        """
        if self.resultdir is None:
            return
        info = {
            'triggered': self.triggered,
            'metric': self.uncertainty_metric,
            'units': self.units,
            'max_uncertainty_seen': self.max_uncertainty_seen,
            'max_uncertainty_seen_iteration': self.max_uncertainty_seen_iteration,
            'threshold': self.threshold_eV,
            'n_models': self.n_models,
            'iterations_run': self.iteration_count,
        }
        info.update(extra or {})
        try:
            (self.resultdir / 'ensemble_uncertainty.json').write_text(
                json.dumps(info, indent=2))
        except OSError as e:
            logger.warning('Could not write ensemble_uncertainty.json: %s', e)

    def check_and_maybe_trigger(self, uncertainty_per_image, iteration):
        """Given this iteration's per-image uncertainty, decide whether to
        trigger the early stop. Units follow self.uncertainty_metric: eV for
        'energy', eV/Angstrom for 'force'. Called once per iteration from
        ensemble_mace_path_engrads()."""
        if self.triggered:
            return  # already triggered, nothing to do

        if np.all(np.isnan(uncertainty_per_image)):
            logger.warning('Ensemble uncertainty could not be computed for any '
                           'image this iteration (fewer than 2 models succeeded '
                           'everywhere) -- skipping the uncertainty check for '
                           'this iteration.')
            return

        worst_image_index = int(np.nanargmax(uncertainty_per_image))
        max_uncertainty = float(uncertainty_per_image[worst_image_index])

        improved = (np.isnan(self.max_uncertainty_seen)
                    or max_uncertainty > self.max_uncertainty_seen)
        if improved:
            self.max_uncertainty_seen = max_uncertainty
            self.max_uncertainty_seen_iteration = iteration

        if max_uncertainty > self.threshold_eV:
            self.triggered = True
            self.trigger_message = (
                f'Ensemble uncertainty threshold exceeded at iteration {iteration}: '
                f'image{worst_image_index + 1} has a {max_uncertainty:.4f} {self.units} '
                f'{self.uncertainty_metric} disagreement across the {self.n_models} '
                f'ensemble models (threshold: {self.threshold_eV:.4f} {self.units}). '
                f'Stopping the NEB here -- the current path is the one to relabel.'
            )
            self.trigger_info = {
                'triggered': True,
                'iteration': iteration,
                'image_index': worst_image_index,
                'metric': self.uncertainty_metric,
                'units': self.units,
                'max_uncertainty': max_uncertainty,
                'threshold': self.threshold_eV,
                'n_models': self.n_models,
            }
            # Kept for anything still reading the old key. Only meaningful for
            # the energy metric, so it is deliberately absent for 'force'
            # rather than quietly holding eV/Angstrom under an eV name.
            if self.uncertainty_metric == 'energy':
                self.trigger_info['max_uncertainty_eV'] = max_uncertainty
                self.trigger_info['threshold_eV'] = self.threshold_eV
            logger.error(self.trigger_message)
            self._write_marker(self.trigger_info)
        elif improved:
            self._write_marker()


def build_mace_ensemble_calculators(model_paths, device='cpu', dtype='float64'):
    """Instantiate the N ensemble MACE calculators, once, from an already
    parsed list of checkpoint paths (splitting the raw comma-separated ini
    value is engrad_interface.setup_mace_ensemble()'s job, not this
    function's)."""
    from mace.calculators import MACECalculator

    calculators = []
    for path in model_paths:
        logger.info('Loading ensemble MACE model from: %s', path)
        calculators.append(MACECalculator(model_paths=str(path),
                                          device=device,
                                          default_dtype=dtype))
    return calculators


def _energy_uncertainty(energies_per_model):
    """Per-image std of the de-meaned per-model energies, in eV.

    Each model's own path mean is subtracted first, so a constant
    reference-energy offset between independently trained models is not
    mistaken for genuine disagreement.
    """
    with np.errstate(invalid='ignore'):
        per_model_mean = np.nanmean(energies_per_model, axis=1, keepdims=True)
        relative_energies_Eh = energies_per_model - per_model_mean
        n_valid_per_image = np.sum(~np.isnan(relative_energies_Eh), axis=0)
        uncertainty_Eh = np.nanstd(relative_energies_Eh, axis=0)
    uncertainty_Eh[n_valid_per_image < 2] = np.nan
    return uncertainty_Eh * eV_in_Eh


def _force_uncertainty(grads_per_model, energies_per_model):
    """Per-image force disagreement, in eV/Angstrom.

    For each atom, the RMS across models of the deviation of its force vector
    from the ensemble mean:

        sigma_a = sqrt( mean_m || F_ma - Fbar_a ||^2 )

    and the image's uncertainty is the worst atom's sigma_a -- most sensitive
    to a single region going wrong, which is the point: catch it where it
    happens rather than average it away over a large complex.

    No de-meaning, unlike the energy metric. Forces have no arbitrary per-model
    reference, so there is nothing to subtract and nothing that step could hide.

    A model that failed on an image (NaN energy) is excluded from that image;
    its gradient array is a zero placeholder and would otherwise register as an
    enormous disagreement.
    """
    n_models, n_images = energies_per_model.shape
    out = np.full(n_images, np.nan)

    for i in range(n_images):
        valid = [m for m in range(n_models)
                 if not np.isnan(energies_per_model[m, i])
                 and grads_per_model[m][i] is not None]
        if len(valid) < 2:
            continue
        # (M, n_atoms, 3); gradients are -force, and the sign cancels in a
        # deviation from the mean, so no conversion is needed beyond units
        stack = np.array([np.asarray(grads_per_model[m][i]).reshape(-1, 3) for m in valid])
        deviation = stack - stack.mean(axis=0)
        per_atom_Eh = np.sqrt((deviation ** 2).sum(axis=2).mean(axis=0))
        out[i] = per_atom_Eh.max()

    return out * eV_in_Eh


def ensemble_mace_path_engrads(image_pvecs, labels, series_name, driver):
    """
    Evaluate every image with every ensemble model, and return the
    per-image *mean* energy/gradient across models -- this is what
    actually drives the shared NEB path. Also computes this iteration's
    per-image uncertainty and checks it against driver's threshold.

    - image_pvecs: list of the flattened image vectors (shared across all
      models -- there is only one path)
    - labels: list of the atom types
    - series_name: prefix used for logging image names
    - driver: an EnsembleUncertaintyDriver holding the pre-loaded models
      and the running trigger state
    """
    start_time = time.perf_counter()
    n_images = len(image_pvecs)
    n_models = driver.n_models
    image_names = [f"{series_name}{i+1}" for i in range(n_images)]

    energies_per_model = np.full((n_models, n_images), np.nan)
    grads_per_model = [[None] * n_images for _ in range(n_models)]

    for m, calculator in enumerate(driver.calculators):
        used_batch = False
        if getattr(driver, 'batch_images', False):
            try:
                energies, grads = mac.batched_mace_engrads(
                    image_pvecs, labels, calculator,
                    max_batch_atoms=getattr(driver, 'max_batch_atoms', None))
                for i, (energy, grad) in enumerate(zip(energies, grads)):
                    if energy is not None and np.isfinite(energy):
                        energies_per_model[m, i] = energy
                    grads_per_model[m][i] = grad
                used_batch = True
            except Exception as e:  # noqa: BLE001
                # Fall back to the per-image loop for this model rather than
                # failing the whole path. Batched inference is all-or-nothing,
                # so one pathological geometry would otherwise mark every image
                # of this model as failed -- and NEB paths do pass through
                # geometries that break things. Slower, but it keeps going.
                if m not in driver._batch_fallback_warned:
                    logger.warning('Batched MACE evaluation failed for model %d (%s); '
                                   'falling back to the per-image loop for this model. '
                                   'This message is logged once per model.', m + 1, e)
                    driver._batch_fallback_warned.add(m)

        if not used_batch:
            for i, (pvec, inpfile_name) in enumerate(zip(image_pvecs, image_names)):
                energy, grad = mac.calculate_mace_engrad(
                    pvec, labels, f"{inpfile_name}_model{m+1}", calculator)
                if energy is not None:
                    energies_per_model[m, i] = energy
                grads_per_model[m][i] = grad

    if getattr(driver, 'uncertainty_metric', 'energy') == 'force':
        uncertainty_per_image = _force_uncertainty(grads_per_model, energies_per_model)
    else:
        uncertainty_per_image = _energy_uncertainty(energies_per_model)

    driver.last_energies_per_model_Eh = energies_per_model
    driver.last_uncertainty_per_image = uncertainty_per_image
    # old attribute name, kept for anything reading it
    driver.last_uncertainty_per_image_eV = uncertainty_per_image

    # the mean is what drives the path -- Hartree, matching every other
    # interface's convention. An image is only marked failed (None) if
    # every model failed on it.
    with np.errstate(invalid='ignore'):
        mean_energies = np.nanmean(energies_per_model, axis=0)
    mean_energies = [None if np.isnan(e) else float(e) for e in mean_energies]

    mean_engrads = []
    for i in range(n_images):
        valid_grads = [grads_per_model[m][i] for m in range(n_models)
                       if not np.isnan(energies_per_model[m, i])]
        if valid_grads:
            mean_engrads.append(np.mean(valid_grads, axis=0))
        else:
            mean_engrads.append(np.zeros_like(image_pvecs[i]))

    elapsed = time.perf_counter() - start_time
    metric = getattr(driver, 'uncertainty_metric', 'energy')
    logger.debug('Per-model energies (Eh): %s', energies_per_model)
    logger.debug('Per-image %s uncertainty (%s): %s', metric,
                 getattr(driver, 'units', 'eV'), uncertainty_per_image)
    logger.info('Ensemble MACE engrad calculation for %d images x %d models took '
               '%.3f s (%.3f s/image-model)%s', n_images, n_models, elapsed,
               elapsed / max(n_images * n_models, 1),
               ' [batched]' if getattr(driver, 'batch_images', False) else '')

    # Skip the trigger check (but not the mean-energy/grad computation
    # above, which NEBPath's constructor genuinely needs) for the one-off
    # endpoint sanity check NEBPath.__init__() makes before the real
    # optimization loop starts -- see EnsembleUncertaintyDriver's docstring.
    # That call always passes exactly the 2 fixed endpoints, which doesn't
    # match the real per-iteration image count (settings.n_images) unless
    # n_images == 2, an unavoidable ambiguity flagged separately.
    is_real_iteration = (driver.expected_n_images is None or
                         n_images == driver.expected_n_images)
    if is_real_iteration:
        driver.iteration_count += 1
        driver.check_and_maybe_trigger(uncertainty_per_image, driver.iteration_count)
    else:
        logger.debug('Skipping ensemble uncertainty trigger check for this call '
                    '(%d images passed, expected %d for a real NEB iteration -- '
                    'this looks like NEBPath\'s one-off endpoint sanity check).',
                    n_images, driver.expected_n_images)

    return np.array(mean_energies), np.array(mean_engrads)
