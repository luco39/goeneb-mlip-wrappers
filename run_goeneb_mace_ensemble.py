"""Run a GoNEB NEB calculation driven by an ensemble of MACE models, for
active-learning uncertainty quantification.

Unlike run_goeneb_mace.py (single model), this loads N independently-trained
MACE checkpoints and evaluates every image with every model each iteration.
The NEB path itself is driven by the *mean* energy/force across all N models
(a single shared path, not N independent ones -- see
goeneb/interfaces/mace_ensemble_interface.py's module docstring for why).
Each model's own per-image energies are de-meaned before comparing across
models, so a constant reference-energy offset between independently trained
models isn't mistaken for genuine disagreement. If the per-image energy
disagreement (std across models, worst image) ever exceeds
--threshold (eV), the NEB stops immediately and the current path is written
out -- meant to be relabelled with a higher level of theory and used as new
training data, not treated as a converged reaction path.

Climbing image is fully compatible with this mode: there is only ever one
path (driven by the mean), so CI doesn't cause per-model paths to diverge
the way it would if each model ran its own independent NEB.

GoNEB is not pip-installed; its source directory is added to sys.path (via
goeneb_runner_common). Must be run with an interpreter that has mace-torch +
ase installed (chemcoord too, if using --interp-mode internal), not the
sandbox/system python:

Example:
    MACE_venv/bin/python run_goeneb_mace_ensemble.py \\
        --start testjobs/example/pentadiene-1.xyz \\
        --end testjobs/example/pentadiene-2.xyz \\
        --model-paths model_0.model,model_1.model,model_2.model,model_3.model,model_4.model \\
        --threshold 0.05 \\
        --climbing-image
"""

import json
from pathlib import Path
import numpy as np

np.random.seed(343)                     ### RANDOMNESS SEEDED HERE FOR REPRODUCIBILITY

from goeneb_runner_common import (make_parser, parse_neb_args, pick_neb_options,  # noqa: E402
                                  resolve_jobdir, run_and_get_barrier,
                                  print_result_paths, print_barriers)

RUNS_DIR = 'goeneb_mace_ensemble_runs'


def build_mace_ensemble_model_lines(model_paths, threshold_eV, device='cpu', dtype='float64',
                                    batch_images=False, max_batch_atoms=0,
                                    uncertainty_metric='energy'):
    """Build the mace_ensemble_* lines of the ini. model_paths is a list of
    checkpoint paths (at least 2, goeneb itself enforces this and raises
    MissingKeyword otherwise)."""
    if len(model_paths) < 2:
        raise ValueError(f"Need at least 2 model paths for an ensemble, got {len(model_paths)}.")
    joined_paths = ','.join(str(p) for p in model_paths)
    lines = (f"mace_ensemble_model_paths = {joined_paths}\n"
            f"mace_ensemble_uncertainty_threshold = {threshold_eV}\n"
            f"mace_ensemble_device = {device}\n"
            f"mace_ensemble_dtype = {dtype}\n"
            f"mace_ensemble_uncertainty_metric = {uncertainty_metric}\n"
            f"mace_ensemble_batch_images = {batch_images}\n"
            f"mace_ensemble_max_batch_atoms = {int(max_batch_atoms or 0)}\n")
    return lines


def run_mace_ensemble_neb(start_xyz=None, end_xyz=None, model_paths=None,
                          threshold_eV=None, jobdir=None, *,
                          starttraj=None,
                          interp_mode='internal', n_images=11,
                          climbing_image=True, relaxed_neb=False,
                          sidpp=False, idpp=None,
                          step_pred_method='amgd', max_step=0.05, stepsize_fac=0.2,
                          spring_gradient='difference', k_const=0.003,
                          use_vark=False, vark_min_fac=0.1,
                          use_analytical_springpos=False, tangents='henkjon',
                          device='cpu', dtype='float64', maxiter=500, verbose='debug',
                          batch_images=False, max_batch_atoms=0,
                          uncertainty_metric='energy'):
                          
    if threshold_eV is None:
        raise ValueError("threshold_eV is required (the uncertainty threshold, in eV).")
    if model_paths is None or len(model_paths) < 2:
        raise ValueError("model_paths must be a list of at least 2 MACE checkpoint paths.")

    model_paths = [Path(p).resolve() for p in model_paths]
    jobdir = resolve_jobdir(jobdir, RUNS_DIR)
    model_lines = build_mace_ensemble_model_lines(model_paths, threshold_eV, device, dtype,
                                                 batch_images, max_batch_atoms,
                                                 uncertainty_metric)

    result = run_and_get_barrier(jobdir, 'mace_ensemble', model_lines,
                                 **pick_neb_options(locals()))

    # The marker is now written whenever the running maximum improves, not only
    # on a trigger, so that a run finishing at maxiter still reports how close it
    # came. Its 'triggered' field decides, not its existence. Marker files
    # written before that change always carried triggered=True, so defaulting to
    # True keeps them reading correctly.
    marker_path = result['resultdir'] / 'ensemble_uncertainty.json'
    if marker_path.exists():
        info = json.loads(marker_path.read_text())
        result['uncertainty_triggered'] = bool(info.get('triggered', True))
        result['uncertainty_info'] = info
    else:
        result['uncertainty_triggered'] = False
        result['uncertainty_info'] = None

    return result


def print_ensemble_report(result: dict):
    """CLI printout distinguishing "stopped due to uncertainty" from
    "converged" or "reached maxiter" -- barrier numbers from a
    driver-stopped run are NOT a converged barrier and shouldn't be read as
    one; the path itself (finaltraj.xyz) is the useful output in that case,
    meant for relabelling."""
    print_result_paths(result)

    if result['uncertainty_triggered']:
        info = result['uncertainty_info']
        # 'max_uncertainty'/'threshold' carry the metric's own units; markers
        # written before the force metric existed only have the *_eV keys
        units = info.get('units', 'eV')
        max_uncertainty = info.get('max_uncertainty', info.get('max_uncertainty_eV'))
        threshold = info.get('threshold', info.get('threshold_eV'))
        print(f"\n*** STOPPED EARLY: ensemble uncertainty threshold exceeded ***")
        print(f"Iteration:                {info['iteration']}")
        print(f"Worst-disagreement image: {info['image_index']}")
        print(f"{f'Max uncertainty ({units}):':<26}{max_uncertainty:.4f}")
        print(f"{f'Threshold ({units}):':<26}{threshold:.4f}")
        print(f"Ensemble size:            {info['n_models']} models")
        print(f"\nThe path at this point (results/finaltraj.xyz) is NOT converged -- "
             f"this is expected. Relabel it at a higher level of theory and add it "
             f"to the training set. The barrier numbers below describe this "
             f"intermediate, unconverged path only.")
    else:
        print(f"\nNo uncertainty trigger -- the run either converged normally or "
             f"reached maxiter without any model disagreement exceeding threshold.")

    print_barriers(result)


def main():
    p = make_parser(__doc__, RUNS_DIR)
    p.add_argument('--model-paths', required=True,
                   help='comma-separated list of at least 2 MACE checkpoint paths '
                        '(a common ensemble size is 5, but this is up to you)')
    p.add_argument('--threshold', type=float, required=True,
                   help='uncertainty threshold in eV: stop the NEB as soon as the '
                        'worst-image energy disagreement (std across models, after '
                        'de-meaning each model\'s own path) exceeds this')
    p.add_argument('--device', default='cpu', choices=['cpu', 'cuda', 'mps'])
    p.add_argument('--dtype', default='float64', choices=['float32', 'float64'])
    p.add_argument('--uncertainty-metric', default='energy', choices=['energy', 'force'],
                   help="what the --threshold measures: 'energy' (default) is the std "
                        "across models of the per-image energy after de-meaning each "
                        "model's own path, in eV. 'force' is the worst atom's RMS force "
                        "disagreement, in eV/Angstrom -- needs no de-meaning, is intensive "
                        "so one value means the same for small and large complexes, and is "
                        "the quantity the NEB optimiser actually moves the path with.")
    p.add_argument('--batch-images', dest='batch_images', action='store_true', default=False,
                   help='pack every image into one forward pass per model per iteration '
                        '(n_models calls instead of n_models x n_images -- markedly faster, '
                        'especially on GPU). A batch that raises falls back to the per-image '
                        'loop for that model, so it is not all-or-nothing.')
    p.add_argument('--max-batch-atoms', type=int, default=0,
                   help='with --batch-images, cap the atoms per forward pass so a long path '
                        'or large complex is split across several batches (default: 0, one '
                        'batch for the whole path)')
    args = parse_neb_args(p)

    model_paths = [s.strip() for s in args.model_paths.split(',') if s.strip()]

    result = run_mace_ensemble_neb(args.start, args.end, model_paths, args.threshold, args.jobdir,
                                   device=args.device, dtype=args.dtype,
                                   batch_images=args.batch_images,
                                   max_batch_atoms=args.max_batch_atoms,
                                   uncertainty_metric=args.uncertainty_metric,
                                   **pick_neb_options(vars(args)))

    print_ensemble_report(result)


if __name__ == '__main__':
    main()
