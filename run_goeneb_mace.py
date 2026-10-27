"""Run a GoNEB NEB calculation driven by MACE -- either your own trained
checkpoint or a MACE foundation model (mace-off / mace-mp) -- and read back
the resulting barrier.

Unlike run_neb.py (which drives ASE's own NEB with a UMA calculator
directly), this goes through the actual GoNEB program (interfaces/mace_interface.py)
so you get GoNEB's own tangent/spring/step-predictor machinery, convergence
criteria, and output files (results/finaltraj.xyz, HEI.xyz, TS_guess_cubic.xyz,
optlog.csv) -- exactly like a normal GoNEB job, just launched from Python and
pointed at MACE instead of ORCA/tblite.

For the UMA equivalent, see run_goeneb_uma.py -- both share the interface-
agnostic plumbing in goeneb_runner_common.py.

GoNEB is not pip-installed; its source directory is added to sys.path (via
goeneb_runner_common). Must be run with the MACE_venv interpreter (mace-torch
+ ase + chemcoord installed there), not the sandbox/system python:

Example (custom checkpoint):
    MACE_venv/bin/python run_goeneb_mace.py \\
        --start testjobs/example/pentadiene-1.xyz \\
        --end testjobs/example/pentadiene-2.xyz \\
        --mace-model /path/to/my_trained_model.model \\
        --climbing-image

Example (foundation model):
    MACE_venv/bin/python run_goeneb_mace.py \\
        --start testjobs/example/pentadiene-1.xyz \\
        --end testjobs/example/pentadiene-2.xyz \\
        --mace-foundation mace_off --mace-size medium \\
        --climbing-image
"""

from pathlib import Path

from goeneb_runner_common import (make_parser, parse_neb_args, pick_neb_options,
                                  resolve_jobdir, run_and_get_barrier,
                                  print_barrier_report)

RUNS_DIR = 'goeneb_mace_runs'


def build_mace_model_lines(mace_model_path=None, mace_foundation=None,
                           mace_size='medium', device='cpu', dtype='float64',
                           batch_images=False, max_batch_atoms=0):
    """Build the mace_* lines of the ini. Exactly one of mace_model_path (a
    custom checkpoint) or mace_foundation ('mace_off' / 'mace_mp', combined
    with mace_size) must be given -- same priority rule as
    engrad_interface.build_mace_calculator()."""
    if (mace_model_path is None) == (mace_foundation is None):
        raise ValueError("Specify exactly one of mace_model_path or mace_foundation.")

    if mace_model_path is not None:
        lines = f"mace_model_path = {mace_model_path}\n"
    else:
        lines = f"mace_foundation = {mace_foundation}\nmace_size = {mace_size}\n"
    lines += (f"mace_device = {device}\nmace_dtype = {dtype}\n"
              f"mace_batch_images = {batch_images}\n"
              f"mace_max_batch_atoms = {int(max_batch_atoms or 0)}\n")
    return lines


def run_mace_neb(start_xyz=None, end_xyz=None, mace_model_path=None, jobdir=None, *,
                 starttraj=None,
                 mace_foundation=None, mace_size='medium', interp_mode='internal',
                 n_images=11, climbing_image=True, relaxed_neb=False,
                 sidpp=False, idpp=None,
                 step_pred_method='amgd', max_step=0.05, stepsize_fac=0.2,
                 spring_gradient='difference', k_const=0.003,
                 use_vark=False, vark_min_fac=0.1,
                 use_analytical_springpos=False, tangents='henkjon',
                 device='cpu', dtype='float64', batch_images=False, max_batch_atoms=0,
                 maxiter=500, verbose='debug'):

    if mace_model_path is not None:
        mace_model_path = Path(mace_model_path).resolve()

    jobdir = resolve_jobdir(jobdir, RUNS_DIR)
    model_lines = build_mace_model_lines(mace_model_path, mace_foundation,
                                         mace_size, device, dtype,
                                         batch_images, max_batch_atoms)

    return run_and_get_barrier(jobdir, 'mace', model_lines, **pick_neb_options(locals()))


def main():
    p = make_parser(__doc__, RUNS_DIR)
    model_group = p.add_mutually_exclusive_group(required=True)
    model_group.add_argument('--mace-model', type=Path,
                             help='path to your trained MACE checkpoint (.model)')
    model_group.add_argument('--mace-foundation', choices=['mace_off', 'mace_mp'],
                             help='use a MACE foundation model instead of a custom checkpoint')
    p.add_argument('--mace-size', default='medium', choices=['small', 'medium', 'large'],
                   help='foundation model size (ignored if --mace-model is given)')
    p.add_argument('--device', default='cpu', choices=['cpu', 'cuda', 'mps'])
    p.add_argument('--dtype', default='float64', choices=['float32', 'float64'])
    p.add_argument('--batch-images', dest='batch_images', action='store_true', default=False,
                   help='evaluate all images of an iteration in as few forward passes as '
                        'possible (faster, especially on GPU). A batch that raises falls '
                        'back to the per-image loop for that iteration')
    p.add_argument('--max-batch-atoms', type=int, default=0,
                   help='with --batch-images, cap the atoms per forward pass so a long path '
                        'or large complex is split across several batches (default: 0, one '
                        'batch for the whole path)')
    args = parse_neb_args(p)

    result = run_mace_neb(args.start, args.end, args.mace_model, args.jobdir,
                          mace_foundation=args.mace_foundation, mace_size=args.mace_size,
                          device=args.device, dtype=args.dtype,
                          batch_images=args.batch_images,
                          max_batch_atoms=args.max_batch_atoms,
                          **pick_neb_options(vars(args)))

    print_barrier_report(result)


if __name__ == '__main__':
    main()
