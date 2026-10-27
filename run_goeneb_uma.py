"""Run a GoNEB NEB calculation driven by Meta's UMA model (fairchem-core),
and read back the resulting barrier.

Unlike run_neb.py (which drives ASE's own NEB with a UMA calculator
directly), this goes through the actual GoNEB program (interfaces/uma_interface.py)
so you get GoNEB's own tangent/spring/step-predictor machinery, convergence
criteria, and output files (results/finaltraj.xyz, HEI.xyz, TS_guess_cubic.xyz,
optlog.csv) -- exactly like a normal GoNEB job, just launched from Python and
pointed at UMA instead of ORCA/tblite/MACE.

See run_goeneb_mace.py for the MACE equivalent -- both share the interface-
agnostic plumbing in goeneb_runner_common.py.

GoNEB is not pip-installed; its source directory is added to sys.path (via
goeneb_runner_common). Must be run with an interpreter that has fairchem-core
installed -- your main .venv (not MACE_venv, which doesn't have fairchem-core
and whose torch version could conflict with it):

    .venv/bin/python run_goeneb_uma.py \\
        --start testjobs/example/pentadiene-1.xyz \\
        --end testjobs/example/pentadiene-2.xyz \\
        --uma-model uma-s-1p1 --climbing-image

UMA checkpoints download from Hugging Face on first use (gated repo -- run
`huggingface-cli login` once on the machine first, same prerequisite as
run_neb.py).
"""

from goeneb_runner_common import (make_parser, parse_neb_args, pick_neb_options,
                                  resolve_jobdir, run_and_get_barrier,
                                  print_barrier_report)

RUNS_DIR = 'goeneb_uma_runs'


def build_uma_model_lines(uma_model='uma-s-1p1', uma_task='omol', device='cpu', batch_images=False,
                          charge=0, spin=1):
    """Build the uma_* (+ charge, spin) lines of the ini. charge/spin must be
    written explicitly: uma_interface hands goeneb's charge/spin settings to
    the model, and without these lines they silently stay at 0/1."""
    return (f"uma_model = {uma_model}\numa_task = {uma_task}\numa_device = {device}\n"
           f"uma_batch_images = {batch_images}\ncharge = {charge}\nspin = {spin}\n")


def run_uma_neb(start_xyz=None, end_xyz=None, jobdir=None, *,
                starttraj=None,
                uma_model='uma-s-1p1', uma_task='omol', batch_images=False,
                interp_mode='internal',
                n_images=11, climbing_image=True, relaxed_neb=False,
                sidpp=False, idpp=None,
                step_pred_method='amgd', max_step=0.05, stepsize_fac=0.2,
                spring_gradient='difference', k_const=0.003,
                use_vark=False, vark_min_fac=0.1,
                use_analytical_springpos=False, tangents='henkjon',
                charge=0, spin=1, device='cpu', maxiter=500, verbose='debug'):

    jobdir = resolve_jobdir(jobdir, RUNS_DIR)
    model_lines = build_uma_model_lines(uma_model, uma_task, device, batch_images, charge, spin)

    return run_and_get_barrier(jobdir, 'uma', model_lines, **pick_neb_options(locals()))


def main():
    p = make_parser(__doc__, RUNS_DIR)
    p.add_argument('--uma-model', default='uma-s-1p1',
                   help='fairchem-core UMA checkpoint name (default: uma-s-1p1)')
    p.add_argument('--uma-task', default='omol',
                   help="fairchem task name (default: 'omol' for molecular systems)")
    p.add_argument('--uma-batch', dest='uma_batch_images', action='store_true', default=False,
                   help='batch all images into a single predictor call per iteration '
                        '(faster, especially on GPU; a failure anywhere in the batch '
                        'fails the whole call rather than just one image)')
    p.add_argument('--charge', type=int, default=0)
    p.add_argument('--spin', type=int, default=1, help='2S+1, i.e. 1 for closed-shell singlet')
    p.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    args = parse_neb_args(p)

    result = run_uma_neb(args.start, args.end, args.jobdir,
                         uma_model=args.uma_model, uma_task=args.uma_task,
                         batch_images=args.uma_batch_images,
                         charge=args.charge, spin=args.spin, device=args.device,
                         **pick_neb_options(vars(args)))

    print_barrier_report(result)


if __name__ == '__main__':
    main()
