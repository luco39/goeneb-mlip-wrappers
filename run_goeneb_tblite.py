"""Run a GoNEB NEB calculation driven by tblite (GFN1-xTB / GFN2-xTB), and
read back the resulting barrier.

Useful as a fast, independent sanity check against a MACE/UMA-driven run:
different level of theory (semiempirical tight-binding, not an MLIP) *and*
goeneb's own NEB machinery (tangents, springs, step predictor, climbing
image, convergence criteria) rather than standing up a separate ASE-native
NEB -- so a run here isolates "does this path/geometry hold together under a
different potential", reusing everything else you already trust about how
these jobs are run and logged.

See run_goeneb_mace.py / run_goeneb_uma.py for the MLIP equivalents -- all
three share the interface-agnostic plumbing in goeneb_runner_common.py.

GoNEB is not pip-installed; its source directory is added to sys.path (via
goeneb_runner_common). Must be run with an interpreter that has tblite
installed (its Python bindings, i.e. `import tblite.interface` must work) --
check which of your venvs (if any) has this before running; it doesn't need
mace-torch or fairchem-core.

Example (starting fresh from an existing multi-frame path, e.g. the same
starttraj used to seed a MACE run -- the most independent check, since
nothing in the initial geometry has been shaped by the MLIP being checked):
    some_venv/bin/python run_goeneb_tblite.py \\
        --starttraj neb_test_traj.xyz \\
        --method GFN2-xTB \\
        --climbing-image --n-threads 8 --maxiter 1000

Example (starting from two end structures instead):
    some_venv/bin/python run_goeneb_tblite.py \\
        --start testjobs/example/pentadiene-1.xyz \\
        --end testjobs/example/pentadiene-2.xyz \\
        --method GFN2-xTB --climbing-image
"""

from goeneb_runner_common import (make_parser, parse_neb_args, pick_neb_options,
                                  resolve_jobdir, run_and_get_barrier,
                                  print_barrier_report)

RUNS_DIR = 'goeneb_tblite_runs'


def build_tblite_model_lines(method_keywords='GFN2-xTB', n_threads=1, charge=0, spin=1):
    """Build the tblite_keywords (+ n_threads, charge, spin) lines of the
    ini. method_keywords is passed straight through to
    tblite.interface.Calculator (e.g. 'GFN2-xTB' or 'GFN1-xTB') -- see
    interfaces/tblite_interface.py. n_threads sets OMP_NUM_THREADS for
    tblite's own parallelization (set via engrad_interface.setup_tblite()
    before each run).

    charge/spin are folded in here too (rather than into
    goeneb_runner_common.write_common_ini(), which doesn't expose them --
    MACE ignores these settings, but tblite genuinely needs them
    for the calculation to be correct, via charge/spin -> uhf). Without
    explicit lines here, goeneb would silently fall back to its own
    defaults (charge=0, spin=1) regardless of what's passed to
    run_tblite_neb(), which would make a --charge/--spin override a no-op."""
    return (f"tblite_keywords = {method_keywords}\nn_threads = {n_threads}\n"
           f"charge = {charge}\nspin = {spin}\n")


def run_tblite_neb(start_xyz=None, end_xyz=None, jobdir=None, *,
                   starttraj=None,
                   method_keywords='GFN2-xTB', n_threads=1,
                   interp_mode='internal', n_images=11,
                   climbing_image=True, relaxed_neb=False,
                   sidpp=False, idpp=None,
                   step_pred_method='amgd', max_step=0.05, stepsize_fac=0.2,
                   spring_gradient='difference', k_const=0.003,
                   use_vark=False, vark_min_fac=0.1,
                   use_analytical_springpos=False, tangents='henkjon',
                   charge=0, spin=1, maxiter=500, verbose='debug'):
    """Run a GoNEB NEB calculation using tblite (GFN-xTB) and return the
    resulting barrier (and where to find the full results on disk).

    method_keywords: tblite method string, 'GFN2-xTB' (default, the modern
    standard choice) or 'GFN1-xTB'.

    n_threads: OMP_NUM_THREADS for tblite. xTB is cheap enough that this
    mostly matters for turnaround time, not feasibility -- set it to
    whatever your SLURM allocation gives you.

    charge, spin: same meaning as goeneb's charge/spin settings generally
    (spin = 2S+1, i.e. 1 for closed-shell singlet). Match whatever was used
    for the run you're sanity-checking unless you have a specific reason to
    think the true electronic state differs (e.g. if you suspect a
    near-dissociation region is actually open-shell/homolytic -- tblite
    supports this via charge/spin -> uhf, same as the underlying xTB
    convention, but you may want to check this separately rather than
    assume the closed-shell default is right there).

    Structures: either start_xyz and end_xyz, or starttraj -- e.g. the same
    starttraj used to seed a MACE/UMA run, for the most independent possible
    sanity check (the starting geometry hasn't been shaped by the MLIP being
    checked at all). The remaining options configure the NEB itself and are
    the same for every run_goeneb_*.py wrapper -- see
    goeneb_runner_common.write_common_ini()'s docstring for what each does.
    The spring/tangent group in particular is worth exploring if climbing
    image keeps settling somewhere you don't believe is the real TS."""
    jobdir = resolve_jobdir(jobdir, RUNS_DIR)
    model_lines = build_tblite_model_lines(method_keywords, n_threads, charge, spin)

    return run_and_get_barrier(jobdir, 'tblite', model_lines, **pick_neb_options(locals()))


def main():
    p = make_parser(__doc__, RUNS_DIR)
    p.add_argument('--method', default='GFN2-xTB', choices=['GFN2-xTB', 'GFN1-xTB'],
                   help='tblite method (default: GFN2-xTB)')
    p.add_argument('--n-threads', type=int, default=1,
                   help='OMP_NUM_THREADS for tblite (default: 1; set to match your '
                        'SLURM allocation)')
    p.add_argument('--charge', type=int, default=0)
    p.add_argument('--spin', type=int, default=1, help='2S+1, i.e. 1 for closed-shell singlet')
    args = parse_neb_args(p)

    result = run_tblite_neb(args.start, args.end, args.jobdir,
                            method_keywords=args.method, n_threads=args.n_threads,
                            charge=args.charge, spin=args.spin,
                            **pick_neb_options(vars(args)))

    print_barrier_report(result)


if __name__ == '__main__':
    main()
