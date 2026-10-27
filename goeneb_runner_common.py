"""Shared plumbing for the run_goeneb_*.py wrapper scripts (run_goeneb_mace.py,
run_goeneb_uma.py, ...). Each wrapper builds its own interface-specific ini
lines (mace_model_path / mace_foundation / ..., or uma_model / uma_task / ...)
and hands them to write_common_ini() here; everything else -- jobdir/tempdir
setup, logging, running goeneb, reading the barrier back out of optlog.csv,
and the command-line arguments every wrapper shares -- is interface-agnostic
and lives in one place.

GoNEB is not pip-installed; its source directory is added to sys.path below,
which every run_goeneb_*.py script picks up by importing from this module
first.
"""

import argparse
import inspect
import logging
import os
import sys
from pathlib import Path
from datetime import datetime

import numpy as np

# Resolved relative to this file rather than hardcoded, so this works
# unmodified wherever the Summer Project folder (with goeneb/ as a sibling
# of this file) ends up -- e.g. copied over to an HPC cluster. Override with
# the GOENEB_PATH environment variable if goeneb/ lives somewhere else.
GOENEB_PATH = Path(os.environ.get("GOENEB_PATH", Path(__file__).resolve().parent / "goeneb"))
if not GOENEB_PATH.is_dir():
    raise FileNotFoundError(
        f"goeneb source directory not found at {GOENEB_PATH}. "
        "Either place a 'goeneb' clone next to this file, or set the "
        "GOENEB_PATH environment variable to point at it."
    )
sys.path.insert(0, str(GOENEB_PATH))

from logging_module import setup_logger  # noqa: E402
from neb import main as goeneb_main  # noqa: E402
from neb_optimizer import Hartree_in_kJmol  # noqa: E402
from optplot import load_optlog, get_energy_profile_array, get_singleval_array  # noqa: E402

kJmol_per_eV = 96.4853075
kJmol_in_kcalmol = 1 / 4.184


def reset_root_logger():
    """logging_module.setup_logger() wraps logging.basicConfig(), which is a
    no-op after the first call in a process. Without this, calling a
    run_*_neb() function more than once in the same Python session/notebook
    kernel (e.g. comparing two models back to back) silently keeps writing to
    the *first* run's output.log -- or to no file at all -- instead of the
    current jobdir. Clear the handlers first so each run gets its own
    correctly-located log."""
    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)
        handler.close()


def resolve_jobdir(jobdir, default_subdir: str) -> Path:
    """Resolve jobdir to an absolute, existing Path. If jobdir is None,
    generate a timestamped one under ./<default_subdir>/. Microsecond
    resolution: two calls in the same notebook cell can easily land in the
    same wall-clock second otherwise, which would make them collide."""
    if jobdir is None:
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        jobdir = Path.cwd() / default_subdir / stamp
    jobdir = Path(jobdir).resolve()
    jobdir.mkdir(parents=True, exist_ok=True)
    return jobdir


def write_common_ini(jobdir: Path, interface: str, model_ini_lines: str,
                     start_xyz: Path = None, end_xyz: Path = None, starttraj: Path = None,
                     interp_mode: str = 'internal', n_images=11,
                     climbing_image=True, relaxed_neb=False,
                     sidpp=False, idpp=None,
                     step_pred_method='amgd', max_step=0.05, stepsize_fac=0.2,
                     spring_gradient='difference', k_const=0.003,
                     use_vark=False, vark_min_fac=0.1,
                     use_analytical_springpos=False, tangents='henkjon',
                     maxiter=500, verbose='debug') -> Path:
    """Write a neb.ini for the given interface. model_ini_lines is the
    interface-specific block (e.g. 'mace_foundation = ...\\nmace_size = ...\\n'
    or 'uma_model = ...\\n'), built by the calling wrapper.

    Structures: pass either (start_xyz and end_xyz) to interpolate a path, or
    starttraj to start NEB directly from an existing multi-frame xyz path
    (e.g. one of the reaction trajectories under labelled_data/ -- goeneb
    only reads the atom label + first three numeric columns per line, so
    extra columns like forces are ignored and don't need stripping first).
    Exactly one of the two must be given. When starttraj is given,
    n_images/interp_mode are not used -- goeneb runs NEB on exactly however
    many frames the file contains.

    interp_mode (ignored if starttraj or sidpp is given) controls how the
    initial path is generated: 'internal' (default, z-matrix-based, needs
    chemcoord), 'cartesian', or 'geodesic' (Zhu et al.'s geodesic
    interpolation, bundled in goeneb's own geodesic_module.py -- no extra
    dependency, but non-deterministic, so results vary slightly run to run).

    sidpp: if True, use Sequential IDPP (Sharada/Zhang et al.,
    https://doi.org/10.1021/acs.jctc.3c01111) to build the initial path
    instead of interp_mode + plain IDPP -- goeneb's path_interpolator_module
    branches on this before ever looking at interp_mode or IDPP, so those
    settings are simply ignored when sidpp=True. Only meaningful when
    generating a path from start_xyz/end_xyz; has no effect with starttraj,
    since there's no path to (re-)generate in that case.

    idpp: whether to run a plain IDPP pre-optimization pass on the initial
    path. IDPP is a purely geometric, pairwise-distance-based smoothing step
    -- it knows nothing about the real (MACE/UMA/QM) energy surface. This
    matters because goeneb applies it in *both* cases: after a naive
    interpolation from start_xyz/end_xyz (where it's genuinely useful, to
    spread out a straight-line interpolation into reasonable geometries),
    and -- easy to miss -- after reading in a starttraj (goeneb's
    produce_starting_path() runs it unconditionally if settings.IDPP is
    True, regardless of path source). Since a starttraj is usually already a
    sensible/relaxed path (e.g. a previous NEB run's finaltraj.xyz, or a
    real reaction trajectory), re-running a geometry-only IDPP pass on it
    can distort it significantly with no relation to the actual energy
    landscape -- if you restart a run from finaltraj.xyz and the reported
    barrier jumps to something wildly different on iteration 1, this is the
    most likely cause. Default (idpp=None) is smart: False when starttraj is
    given, True when interpolating from start_xyz/end_xyz (goeneb's own
    default). Pass explicitly to override either way.

    step_pred_method: which optimizer predicts each image's step from its
    NEB gradient. 'amgd' (default, adaptive momentum gradient descent) is
    goeneb's usual choice and generally robust, but its momentum term can
    overshoot and oscillate around a sharp/narrow barrier top or a slightly
    noisy PES (as MLIP surfaces like MACE/UMA can be) -- exactly the
    "reached maxiter but the climbing image never settled" failure mode.
    Other options (case-insensitive, see step_pred_module.py):
    'sd' (plain steepest descent, slower but non-oscillatory -- no momentum
    term to overshoot with), 'nr'/'rfo' (global Newton-Raphson / rational
    function optimization once a Hessian estimate is built up, after
    BFGS_start/NR_start iterations of AMGD warm-up), 'l-nr'/'l-rfo' (same
    but per-image local Hessians instead of one global one), 'sct'
    (self-consistent-tangent harmonic NEB). See
    plot_convergence_diagnostics() to check whether a given choice is
    actually settling down or still oscillating.

    max_step: the maximum allowed norm of any single image's step per
    iteration (goeneb's own default is 0.05); steps longer than this get
    rescaled down to this length (see enforce_maxstep() in
    step_pred_module.py) rather than rejected outright. Lowering this is a
    direct, blunt way to damp oscillation -- smaller steps can't overshoot
    as far -- at the cost of needing more iterations to get anywhere.

    stepsize_fac: scales the raw step every predictor takes (goeneb's own
    default 0.2) -- for 'sd' it's the entire step size (step = -stepsize_fac
    * gradient, no momentum, so this is the only lever controlling how fast
    it moves); for 'amgd' it scales the gradient term each iteration
    (on top of whatever momentum carries over); for the BFGS-family methods
    ('nr'/'rfo'/'l-nr'/'l-rfo'/'sct') it scales both the AMGD warm-up phase
    and the eventual Newton/RFO step. Raising it speeds up a slow-but-stable
    run (e.g. 'sd', which has no momentum to run away with even at a larger
    stepsize_fac); lowering it is another way to damp instability, similar
    in spirit to lowering max_step but scaling the predicted step itself
    rather than clipping an oversized one after the fact.

    The following six control the tangent/spring-force machinery that
    actually keeps the climbing image on the intended path and everything
    else roughly evenly spaced -- relevant if CI keeps landing somewhere you
    don't think is the real TS, since a poorly-behaved spring/tangent setup
    can let CI drift off the ridge you want it to climb, or let neighboring
    images bunch up/spread out in a way that distorts the tangent CI relies
    on:

    spring_gradient: how the spring force between adjacent images is
    computed (case-insensitive). 'difference' (default) is Henkelman &
    Jonsson's improved definition; 'projected' is the original NEB
    formulation; 'raw' is the un-projected spring force (not projected onto
    the tangent direction) -- rarely what you want, but available. If CI is
    drifting off-path, 'difference' vs 'projected' is one of the first
    things worth comparing, since it changes how strongly/where the path is
    held taut around the climbing image.

    k_const: the spring force constant, in Eh/Angstrom^2 (default 0.003).
    Larger values hold images more rigidly evenly-spaced (stronger
    resistance to bunching near a sharp feature like a TS) but can also
    fight harder against the real energy gradient's attempt to move an
    image toward where it actually wants to go; smaller values let images
    space themselves more freely along the energy landscape at the cost of
    weaker path-taughtness. If use_vark is True, this is instead the
    *maximum* spring constant used.

    use_vark: whether to use Henkelman & Jonsson's variable-k scheme, which
    scales each inter-image spring constant based on local energy (stiffer
    springs near the high-energy/TS region, softer ones near the low-energy
    ends) rather than one fixed k_const everywhere. Often more physically
    sensible for asymmetric barriers, and can help CI "lock on" to a sharp
    feature without the rest of the path being over/under-constrained.

    vark_min_fac: only used when use_vark=True -- the minimum spring
    constant between images is vark_min_fac * k_const (default 0.1, i.e.
    the softest springs are 10% of the max/k_const value).

    use_analytical_springpos: whether to use the analytical spring
    positioning scheme, which computes each image's ideal 1D position along
    the path directly rather than letting spring forces push it there
    iteratively. Does NOT work with the 'sct' step predictor. Can help
    images settle into evenly-spaced positions faster/more predictably, but
    is a more rigid scheme than the default force-based spacing.

    tangents: how the local tangent direction at each image is estimated
    (case-insensitive). 'henkjon' (default) is Henkelman & Jonsson's
    improved, energy-weighted tangent estimate (more robust near a sharp
    kink like a real barrier top); 'simple' is the older bisection-based
    tangent. Since climbing image's whole mechanism depends on correctly
    identifying "uphill along the tangent, downhill/spring-balanced
    everything else", a poor tangent estimate right where the path is
    sharply curved (exactly where a real TS tends to be) is a very plausible
    reason CI could get pushed somewhere other than where you expect --
    'henkjon' should generally be preferred, but worth trying 'simple' as a
    comparison point if you suspect the tangent estimate itself is the
    problem."""
    have_ends = start_xyz is not None and end_xyz is not None
    if have_ends == (starttraj is not None):
        raise ValueError("Specify either both start_xyz and end_xyz, or starttraj, not both/neither.")

    if idpp is None:
        idpp = starttraj is None

    # goeneb needs a writable scratch dir (normally set up by scripts/environment.sh
    # via NEB_TMPDIR when launched from the shell; set it explicitly here instead
    # since we're driving goeneb straight from Python)
    tempdir = jobdir / '_tmp'
    tempdir.mkdir(parents=True, exist_ok=True)

    if starttraj is not None:
        path_lines = f"starttraj = {starttraj}\n"
    else:
        path_lines = (f"start_structure = {start_xyz}\n"
                      f"end_structure = {end_xyz}\n"
                      f"interp_mode = {interp_mode}\n"
                      f"n_images = {n_images}\n")

    ini_path = jobdir / 'neb.ini'
    ini_text = f"""[options]
interface = {interface}
{model_ini_lines}{path_lines}SIDPP = {sidpp}
IDPP = {idpp}
tempdir = {tempdir}
climbing_image = {climbing_image}
relaxed_neb = {relaxed_neb}
maxiter = {maxiter}
step_pred_method = {step_pred_method}
max_step = {max_step}
stepsize_fac = {stepsize_fac}
spring_gradient = {spring_gradient}
k_const = {k_const}
use_vark = {use_vark}
vark_min_fac = {vark_min_fac}
use_analytical_springpos = {use_analytical_springpos}
tangents = {tangents}
verbose = {verbose}
"""
    ini_path.write_text(ini_text)
    return ini_path


# Every keyword write_common_ini() takes besides jobdir/interface/model lines.
# The run_*_neb() wrappers spell these out in their own signatures (so they
# show up in a notebook's autocomplete) and pass them on by name.
NEB_OPTIONS = tuple(inspect.signature(write_common_ini).parameters)[3:]


def pick_neb_options(namespace: dict) -> dict:
    """The NEB_OPTIONS entries of a dict of names -- a wrapper's locals(), or
    vars() of the parsed command line."""
    return {name: namespace[name] for name in NEB_OPTIONS if name in namespace}


def run_goeneb_job(jobdir: Path, ini_path: Path, verbose='debug') -> Path:
    """Run goeneb on the given ini file and return the results directory it
    produced (identified by diffing results* dirs before/after, so this
    works even if jobdir is reused across multiple calls)."""
    existing_resultdirs = set(jobdir.glob('results*'))

    reset_root_logger()
    setup_logger(workdir=str(jobdir), level=verbose)
    args = argparse.Namespace(input_file=str(ini_path), verbose=None,
                              trajtest=False, maxiter=None, images=None)
    goeneb_main(args)

    new_resultdirs = set(jobdir.glob('results*')) - existing_resultdirs
    if len(new_resultdirs) != 1:
        raise RuntimeError(f"Could not unambiguously identify the results "
                           f"directory for this run in {jobdir} "
                           f"(found: {new_resultdirs})")
    return new_resultdirs.pop()


def barrier_from_results(resultdir: Path) -> dict:
    """Read the final iteration's path energies out of optlog.csv (the same
    structured log GoNEB itself uses for optplot.py) and compute the barrier
    exactly as neb_optimizer.py does, rather than parsing the free-text log."""
    optlog_path = resultdir / 'optlog.csv'
    token_lines = load_optlog(optlog_path)
    energies_array = get_energy_profile_array(token_lines)  # (n_iters, n_images), Hartree

    final_energies = energies_array[-1]
    max_energy = final_energies.max()

    left_barrier_kJmol = (max_energy - final_energies[0]) * Hartree_in_kJmol
    right_barrier_kJmol = (max_energy - final_energies[-1]) * Hartree_in_kJmol

    return {
        'path_energies_Eh': final_energies,
        'left_barrier_kJmol': left_barrier_kJmol,
        'right_barrier_kJmol': right_barrier_kJmol,
    }


def barrier_trend_from_optlog(optlog_path: Path):
    """Like barrier_from_results(), but for every iteration instead of just
    the final one -- lets you see whether the barrier estimate is actually
    settling down over the run or oscillating without ever damping (a real
    failure mode for the climbing image: goeneb can report 'reached maxiter'
    while the barrier is still swinging by hundreds of kJ/mol iteration to
    iteration, in which case the final number is just an arbitrary snapshot,
    not a converged answer). See plot_convergence_diagnostics() for a plot
    of this.

    Returns (iterations, left_barrier_kJmol, right_barrier_kJmol), each a
    1D array of length n_iterations."""
    token_lines = load_optlog(optlog_path)
    energies_array = get_energy_profile_array(token_lines)  # (n_iters, n_images), Hartree

    max_energies = energies_array.max(axis=1)
    left_barrier_kJmol = (max_energies - energies_array[:, 0]) * Hartree_in_kJmol
    right_barrier_kJmol = (max_energies - energies_array[:, -1]) * Hartree_in_kJmol
    iterations = np.arange(1, len(energies_array) + 1)

    return iterations, left_barrier_kJmol, right_barrier_kJmol


# goeneb's own convergence tolerance defaults (neb_configparse.py Settings.__init__).
# Used to draw tolerance lines in plot_convergence_diagnostics() when the
# caller doesn't know/pass the tolerances the run actually used.
DEFAULT_MAX_RMSF_TOL = 0.000945
DEFAULT_MAX_ABSF_TOL = 0.00189
DEFAULT_CI_RMSF_TOL = 0.000473
DEFAULT_CI_ABSF_TOL = 0.000945


def plot_convergence_diagnostics(resultdir_or_optlog, savepath=None, show=False, *,
                                 Max_RMSF_tol=DEFAULT_MAX_RMSF_TOL,
                                 Max_AbsF_tol=DEFAULT_MAX_ABSF_TOL,
                                 CI_RMSF_tol=DEFAULT_CI_RMSF_TOL,
                                 CI_AbsF_tol=DEFAULT_CI_ABSF_TOL):
    """Plot barrier-vs-iteration next to convergence-measures-vs-iteration,
    read straight from optlog.csv. Much faster and more reliable than
    grepping a free-text output.log for a long run -- optlog.csv has one
    compact row per iteration no matter how many thousand lines the
    free-text log grew to.

    Left panel: left/right barrier per iteration (kJ/mol). A genuinely
    converging run shows this settling to a flat line. A run stuck in a
    limit cycle (climbing image bouncing across the barrier top instead of
    settling into it -- see conv_checker_func_ci in neb_optimizer.py) shows
    persistent, non-damping oscillation instead. Worth checking before
    trusting a "reached maxiter" result at face value.

    Right panel: RMSF/AbsF (path) and RMSF_CI/AbsF_CI (climbing image, only
    present if climbing_image was on) per iteration, log-scale, with dotted
    horizontal lines at the tolerances goeneb checks against. Defaults are
    goeneb's own stock tolerances (Settings.__init__ in neb_configparse.py)
    -- pass the *_tol kwargs if your run used non-default values (e.g.
    relaxed_neb uses Relaxed_Max_RMSF_tol/Relaxed_Max_AbsF_tol instead, 10x
    looser, and has no separate CI tolerances since relaxed_neb bypasses the
    CI-specific check entirely).

    resultdir_or_optlog: either a results directory (as returned by
    run_goeneb_job()/run_and_get_barrier(), i.e. result['resultdir']) or a
    direct path to an optlog.csv. savepath: if given, save the figure there.
    show: whether to call plt.show() (blocks in a script; fine in a
    notebook)."""
    import matplotlib.pyplot as plt

    resultdir_or_optlog = Path(resultdir_or_optlog)
    optlog_path = (resultdir_or_optlog if resultdir_or_optlog.name == 'optlog.csv'
                  else resultdir_or_optlog / 'optlog.csv')

    iterations, left_kJmol, right_kJmol = barrier_trend_from_optlog(optlog_path)

    token_lines = load_optlog(optlog_path)
    sv_names, sv_array = get_singleval_array(token_lines)

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(11, 4.5))

    ax1.plot(iterations, left_kJmol, label='left barrier', color='tab:blue')
    ax1.plot(iterations, right_kJmol, label='right barrier', color='tab:orange')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Barrier [kJ/mol]')
    ax1.set_title('Barrier vs. iteration')
    ax1.grid(True, which='both', alpha=0.3)
    ax1.legend()

    tol_map = {'RMSF': Max_RMSF_tol, 'AbsF': Max_AbsF_tol,
              'RMSF_o': Max_RMSF_tol, 'AbsF_o': Max_AbsF_tol,
              'RMSF_CI': CI_RMSF_tol, 'AbsF_CI': CI_AbsF_tol}
    for i, name in enumerate(sv_names):
        line, = ax2.plot(iterations, sv_array[:, i], label=name)
        if name in tol_map:
            ax2.axhline(tol_map[name], color=line.get_color(), ls=':', alpha=0.6)
    ax2.set_yscale('log')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel(r'Value [$E_\mathrm{h}/\mathrm{\AA}$]')
    ax2.set_title('Convergence measures (dotted = tolerance)')
    ax2.grid(True, which='both', alpha=0.3)
    ax2.legend(fontsize=8)

    plt.tight_layout()

    if savepath is not None:
        plt.savefig(savepath, dpi=150)
        logging.getLogger(__name__).info('Convergence diagnostics plot saved to %s', savepath)
    if show:
        plt.show()

    return fig


def run_and_get_barrier(jobdir: Path, interface: str, model_ini_lines: str,
                        **neb_options) -> dict:
    """High-level helper tying the above together: write the ini, run
    goeneb, and read back the barrier. Returns barrier_from_results()'s dict
    plus jobdir/resultdir/output_log. neb_options are write_common_ini()'s
    keyword arguments (see its docstring for what each does); the structure
    paths among them are resolved against the current directory here."""
    for name in ('start_xyz', 'end_xyz', 'starttraj'):
        if neb_options.get(name) is not None:
            neb_options[name] = Path(neb_options[name]).resolve()

    ini_path = write_common_ini(jobdir, interface, model_ini_lines, **neb_options)
    resultdir = run_goeneb_job(jobdir, ini_path, verbose=neb_options.get('verbose', 'debug'))
    result = barrier_from_results(resultdir)
    result['jobdir'] = jobdir
    result['resultdir'] = resultdir
    result['output_log'] = jobdir / 'output.log'
    return result


def max_step_type(s):
    """argparse type for --max-step: accepts a float, or 'none' to disable
    step clipping entirely (goeneb treats max_step=None that way)."""
    if s.lower() == 'none':
        return None
    return float(s)


def make_parser(description: str, default_runs_dir: str) -> argparse.ArgumentParser:
    """Parser holding the arguments every run_goeneb_*.py script shares (the
    structures, --jobdir, and every NEB_OPTIONS setting); each script adds
    its own model arguments on top. Dest names match NEB_OPTIONS, so
    pick_neb_options(vars(args)) collects them all -- except --start/--end,
    which the scripts pass positionally as start_xyz/end_xyz."""
    p = argparse.ArgumentParser(description=description,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--start', type=Path, help='start structure .xyz (with --end)')
    p.add_argument('--end', type=Path, help='end structure .xyz (with --start)')
    p.add_argument('--starttraj', type=Path,
                   help='existing multi-frame xyz path to start NEB from directly, '
                        'instead of --start/--end (e.g. a labelled_data/*.xyz file, or a '
                        "previous run's finaltraj.xyz)")
    p.add_argument('--jobdir', type=Path, default=None,
                   help=f'where to run the job (default: timestamped dir under ./{default_runs_dir})')
    p.add_argument('--interp-mode', default='internal',
                   choices=['internal', 'cartesian', 'geodesic'],
                   help='initial path interpolation method (default: internal/z-matrix; '
                        'ignored with --starttraj)')
    p.add_argument('--n-images', type=int, default=11)
    p.add_argument('--climbing-image', action='store_true', default=True)
    p.add_argument('--no-climbing-image', dest='climbing_image', action='store_false')
    p.add_argument('--relaxed-neb', action='store_true', default=False)
    p.add_argument('--sidpp', action='store_true', default=False,
                   help='build the initial path with Sequential IDPP instead of '
                        '--interp-mode + plain IDPP (ignored with --starttraj)')
    p.add_argument('--idpp', dest='idpp', action='store_true', default=None,
                   help='force a plain IDPP pre-optimization pass on the initial path '
                        '(default: on when interpolating from --start/--end, off when '
                        'using --starttraj -- IDPP is purely geometric and can distort '
                        'an already-good path, e.g. when --starttraj is a previous '
                        "run's finaltraj.xyz)")
    p.add_argument('--no-idpp', dest='idpp', action='store_false',
                   help='force-disable the IDPP pre-optimization pass')
    p.add_argument('--step-pred-method', default='amgd',
                   choices=['amgd', 'sd', 'nr', 'rfo', 'l-nr', 'l-rfo', 'sct'],
                   help="step predictor (default: 'amgd'; try 'sd' or a lower "
                        "--max-step if the climbing image oscillates instead of converging)")
    p.add_argument('--max-step', type=max_step_type, default=0.05,
                   help="max per-image step norm per iteration (default: 0.05; "
                        "pass 'none' to disable clipping entirely)")
    p.add_argument('--stepsize-fac', type=float, default=0.2,
                   help="scales the raw step every predictor takes (default: 0.2); "
                        "for 'sd' this is the only speed knob (no momentum term)")
    p.add_argument('--spring-gradient', default='difference',
                   choices=['difference', 'projected', 'raw'],
                   help="spring force definition (default: 'difference', Henkelman & "
                        "Jonsson's improved version)")
    p.add_argument('--k-const', type=float, default=0.003,
                   help='spring force constant in Eh/Angstrom^2 (default: 0.003; if '
                        '--use-vark, this is the maximum spring constant instead)')
    p.add_argument('--use-vark', action='store_true', default=False,
                   help="use Henkelman & Jonsson's variable-k spring scheme (stiffer "
                        "springs near high-energy/TS region) instead of a fixed --k-const")
    p.add_argument('--vark-min-fac', type=float, default=0.1,
                   help='only used with --use-vark: minimum spring constant is '
                        'this fraction of --k-const (default: 0.1)')
    p.add_argument('--use-analytical-springpos', action='store_true', default=False,
                   help="use the analytical spring positioning scheme instead of "
                        "force-based image spacing (not compatible with --step-pred-method sct)")
    p.add_argument('--tangents', default='henkjon', choices=['henkjon', 'simple'],
                   help="tangent estimation method (default: 'henkjon', Henkelman & "
                        "Jonsson's improved estimate)")
    p.add_argument('--maxiter', type=int, default=500)
    p.add_argument('--verbose', default='debug',
                   choices=['debug', 'info', 'warning', 'error', 'critical'])
    return p


def parse_neb_args(parser: argparse.ArgumentParser) -> argparse.Namespace:
    """Parse the command line, insisting on --start/--end or --starttraj."""
    args = parser.parse_args()
    have_ends = args.start is not None and args.end is not None
    if have_ends == (args.starttraj is not None):
        parser.error("specify either both --start and --end, or --starttraj, not both/neither")
    return args


def print_result_paths(result: dict):
    print(f"\nResults directory: {result['resultdir']}")
    print(f"Full log:           {result['output_log']}")


def print_barriers(result: dict):
    left_eV = result['left_barrier_kJmol'] / kJmol_per_eV
    right_eV = result['right_barrier_kJmol'] / kJmol_per_eV
    print(f"\nBarrier w.r.t. left end:  {result['left_barrier_kJmol']:.3f} kJ/mol "
         f"({result['left_barrier_kJmol'] * kJmol_in_kcalmol:.3f} kcal/mol, {left_eV:.4f} eV)")
    print(f"Barrier w.r.t. right end: {result['right_barrier_kJmol']:.3f} kJ/mol "
         f"({result['right_barrier_kJmol'] * kJmol_in_kcalmol:.3f} kcal/mol, {right_eV:.4f} eV)")


def print_barrier_report(result: dict):
    """CLI printout used by the run_goeneb_*.py scripts' main()."""
    print_result_paths(result)
    print_barriers(result)
