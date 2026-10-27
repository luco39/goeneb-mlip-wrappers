# GöNEB wrappers for MLIP and xTB reaction paths

Python and command-line wrappers for running [GöNEB](https://github.com/cchembio/goeneb)
nudged elastic band (NEB) calculations with machine-learned interatomic potentials and
semiempirical xTB. Each wrapper writes the GöNEB input file, runs GöNEB in-process, and hands back
the reaction barrier together with the location of every output file.

| Wrapper | Energy/gradient source | Typical use |
|---|---|---|
| `run_goeneb_mace.py` | [MACE](https://github.com/ACEsuit/mace): your own checkpoint, or the MACE-OFF / MACE-MP foundation models | NEB on a trained MACE model |
| `run_goeneb_uma.py` | Meta's [UMA](https://github.com/facebookresearch/fairchem) (fairchem-core) | NEB with a general-purpose foundation model |
| `run_goeneb_mace_ensemble.py` | An ensemble of MACE models, with the path driven by their mean | Active learning: stops as soon as the models disagree and saves the path for relabelling |

Because the wrappers drive GöNEB itself, you get GöNEB's tangents, spring forces, step predictors,
climbing image, convergence criteria and output files. Only the energy surface changes.

## Contents

```
goeneb_runner_common.py      shared code: input file, running GöNEB, barrier, shared CLI options
run_goeneb_mace.py           MACE wrapper
run_goeneb_uma.py            UMA wrapper
run_goeneb_mace_ensemble.py  MACE ensemble wrapper
goeneb/                      GöNEB, with the MACE / UMA / MACE-ensemble interfaces added
tests/                       tests that run without any ML model (see "Running the tests")
WRAPPER_CONTRACT.md          developer notes: exact behaviour the wrappers guarantee
```

`goeneb/` is a modified copy of the upstream [GöNEB repository](https://github.com/cchembio/goeneb),
taken at commit `3d59e4d` ("Better support for freezing atoms"); see upstream for its history. It adds:
- `interfaces/mace_interface.py`, `interfaces/uma_interface.py` and `interfaces/mace_ensemble_interface.py`,
  plus their settings in `neb_configparse.py` and `interfaces/engrad_interface.py`;
- a hook that lets an interface stop the optimisation early (`neb_optimizer.py`, `basic_neb.py`, `neb.py`);
- an xyz-writing fix for numpy ≥ 2 (`file_sys_io.py`);
- `settings.n_images` now follows the frame count of a `starttraj` (`neb.py`).

GöNEB is not pip-installed. `goeneb_runner_common.py` adds `goeneb/` to `sys.path`, looking for it
next to itself. To use a GöNEB checkout somewhere else, set the `GOENEB_PATH` environment variable.

## Installation

MACE and fairchem-core pin different PyTorch versions, so use **two virtual environments**:

| Environment | For | Install |
|---|---|---|
| `MACE_venv` | `run_goeneb_mace.py`, `run_goeneb_mace_ensemble.py` | `pip install mace-torch ase chemcoord matplotlib` |
| `.venv` | `run_goeneb_uma.py`, tests | `pip install fairchem-core tblite ase chemcoord matplotlib pytest` |

```bash
git clone <this-repository> goeneb-wrappers
cd goeneb-wrappers

python3 -m venv MACE_venv
MACE_venv/bin/pip install mace-torch ase chemcoord matplotlib

python3 -m venv .venv
.venv/bin/pip install fairchem-core tblite ase chemcoord matplotlib pytest
```

Tested with Python 3.13, plus:
- `MACE_venv`: mace-torch 0.3.16, torch 2.13, ase 3.29, chemcoord 2.2, numpy 2.4
- `.venv`: fairchem-core 2.19, torch 2.8, tblite 0.7, ase 3.28, chemcoord 2.2, numpy 2.4

(Upstream GöNEB's own `goeneb/requirements.txt` pins Python 3.8-era versions. The wrappers don't need those.)

Additional notes:
- **UMA** checkpoints come from a gated Hugging Face repository. Request access on Hugging Face,
  then run `huggingface-cli login` once on each machine.
- **MACE foundation models** are downloaded to `~/.cache/mace` the first time you use them.
- **chemcoord** is only needed for `interp_mode='internal'` (the default). With `cartesian` or
  `geodesic` interpolation you can leave it out.

## Quick start: command line

Every script takes either `--start` and `--end` (two end structures to interpolate between), or
`--starttraj` (an existing multi-frame xyz path to start from).

```bash
# MACE, own checkpoint
MACE_venv/bin/python run_goeneb_mace.py \
    --start tests/fixtures/start.xyz --end tests/fixtures/end.xyz \
    --mace-model path/to/model.model

# MACE foundation model
MACE_venv/bin/python run_goeneb_mace.py \
    --start tests/fixtures/start.xyz --end tests/fixtures/end.xyz \
    --mace-foundation mace_off --mace-size medium

# UMA, continuing from an existing path, on a GPU
.venv/bin/python run_goeneb_uma.py \
    --starttraj tests/fixtures/traj.xyz --uma-model uma-s-1p1 --device cuda


# MACE ensemble for active learning
MACE_venv/bin/python run_goeneb_mace_ensemble.py \
    --start tests/fixtures/start.xyz --end tests/fixtures/end.xyz \
    --model-paths m0.model,m1.model,m2.model,m3.model,m4.model \
    --threshold 0.05
```

Each run prints the results directory, the log file and the barrier from both ends, in kJ/mol,
kcal/mol and eV. `--help` on any script lists every option.

## Quick start: Python

Run from the repository root, or add it to `sys.path` first.

```python
from run_goeneb_mace import run_mace_neb

result = run_mace_neb('reactant.xyz', 'product.xyz', mace_model_path='model.model',
                      jobdir='runs/my_reaction', n_images=11, step_pred_method='sd', maxiter=300)

print(result['left_barrier_kJmol'], result['right_barrier_kJmol'])
final_path = result['resultdir'] / 'finaltraj.xyz'
```

| Function | Positional arguments |
|---|---|
| `run_goeneb_mace.run_mace_neb` | `start_xyz, end_xyz, mace_model_path, jobdir` |
| `run_goeneb_uma.run_uma_neb` | `start_xyz, end_xyz, jobdir` |
| `run_goeneb_mace_ensemble.run_mace_ensemble_neb` | `start_xyz, end_xyz, model_paths, threshold_eV, jobdir` |

All other arguments are keyword-only. Every function returns a dict:

| Key | Meaning |
|---|---|
| `left_barrier_kJmol`, `right_barrier_kJmol` | Highest image energy minus the first / last image energy, at the final iteration |
| `path_energies_Eh` | Final energy of every image, in Hartree (numpy array) |
| `jobdir`, `resultdir`, `output_log` | `Path`s to the job folder, the results folder and the full log |
| `uncertainty_triggered`, `uncertainty_info` | Ensemble only; see [MACE ensemble](#mace-ensemble-active-learning) |

Calling a wrapper several times in one notebook is fine: each run writes its own `output.log`.

## Options shared by every wrapper

Python keyword on the left, CLI flag in brackets. CLI flags use dashes, e.g. `--k-const`.

| Option | Default | Meaning |
|---|---|---|
| `start_xyz`, `end_xyz` (`--start`, `--end`) | – | End structures to interpolate between |
| `starttraj` (`--starttraj`) | – | Existing multi-frame xyz path; used *instead of* start/end. Only the element and the first three numbers of each atom line are read, so extra columns (e.g. forces) are fine |
| `jobdir` (`--jobdir`) | timestamped folder under `./goeneb_<wrapper>_runs/` | Where the job runs. It can be reused: each run gets a new `results*` folder |
| `interp_mode` | `internal` | Initial path: `internal` (z-matrix, needs chemcoord), `cartesian`, or `geodesic` (non-deterministic) |
| `n_images` | `11` | Number of *intermediate* images; GöNEB adds the two end points. Ignored with `starttraj`, which uses all of its frames |
| `climbing_image` (`--no-climbing-image` to turn off) | on | Climbing-image NEB |
| `relaxed_neb` | off | Stop after the relaxed (loose-tolerance) pre-optimisation |
| `sidpp` | off | Build the initial path with sequential IDPP instead of `interp_mode` + IDPP |
| `idpp` (`--idpp` / `--no-idpp`) | on for start/end, off for `starttraj` | Geometric IDPP smoothing of the initial path. It ignores the energy surface, so it is off by default when continuing an existing path |
| `step_pred_method` | `amgd` | Optimiser: `amgd`, `sd`, `nr`, `rfo`, `l-nr`, `l-rfo`, `sct` |
| `max_step` | `0.05` | Largest step any image may take per iteration (longer steps are scaled down); `None` (CLI: `none`) disables the limit |
| `stepsize_fac` | `0.2` | Scales every predicted step |
| `spring_gradient` | `difference` | Spring force: `difference`, `projected`, `raw` |
| `k_const` | `0.003` | Spring constant (Eh/Å²); the maximum constant when `use_vark` is on |
| `use_vark`, `vark_min_fac` | off, `0.1` | Variable spring constants, stiffer near the barrier; the softest is `vark_min_fac × k_const` |
| `use_analytical_springpos` | off | Analytical image spacing (not with `sct`) |
| `tangents` | `henkjon` | Tangent estimate: `henkjon` (improved) or `simple` |
| `maxiter` | `500` | Maximum NEB iterations |
| `verbose` | `debug` | Log level for `output.log`: `debug`, `info`, `warning`, `error`, `critical` |

`goeneb_runner_common.write_common_ini()` has a longer explanation of each option, including when to
change it. GöNEB's own [README](goeneb/README.md) documents the underlying keywords.

## Wrapper-specific options

**MACE** (`run_mace_neb`):
| Python | CLI | Default | |
|---|---|---|---|
| `mace_model_path` | `--mace-model` | – | Your checkpoint. Give either this or `mace_foundation` |
| `mace_foundation` | `--mace-foundation` | – | `mace_off` or `mace_mp` |
| `mace_size` | `--mace-size` | `medium` | Foundation model size: `small`, `medium`, `large` |
| `device`, `dtype` | `--device`, `--dtype` | `cpu`, `float64` | `cpu` / `cuda` / `mps`; `float32` / `float64` |

**UMA** (`run_uma_neb`):
| Python | CLI | Default | |
|---|---|---|---|
| `uma_model` | `--uma-model` | `uma-s-1p1` | fairchem checkpoint name |
| `uma_task` | `--uma-task` | `omol` | fairchem task (`omol` for molecules) |
| `charge`, `spin` | `--charge`, `--spin` | `0`, `1` | Total charge and multiplicity 2S+1 |
| `batch_images` | `--uma-batch` | off | All images in one forward pass. Faster, but one failing image fails the whole iteration |
| `device` | `--device` | `cpu` | `cpu` / `cuda` |


**MACE ensemble** (`run_mace_ensemble_neb`): see the next section.

## MACE ensemble (active learning)

The ensemble wrapper evaluates every image with every model and moves **one shared path** along the
models' mean energy and force. It stops when the models disagree too much. Each iteration it measures
the models' disagreement on every image and stops as soon as the worst image exceeds the threshold.
The path at that point is written to `finaltraj.xyz`, ready to relabel at a higher level of theory and
add to the training set.

| Python | CLI | Default | |
|---|---|---|---|
| `model_paths` | `--model-paths a,b,c` | required | At least 2 MACE checkpoints |
| `threshold_eV` | `--threshold` | required | Stop threshold, in the units of the metric |
| `uncertainty_metric` | `--uncertainty-metric` | `energy` | `energy`: spread (standard deviation) of the per-image energy across models, after subtracting each model's own path mean (eV). `force`: worst atom's RMS force disagreement (eV/Å); it means the same thing regardless of system size |
| `batch_images`, `max_batch_atoms` | `--batch-images`, `--max-batch-atoms` | off, `0` | One forward pass per model per iteration (much faster); optionally capped at that many atoms per batch. A batch that fails falls back to image-by-image |
| `device`, `dtype` | `--device`, `--dtype` | `cpu`, `float64` | |

Reading the result:

```python
result = run_mace_ensemble_neb('r.xyz', 'p.xyz', model_paths, 0.05, jobdir='runs/r1')
if result['uncertainty_triggered']:
    info = result['uncertainty_info']   # contents of results/ensemble_uncertainty.json
    print(f"stopped at iteration {info['iteration']}, image {info['image_index']}: "
          f"{info['max_uncertainty']:.3f} {info['units']} > {info['threshold']}")
```

`ensemble_uncertainty.json` is also written for runs that never trigger. In that case it records the
largest disagreement seen (`max_uncertainty_seen`, `max_uncertainty_seen_iteration`), which helps when
choosing a threshold. Its `triggered` field tells the two cases apart.

A triggered run **has not converged**: its barrier describes an intermediate path only.

Importing `run_goeneb_mace_ensemble` seeds numpy's global random generator with 343, so geodesic
interpolation is reproducible between ensemble runs.

## Output files

```
<jobdir>/
├── neb.ini            input file generated by the wrapper (can be rerun with GöNEB directly)
├── output.log         full GöNEB log
├── _tmp/              scratch space
└── results/           results1/, results2/, ... if the job folder is reused
    ├── optlog.csv           one row per iteration: convergence measures and every image energy
    ├── starttraj.xyz        initial path
    ├── currenttraj.xyz      path at the latest iteration (updated while the run is going)
    ├── finaltraj.xyz        final path
    ├── HEI.xyz, HEI_trj.xyz highest-energy image (final, and one frame per iteration)
    ├── TS_guess_cubic.xyz, TS_guess_cubic_trj.xyz
    │                        cubic-interpolated transition-state guess (only without climbing image)
    └── ensemble_uncertainty.json   (ensemble wrapper only)
```



## Credits

- **GöNEB** was written by Björn Hein-Janke and is maintained by Lynn Meeder (University of Göttingen):
  <https://github.com/cchembio/goeneb>. Please cite it and the methods you use (listed in
  [goeneb/README.md](goeneb/README.md#references)). Check its repository for licence terms before
  redistributing `goeneb/`.
- **MACE**: Batatia et al., <https://github.com/ACEsuit/mace>
- **UMA / fairchem**: Meta FAIR Chemistry, <https://github.com/facebookresearch/fairchem>
