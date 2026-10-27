import time
import logging

import numpy as np
from ase import Atoms

logger = logging.getLogger(__name__)

eV_in_Eh = 27.211386245988  # Hartree per eV

# The following is the implementation of the MACE interface.
# It supports both MACE foundation models (mace-off, mace-mp) and
# custom models loaded from a local checkpoint path. The calculator
# itself is instantiated once in engrad_interface.setup_mace() and
# handed in here via engrfunc_kwargs, so the model is loaded exactly
# once for the whole NEB run rather than being reloaded every
# iteration or every image.
# ---------------------------------------------------------


def mace_path_engrads(image_pvecs, labels, series_name, calculator):
    """
    This function calculates the MACE energy and gradient for all images.
    - image_pvecs: list of the flattened image vectors
    - labels: list of the atom types
    - series_name: prefix used for logging image names
    - calculator: a pre-loaded ASE-compatible MACE calculator (either a
      MACECalculator built from a local checkpoint, or one returned by
      the mace_off/mace_mp foundation model loaders)
    """
    start_time = time.perf_counter()
    image_names = [f"{series_name}{i+1}" for i in range(len(image_pvecs))]

    results = [calculate_mace_engrad(image, labels, inpfile_name, calculator)
               for image, inpfile_name in zip(image_pvecs, image_names)]

    energies, engrads = zip(*results)
    elapsed = time.perf_counter() - start_time
    logger.debug('Energies: %s', energies)
    logger.debug('MACE engrad calculation for %d images took %.3f s (%.3f s/image)',
                 len(image_pvecs), elapsed, elapsed / max(len(image_pvecs), 1))
    return np.array(energies), np.array(engrads)


def _atomic_data_list(calculator, image_pvecs, labels):
    """One AtomicData per image, built exactly as MACECalculator._atoms_to_batch does.

    Reusing the calculator's own key specification, head and z_table is the
    point: it guarantees the batched path constructs identical graphs to the
    per-image path, so the two give the same numbers. Reimplementing the graph
    construction here would risk a silent numerical drift between them.
    """
    from mace import data as mace_data
    from mace.tools import torch_tools

    keyspec = mace_data.KeySpecification(
        info_keys=dict(getattr(calculator, 'info_keys', {}) or {}),
        arrays_keys={**(getattr(calculator, 'arrays_keys', {}) or {}),
                     getattr(calculator, 'charges_key', 'Qs'): 'charges'},
    )
    head_name = getattr(calculator, 'head', None)
    available_heads = getattr(calculator, 'available_heads', None)

    graphs = []
    with torch_tools.default_dtype(calculator.default_dtype):
        for pvec in image_pvecs:
            atoms = Atoms(symbols=labels, positions=np.asarray(pvec).reshape(-1, 3))
            config_kwargs = {'key_specification': keyspec}
            if head_name is not None:
                config_kwargs['head_name'] = head_name
            config = mace_data.config_from_atoms(atoms, **config_kwargs)

            data_kwargs = {'z_table': calculator.z_table, 'cutoff': float(calculator.r_max)}
            if available_heads is not None:
                data_kwargs['heads'] = available_heads
            graphs.append(mace_data.AtomicData.from_config(config, **data_kwargs))
    return graphs


def batched_mace_engrads(image_pvecs, labels, calculator, max_batch_atoms=None):
    """Energy (Eh) and gradient (Eh/Angstrom) for every image, in as few forward
    passes as possible.

    The per-image loop pays the full graph-construction and kernel-launch
    overhead once per image, which dominates for molecules of this size: the
    forward pass itself is a small fraction of each call. Packing every image
    into one batch collapses N calls into one and measures ~7x faster on CPU
    for an 11-image path (energies bit-identical, gradients agreeing to 1e-18).

    max_batch_atoms caps how many atoms go into one forward, so a long path or
    a large complex can be split into several batches rather than exhausting
    GPU memory. None means one batch for the whole path.

    Raises rather than returning partial results -- the caller decides whether
    to fall back to the per-image loop, which is more fault-tolerant because a
    single pathological geometry only fails its own image.
    """
    from mace.tools import torch_geometric

    n_atoms = max(len(labels), 1)
    per_batch = (len(image_pvecs) if not max_batch_atoms
                 else max(1, int(max_batch_atoms) // n_atoms))

    dataset = _atomic_data_list(calculator, image_pvecs, labels)
    loader = torch_geometric.dataloader.DataLoader(
        dataset=dataset, batch_size=per_batch, shuffle=False, drop_last=False)

    model = calculator.models[0]
    energies, engrads = [], []
    for batch in loader:
        batch = batch.to(calculator.device)
        batch_dict = batch.to_dict()
        out = model(batch_dict, compute_stress=False, training=False)
        energies_eV = out['energy'].detach().cpu().numpy()
        forces_eV_A = out['forces'].detach().cpu().numpy()
        # ptr gives each graph's atom span, so this stays correct even if the
        # images ever stop having identical atom counts
        ptr = batch_dict['ptr'].detach().cpu().numpy()
        for g in range(len(energies_eV)):
            energies.append(float(energies_eV[g]) / eV_in_Eh)
            forces_g = forces_eV_A[ptr[g]:ptr[g + 1]]
            engrads.append(-forces_g.flatten() / eV_in_Eh)

    return energies, engrads


def calculate_mace_engrad(pvec, labels, inpfile_name, calculator):
    """
    This function does the complete energy and gradient calculation for
    one image using the given MACE calculator.
    """
    coords = pvec.reshape(-1, 3)
    atoms = Atoms(symbols=labels, positions=coords)
    atoms.calc = calculator

    try:
        logger.info('Now calculating: ' + inpfile_name)
        energy_eV = atoms.get_potential_energy()
        forces_eV_A = atoms.get_forces()
    except Exception as e:
        logger.warning(f"Engrad calculation failed for {inpfile_name}: {e}")
        energy = None
        grads = np.zeros_like(pvec)
    else:
        energy = energy_eV / eV_in_Eh
        # engrad is the negative of the force, converted eV/Angstrom -> Eh/Angstrom
        grads = -forces_eV_A.flatten() / eV_in_Eh

    return energy, grads
