import time
import logging

import numpy as np
from ase import Atoms

logger = logging.getLogger(__name__)

eV_in_Eh = 27.211386245988  # Hartree per eV

# The following is the implementation of the UMA interface, using Meta's
# fairchem-core (FAIRChemCalculator + pretrained_mlip). Same shape as
# mace_interface.py: the predictor is loaded once in engrad_interface.setup_uma()
# and handed in here via engrfunc_kwargs, so the (expensive) model load only
# happens once for the whole NEB run.
#
# Unlike MACE, UMA's 'omol' task reads charge/spin off atoms.info rather than
# ignoring them -- normally ASE populates atoms.info from an extxyz comment
# line (e.g. 'charge=0 spin=1'), but since we build Atoms directly here we set
# atoms.info explicitly from the settings.
# ---------------------------------------------------------


def uma_path_engrads(image_pvecs, labels, series_name, predictor, task_name, charge, spin, batch=False):
    """
    This function calculates the UMA energy and gradient for all images.
    - image_pvecs: list of the flattened image vectors
    - labels: list of the atom types
    - series_name: prefix used for logging image names
    - predictor: a pre-loaded fairchem-core predict unit (from pretrained_mlip.get_predict_unit)
    - task_name: the fairchem task, e.g. 'omol' for molecular systems
    - charge, spin: charge and spin of the system, threaded into atoms.info
    - batch: if True, all images are packed into a single AtomicData batch and
      sent through the predictor in one forward pass instead of looping image
      by image. Faster (fewer, bigger GPU calls), but note the fault-isolation
      tradeoff documented on _uma_path_engrads_batched().
    """
    if batch:
        return _uma_path_engrads_batched(image_pvecs, labels, series_name, predictor, task_name, charge, spin)

    start_time = time.perf_counter()
    image_names = [f"{series_name}{i+1}" for i in range(len(image_pvecs))]

    results = [calculate_uma_engrad(image, labels, inpfile_name, predictor, task_name, charge, spin)
               for image, inpfile_name in zip(image_pvecs, image_names)]

    energies, engrads = zip(*results)
    elapsed = time.perf_counter() - start_time
    logger.debug('Energies: %s', energies)
    logger.debug('UMA engrad calculation for %d images took %.3f s (%.3f s/image)',
                 len(image_pvecs), elapsed, elapsed / max(len(image_pvecs), 1))
    return np.array(energies), np.array(engrads)


def _uma_path_engrads_batched(image_pvecs, labels, series_name, predictor, task_name, charge, spin):
    """
    Same as uma_path_engrads, but packs every image into a single AtomicData
    batch (fairchem.core.datasets.atomic_data.AtomicData.from_ase +
    atomicdata_list_to_batch) and calls predictor.predict() once, following
    the batched-inference pattern documented at
    https://fair-chem.github.io/batch-inference (mirroring the a2g_kwargs
    FAIRChemCalculator itself uses in ase_calculator.py, minus the
    r_edges/max_neigh/radius overrides tied to external_graph_gen, which are
    left at AtomicData.from_ase's own defaults here).

    Important tradeoff vs. the per-image loop: a single failure anywhere in
    the batch (e.g. one pathological geometry) currently fails the whole
    batch call, marking *every* image in this call as failed, rather than
    just the one bad image. The per-image loop is more fault-tolerant;
    this is faster but all-or-nothing per call.
    """
    from fairchem.core.datasets.atomic_data import AtomicData, atomicdata_list_to_batch

    start_time = time.perf_counter()
    image_names = [f"{series_name}{i+1}" for i in range(len(image_pvecs))]

    atoms_list = []
    for pvec in image_pvecs:
        atoms = Atoms(symbols=labels, positions=pvec.reshape(-1, 3))
        atoms.info['charge'] = charge
        atoms.info['spin'] = spin
        atoms_list.append(atoms)

    a2g_kwargs = {'task_name': task_name, 'r_data_keys': ['spin', 'charge']}
    target_dtype = getattr(getattr(predictor, 'inference_settings', None),
                          'base_precision_dtype', None)
    if target_dtype is not None:
        a2g_kwargs['target_dtype'] = target_dtype

    try:
        for name in image_names:
            logger.info('Now calculating (batched): ' + name)
        atomic_data_list = [AtomicData.from_ase(atoms, **a2g_kwargs) for atoms in atoms_list]
        graph_batch = atomicdata_list_to_batch(atomic_data_list)
        preds = predictor.predict(graph_batch)

        energies_eV = preds['energy'].detach().cpu().numpy()
        forces_eV_A = preds['forces'].detach().cpu().numpy()
        batch_index = graph_batch.batch.detach().cpu().numpy()
    except Exception as e:
        logger.warning(f"Batched UMA engrad calculation failed for {series_name}: {e}")
        energies = [None] * len(image_pvecs)
        engrads = [np.zeros_like(pvec) for pvec in image_pvecs]
    else:
        energies = []
        engrads = []
        for i, pvec in enumerate(image_pvecs):
            energies.append(float(energies_eV[i]) / eV_in_Eh)
            forces_i = forces_eV_A[batch_index == i]
            # engrad is the negative of the force, converted eV/Angstrom -> Eh/Angstrom
            engrads.append(-forces_i.flatten() / eV_in_Eh)

    elapsed = time.perf_counter() - start_time
    logger.debug('Energies: %s', energies)
    logger.debug('UMA batched engrad calculation for %d images took %.3f s (%.3f s/image)',
                 len(image_pvecs), elapsed, elapsed / max(len(image_pvecs), 1))
    return np.array(energies), np.array(engrads)


def calculate_uma_engrad(pvec, labels, inpfile_name, predictor, task_name, charge, spin):
    """
    This function does the complete energy and gradient calculation for
    one image using the given UMA predictor.
    """
    from fairchem.core import FAIRChemCalculator

    coords = pvec.reshape(-1, 3)
    atoms = Atoms(symbols=labels, positions=coords)
    atoms.info['charge'] = charge
    atoms.info['spin'] = spin
    atoms.calc = FAIRChemCalculator(predictor, task_name=task_name)

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
