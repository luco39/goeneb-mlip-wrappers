import numpy as np
import logging

logger = logging.getLogger(__name__)

bohr = 0.529177211  # angs per bohr

element_to_atomic_number = {'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10, 'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50, 'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54}

# The following is the implementation of the xtb/tblite interface
# ---------------------------------------------------------

def tblite_path_engrads(image_pvecs,
                        labels,
                        series_name,
                        method_keywords,
                        charge,
                        spin):
    """
    This function calculates the tblite energy and gradient for all images.
    - image_pvecs: list of the flattened image vectors
    - labels: list of the atom types
    - method_keywords: xTB method input
    - charge, spin: charge and spin of the system
    """
    image_names = [f"{series_name}{i+1}" for i in range(len(image_pvecs))]
    results = [calculate_tblite_engrad(image,
                                       labels,
                                       inpfile_name,
                                       method_keywords,
                                       charge,
                                       spin)
                                       for image, inpfile_name in zip(image_pvecs, image_names)]

    energies, engrads = zip(*results)
    logger.debug('Energies: %s', energies)
    return np.array(energies), np.array(engrads)

def calculate_tblite_engrad(pvec,
                            labels,
                            inpfile_name,
                            method_keywords,
                            charge=0,
                            spin=1):
    """
    This function does the complete energy and gradient calculation for one image in tblite.
    """
    # Import tblite here, so the program runs without
    try:
        import tblite.interface as tb
    except ImportError:
        logger.error('Tblite interface selected, but the module can not be imported.')
        raise ImportError

    coords = pvec.reshape(-1,3) / bohr
    num_labels = [element_to_atomic_number[symbol] for symbol in labels]
    uhf = int(spin) - 1
    calc = tb.Calculator(method_keywords, num_labels, coords, int(charge), uhf)
    try:
        logger.info('Now calculating: ' + inpfile_name)
        results = calc.singlepoint()
    except Exception as e:
        logger.warning(f"Engrad calculation failed for {inpfile_name}: {e}")
        energy =  None
        grads = np.zeros_like(pvec)
    else:
        energy = results['energy']
        grads = results['gradient'].flatten() / bohr # convert engrad to Eh/Ang
    return energy, grads