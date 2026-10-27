import numpy as np
import logging

# The chemcoord module imports numba, which produces
# bytecode logging output on the debug level that is not useful
logging.getLogger('numba').setLevel(logging.WARNING)

from helper import pvec_to_crdmat, crdmat_to_pvec

logger = logging.getLogger(__name__)

try:
    import chemcoord as cc
    from chemcoord.exceptions import UndefinedCoordinateSystem
except ImportError:
    cc = None
    UndefinedCoordinateSystem = None
    print("Chemcoord could not be imported. This is important for the internal interpolation.")

try:
    import pandas as pd
except ImportError:
    pd = None
    print("Pandas could not be imported. This is important for the internal interpolation.")


def total_atom_movement(path_pvecs):
    """
    Returns the total atom movement as the sum of the 
    eucledean norms between image vectors.
    """
    path_pvecs = np.array(path_pvecs)
    diffs = np.diff(path_pvecs, axis=0)
    movements = np.linalg.norm(diffs, axis=1)
    return np.sum(movements)


def pvec_to_cartesian(labels, pvec):
    """
    Convert a pvec to the cartesian object used by chemoord.
    """
    assert cc is not None, "chemcoord is needed for internal interpolation"
    assert pd is not None, "pandas is needed for internal interpolation"
    coords = pvec_to_crdmat(pvec)
    frame = pd.DataFrame({'atom': labels, 'x': coords[:,0], 'y': coords[:,1], 'z': coords[:,2]})
    return cc.Cartesian(frame=frame)


def cartesian_to_pvec(cartesian):
    crdmat = np.array(cartesian[['x', 'y', 'z']])
    return crdmat_to_pvec(crdmat)


def pvec_to_zmat(labels, pvec, constr_table=None):
    assert UndefinedCoordinateSystem is not None, "chemcoord is needed for internal interpolation"
    cc_cart = pvec_to_cartesian(labels, pvec)

    if constr_table is None:
        try:
            zmat = cc_cart.get_zmat()
            constr_table = zmat.loc[:, ['b', 'a', 'd']]

        # this catches for example linear molecules
        except UndefinedCoordinateSystem as e:
            logger.warning("ChemCoord could not create Z-matrix: %s", e)
            return None, None

    else:
        zmat = cc_cart.get_zmat(constr_table)

    return zmat, constr_table


def zmat_to_pvec(zmat):
    cc_cart = zmat.get_cartesian()

    # sort by original atom index, to undo the atom shuffling
    # that occurred during z matrix construction
    cc_cart.sort_index(inplace=True)
    pvec = cartesian_to_pvec(cc_cart)
    return pvec


def zmat_interpolate(start_pvec,
                     stop_pvec,
                     labels,
                     interpolations,
                     use_2nd_constr_table=False):
    """Do an interpolation in z-matrix coordinates using the ChemCoord module.
    Expects:
    - start_pvec: coordinates of start structure
    - end_pvec
    - labels: A list of the atom types
    - interpolations: list of numbers between 0 and 1 for the new interpolations
    - use_2nd_constr_table: True/False to choose the other construction table for a possibly better fit

    Returns the interpolated images in pvec format and the construction table.
    Returns None if the z-mat construction failed.
    """
    assert cc is not None, "chemcoord is needed for internal interpolation"
    if use_2nd_constr_table:
        zmat2, ctable = pvec_to_zmat(labels, stop_pvec)
        zmat1, ctable = pvec_to_zmat(labels, start_pvec, constr_table=ctable)
    else:
        # use 1st construction table instead
        zmat1, ctable = pvec_to_zmat(labels, start_pvec)
        zmat2, ctable = pvec_to_zmat(labels, stop_pvec, constr_table=ctable)

    if any(zmat is None for zmat in [zmat1, zmat2]):
        return None, None

    # perform the actual interpolation
    with cc.TestOperators(False):
        interp_zmats = [zmat1 + (zmat2 - zmat1).minimize_dihedrals() * fac
                        for fac in interpolations]

    # convert to pvecs and return
    interp_pvecs = [zmat_to_pvec(zmat) for zmat in interp_zmats]

    return np.array(interp_pvecs), ctable