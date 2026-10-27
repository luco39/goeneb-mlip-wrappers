import numpy as np
import logging

logger = logging.getLogger(__name__)

# This module contains the function to calculate NEB gradients
# from engrads, springgrads, and tangent vectors. It also contains
# methods to remove rotations and translations from the NEB gradient
# vectors. Any new methods relating to that should be added in this
# module.

from helper import reject, pvec_to_crdmat, crdmat_to_pvec, project


# Helper functions for rejecting translations from gradient vectors
# --------------------------------------------------------------------------


def reject_translation(pvec, axis=0):
    """Removes the component of 'pvec' corresponding to translation along a specified Cartesian axis.

    0 : x
    1 : y
    2 : z
    """
    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0 (x), 1 (y), or 2 (z)")

    axis_vec = np.zeros(3)
    axis_vec[axis] = 1.0
    natoms = int(len(pvec) / 3)
    unitvec = np.array([axis_vec] * natoms).flatten()
    unitvec /= np.linalg.norm(unitvec)
    return reject(pvec, unitvec)


def reject_translations(gradvec):
    """Remove all translations along x, y or z coordinate from the vector."""
    gradvec = reject_translation(gradvec, axis=0)
    gradvec = reject_translation(gradvec, axis=1)
    gradvec = reject_translation(gradvec, axis=2)
    return gradvec


# Helper functions for rejecting rotations from gradient vectors.
# --------------------------------------------------------------------------


def reject_rotation(deltavec, pvec, axis=0):
    """
    Removes the component of 'deltavec' corresponding to rigid rotation 
    around the specified Cartesian axis for the given structure 'pvec'.

    0 : x
    1 : y
    2 : z"""
    crdmat = pvec_to_crdmat(pvec)
    # synthesize the 'unit vector' for rotation around x axis
    unitvec = crdmat.copy()
    # remove x component of all atomic coordinates
    unitvec[:, axis] = 0.0
    axis_vec = np.zeros(3)
    axis_vec[axis] = 1.0
    unitvec = np.cross(unitvec, axis_vec)
    norm = np.linalg.norm(unitvec)
    if norm != 0.0:
        # if the norm is 0, reject won't do anything anyway
        unitvec /= norm
    # with the unit vector constructed, reject it from deltavec
    return reject(deltavec, crdmat_to_pvec(unitvec))


def reject_rotations(gradvec, struct_pvec):
    """Remove all rotations from gradvec at the structure struct_pvec."""
    gradvec = reject_rotation(gradvec, struct_pvec, axis=0)
    gradvec = reject_rotation(gradvec, struct_pvec, axis=1)
    gradvec = reject_rotation(gradvec, struct_pvec, axis=2)
    return gradvec


# Helper functions for freezing atom indices
# --------------------------------------------------------------------------

def freeze_atom_indices(vector, atom_indices):
    """this function takes in a vector of length (3*natoms), 
    it can be a gradient or step vector, and it zeros 
    all entries corresponding to the given atomic indices. 
    Note, atom 1 would have index 0!"""
    matshape_vector = pvec_to_crdmat(vector)
    for index in atom_indices:
        matshape_vector[index] = 0.0
    edited_vector = crdmat_to_pvec(matshape_vector)
    return edited_vector


# Functions for calculating and sanitizing the NEB gradients
# --------------------------------------------------------------------------


def sanitize_stepvecs(vectors,
                      pvecs=None,
                      reject_transl=True,
                      reject_rot=False):
    """Remove translations and rotations from the given vectors. If 
    rotations should also be removed from a gradient/step vector, 
    the structures (pvecs) have to be provided."""
    if reject_transl:
        vectors = [reject_translations(vector) for vector in vectors]
    if reject_rot:
        assert pvecs is not None, "pvecs must be provided when reject_rot is True"
        vectors = [reject_rotations(vector, pvec) for vector, pvec 
                   in zip(vectors, pvecs)]
    return vectors


def calculate_nebgrads(engrads,
                       sprgrads,
                       tanvecs,
                       ci_index=None):
    """Calculate the NEB gradient based on the orthogonal 
    energy gradient and the provided spring gradients. 
    Also accounts for the climbing image gradient."""
    engrads   = [np.array(g) for g in engrads]
    tanvecs   = [np.array(t) for t in tanvecs]
    sprgrads  = [np.array(s) for s in sprgrads]
    orth_engrads = [reject(engrad, tanvec) for engrad, tanvec
                    in zip(engrads, tanvecs)]
    nebgrads = [orth_en + par_spr for orth_en, par_spr
                in zip(orth_engrads, sprgrads)]
    # replace the nebgrad at ci_index with the climbing image gradient
    if ci_index is not None:
        ci_par_engrad = project(engrads[ci_index], tanvecs[ci_index])
        nebgrads[ci_index] = orth_engrads[ci_index] - ci_par_engrad
    return nebgrads
