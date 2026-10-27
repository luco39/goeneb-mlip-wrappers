import numpy as np
from helper import pvec_to_crdmat, crdmat_to_pvec


def center_on_centroid(pvec, shift=None, return_centroid=False):
    """
    Recenter all coordinates in pvec so the centroid is at the origin.
    """
    crdmat = pvec_to_crdmat(pvec)
    center = crdmat.mean(axis=0)
    if shift is None:
        crdmat = crdmat - center
    else:
        crdmat = crdmat - shift
    if return_centroid:
        return crdmat_to_pvec(crdmat), center
    # normally return only pvec
    return crdmat_to_pvec(crdmat)


def rmsd(vec1, vec2):
    return np.sqrt(np.mean((vec1 - vec2)**2))


def align_vectors(a, b, weights=None):
    """
    Aligns vector a onto vector b using Kabsch algorithm. Expects two vectors in 
    pvec format (flattened arrays),
    optionally takes a weights argument (list or array of weights for each atom).
    Expects (and only works if) both vectors are centered on their centroids.

    Returns:
    - vector a rotated to be aligned to vector b (flattened) (a @ C)
    - the rotation matrix C (3 x 3)

    Function adapted from the scipy.spatial.transform.Rotation function align_vectors().
    """
    # Check input vectors
    a_crdmat = pvec_to_crdmat(np.array(a, dtype=float))
    b_crdmat = pvec_to_crdmat(np.array(b, dtype=float))

    if a_crdmat.shape != b_crdmat.shape:
        raise ValueError("Expected inputs `a` and `b` to have same shapes, "
                         "got {} and {} respectively.".format(
                         a_crdmat.shape, b_crdmat.shape))
    N = len(a_crdmat)

    # Check weights
    if weights is None:
        weights = np.ones(N)
    else:
        weights = np.array(weights, dtype=float)
        if weights.ndim != 1:
            raise ValueError("Expected `weights` to be 1 dimensional, got "
                             "shape {}.".format(weights.shape))
        if N > 1 and (weights.shape[0] != N):
            raise ValueError("Expected `weights` to have number of values "
                             "equal to number of input vectors, got "
                             "{} values and {} vectors.".format(
                             weights.shape[0], N))
        if (weights < 0).any():
            raise ValueError("`weights` may not contain negative values")

    # Note that einsum('ji,jk->ik', X, Y) is equivalent to np.dot(X.T, Y)
    B = np.einsum('ji,jk->ik', weights[:, None] * a_crdmat, b_crdmat)
    u, s, vh = np.linalg.svd(B)

    # Correct improper rotation if necessary (as in Kabsch algorithm)
    if np.linalg.det(u @ vh) < 0:
        s[-1] = -s[-1]
        u[:, -1] = -u[:, -1]

    # Rotation matrix C
    C = np.dot(u, vh)

    if s[1] + s[2] < 1e-16 * s[0]:
        print("Optimal rotation is not uniquely or poorly defined "
              "for the given sets of vectors.")

    return (a_crdmat @ C).flatten(), C

def align_path(pvecs, rot_align_mode, translate_mode='individual'):
    # First center all the images on their centroids
    if translate_mode == 'individual':
        centered_pvecs = [center_on_centroid(pvec) for pvec in pvecs]
    elif translate_mode == 'collective':
        _, centroid = center_on_centroid(pvecs[0], return_centroid=True)
        centered_pvecs = [center_on_centroid(pvec, shift=centroid) for pvec in pvecs]
    else: # no translation
        centered_pvecs = pvecs

    # Do rotational alignement if desired
    if rot_align_mode is None:
        return centered_pvecs

    # First image is not rotated
    new_structures = [centered_pvecs[0]]
    if rot_align_mode == 'pairwise':
        # Align each image to the image that was before
        for i in range(1, len(pvecs)):
            aligned_struct, C = align_vectors(centered_pvecs[i], new_structures[i-1])
            new_structures.append(aligned_struct)
    elif rot_align_mode == 'single_reference':
        # Align each image to the start structure
        for i in range(1, len(pvecs)):
            aligned_struct, C = align_vectors(centered_pvecs[i], new_structures[0])
            new_structures.append(aligned_struct)
    else:
        raise ValueError('Error in struct_aligner interpolator_module: "' +
                           str(rot_align_mode) + 
                           '" is not a valid rotation alignment mode.')
    return new_structures
