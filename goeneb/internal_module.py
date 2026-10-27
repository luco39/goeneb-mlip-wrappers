import numpy as np
import logging

logger = logging.getLogger(__name__)

try:
    from numba import njit
except ImportError:
    logger.warning("The numba module can not be imported. This way Lindh hessian generation might be very slow.")
    def njit(func):
        return func

# Helper functions

@njit
def zijk(i = None, j = None, k = None):
    """
    Sign factor function (like Bakken and Helgakar)
    zeta(i, j, k) = delta(i, j) - delta(i, k)
    'Is i equal to j (1), equal to k (-1) or equal to none (0)?'
    """
    return int(i == j) - int(i == k)

@njit
def dij(a = None, b = None):
    """
    Kronecker delta
    """
    return int(a == b)

@njit
def norm_is_ok(vec, normmin=1.0e-8, normmax=1.0e+15):
    """
    Takes min and max, and checks whether the norm
    of the vector given is between them.
    """
    norm = np.linalg.norm(vec)
    return normmin <= norm <= normmax

@njit
def are_parallel(v1, v2, tol=1e-8):
    """
    Returns True if v1 and v2 are parallel or anti-parallel.
    """
    v1 = np.asarray(v1)
    v2 = np.asarray(v2)
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    # The cross product will be zero if vectors are parallel
    return np.linalg.norm(np.cross(v1, v2)) < tol

def bend_deriv(flatcoords, atom_ids):
    """
    Get derivative of the bend according to Bakken
    """
    assert flatcoords is not None

    testvec1 = np.array([1.0, -1.0, 1.0])
    testvec2 = np.array([-1.0, 1.0, 1.0])

    # initialize derivative
    retval = np.zeros((9,))
    a, b, c = atom_ids

    # normalized vectors
    uvec = flatcoords[a*3:a*3+3] - flatcoords[b*3:b*3+3]
    vvec = flatcoords[c*3:c*3+3] - flatcoords[b*3:b*3+3]
    ulen, vlen = np.linalg.norm(uvec), np.linalg.norm(vvec)
    uvec, vvec = uvec/ulen, vvec/vlen

    # get w-vector
    if not are_parallel(uvec, vvec):
        wvec = np.cross(uvec, vvec)
    else:
        if not are_parallel(uvec, testvec1):
            wvec = np.cross(uvec, testvec1)
        else:
            wvec = np.cross(uvec, testvec2)
    wlen = np.linalg.norm(wvec)
    wvec = wvec/wlen

    # calculate derivative
    uwcross = np.cross(uvec, wvec)
    wvcross = np.cross(wvec, vvec)
    retval[0:3] = uwcross/ulen
    retval[3:6] = -uwcross/ulen - wvcross/vlen
    retval[6:] = wvcross/vlen
    return retval

def convert_gradient(cart_grad, Bmat):
    """
    Convert cartesian gradient into internal coordinates.
    """
    Bplus = np.linalg.pinv(Bmat)
    int_grad = Bplus.T @ cart_grad
    P = Bmat @ Bplus
    return P @ int_grad

# ----------------------------------------------------------
#                       B-Matrix
# ----------------------------------------------------------

def calcBmat(flatcoords = None, coord_list = None):
    """ 
    calcBmat() calculates Wilson's B matrix from already determined primitive internal coordinates
    which should be given in coord_list. There all stretches, bends and twists should be indicates
    as tuples of the atom indices.

    Returns a numpy 2D array (nintcoords, natoms*3) with the derivatives (B-matrix).
    """
    assert flatcoords is not None
    assert coord_list is not None

    # number of primitive internal coordinates to initialize the new B-Matrix
    pic_num     = len(coord_list)
    newBmat     = np.zeros((pic_num, flatcoords.shape[0]), dtype=np.float64)

    curr_coord  = 0
    for atom_ids in coord_list:

        # A bond
        if len(atom_ids) == 2:
            atom_a, atom_b = atom_ids
            # Bondvector, normalized
            uvec = (flatcoords[atom_b*3:atom_b*3+3] - flatcoords[atom_a*3:atom_a*3+3])
            ulen = np.linalg.norm(uvec)
            uvec /= ulen

            # Define entries of B for atom A and B
            newBmat[curr_coord, atom_a*3]   = -uvec[0]
            newBmat[curr_coord, atom_a*3+1] = -uvec[1]
            newBmat[curr_coord, atom_a*3+2] = -uvec[2]
            newBmat[curr_coord, atom_b*3]   = uvec[0]
            newBmat[curr_coord, atom_b*3+1] = uvec[1]
            newBmat[curr_coord, atom_b*3+2] = uvec[2]

        # An angle
        elif len(atom_ids) == 3:
            atom_a, atom_b, atom_c = atom_ids
            dpdx = bend_deriv(flatcoords, atom_ids=atom_ids)

            # New entries for B based on that derivative
            newBmat[curr_coord, atom_a*3:atom_a*3+3] = dpdx[0:3]
            newBmat[curr_coord, atom_b*3:atom_b*3+3] = dpdx[3:6]
            newBmat[curr_coord, atom_c*3:atom_c*3+3] = dpdx[6:9]

        # A dihedral
        elif len(atom_ids) == 4:
            atom_a, atom_b, atom_c, atom_d = atom_ids
            atom_c  = atom_ids[2] # p
            atom_d  = atom_ids[3] # n

            # Connection vectors and their normalized versions
            uvec    = flatcoords[atom_a*3:atom_a*3+3] - flatcoords[atom_b*3:atom_b*3+3]
            wvec    = flatcoords[atom_c*3:atom_c*3+3] - flatcoords[atom_b*3:atom_b*3+3]
            vvec    = flatcoords[atom_d*3:atom_d*3+3] - flatcoords[atom_c*3:atom_c*3+3]
            ulen    = np.linalg.norm(uvec)
            vlen    = np.linalg.norm(vvec)
            wlen    = np.linalg.norm(wvec)
            uvec   /= ulen
            wvec   /= wlen
            vvec   /= vlen

            # Some cross products (perpendicular to both vectors)
            # and dot products (length of projected vector = cosine)
            uwcross = np.cross(uvec, wvec)
            vwcross = np.cross(vvec, wvec)
            cos_u   = np.dot(uvec, wvec)
            cos_v   = -np.dot(vvec, wvec)
            sin2_u   = 1.0 - cos_u**2
            sin2_v   = 1.0 - cos_v**2

            # If the square sine is too small, the torsion would be too close to 0° or 180°
            if sin2_u <= 1.0e-12 or sin2_v <= 1.0e-12:
                curr_coord += 1
                print('Found torsion with linear angle ' +
                      'for atoms %s, %s, %s, %s. Not calculating derivative!', atom_a, atom_b, atom_c, atom_d)
                continue

            # Get new B-Matrix elements
            else:
                newBmat[curr_coord, atom_a*3]   = uwcross[0]/(ulen * sin2_u)
                newBmat[curr_coord, atom_a*3+1] = uwcross[1]/(ulen * sin2_u)
                newBmat[curr_coord, atom_a*3+2] = uwcross[2]/(ulen * sin2_u)
                newBmat[curr_coord, atom_b*3]   = -uwcross[0]/(ulen * sin2_u) + uwcross[0]*cos_u/(sin2_u * wlen) + vwcross[0]*cos_v/(sin2_v * wlen)
                newBmat[curr_coord, atom_b*3+1] = -uwcross[1]/(ulen * sin2_u) + uwcross[1]*cos_u/(sin2_u * wlen) + vwcross[1]*cos_v/(sin2_v * wlen)
                newBmat[curr_coord, atom_b*3+2] = -uwcross[2]/(ulen * sin2_u) + uwcross[2]*cos_u/(sin2_u * wlen) + vwcross[2]*cos_v/(sin2_v * wlen)
                newBmat[curr_coord, atom_c*3]   = vwcross[0]/(vlen * sin2_v) - uwcross[0]*cos_u/(sin2_u * wlen) - vwcross[0]*cos_v/(sin2_v * wlen)
                newBmat[curr_coord, atom_c*3+1] = vwcross[1]/(vlen * sin2_v) - uwcross[1]*cos_u/(sin2_u * wlen) - vwcross[1]*cos_v/(sin2_v * wlen)
                newBmat[curr_coord, atom_c*3+2] = vwcross[2]/(vlen * sin2_v) - uwcross[2]*cos_u/(sin2_u * wlen) - vwcross[2]*cos_v/(sin2_v * wlen)
                newBmat[curr_coord, atom_d*3]   = -vwcross[0]/(vlen * sin2_v)
                newBmat[curr_coord, atom_d*3+1] = -vwcross[1]/(vlen * sin2_v)
                newBmat[curr_coord, atom_d*3+2] = -vwcross[2]/(vlen * sin2_v)
        curr_coord += 1

    return newBmat

# ----------------------------------------------------------
#                       K-Matrix
# ----------------------------------------------------------

@njit
def kmat_stretch(atom_ids, flatcoords, n_cart, grad_int_id):
    tmpmat  = np.zeros((n_cart, n_cart), dtype = np.float64)
    # Get the normalized bond vector
    if len(atom_ids) == 2:
        atom_a, atom_b = atom_ids
        vec = flatcoords[atom_b*3:atom_b*3+3] - flatcoords[atom_a*3:atom_a*3+3]
        if not norm_is_ok(vec):
            raise ValueError('Norm for Kmat not right.')
        vec_len = np.linalg.norm(vec)
        vec    /= vec_len

        # Get derivatives of the internal coordinates according to the literature
        # https://doi.org/10.1063/1.1515483
        for xyz1 in range(3):
            for xyz2 in range(3):
                tmpval = (vec[xyz1] * vec[xyz2] - dij(xyz1, xyz2))/vec_len
                tmpmat[atom_a*3+xyz1, atom_a*3+xyz2] = -tmpval * grad_int_id
                tmpmat[atom_b*3+xyz1, atom_b*3+xyz2] = -tmpval * grad_int_id
                tmpmat[atom_a*3+xyz1, atom_b*3+xyz2] = tmpval * grad_int_id
                tmpmat[atom_b*3+xyz1, atom_a*3+xyz2] = tmpval * grad_int_id
    return tmpmat

@njit
def kmat_bend(atom_ids, flatcoords, n_cart, grad_int_id):
    tmpmat  = np.zeros((n_cart, n_cart), dtype = np.float64)
    atom_a, atom_b, atom_c = atom_ids
    uvec = flatcoords[atom_a*3:atom_a*3+3] - flatcoords[atom_b*3:atom_b*3+3]
    vvec = flatcoords[atom_c*3:atom_c*3+3] - flatcoords[atom_b*3:atom_b*3+3]
    ulen = np.linalg.norm(uvec)
    vlen = np.linalg.norm(vvec)
    uvec /= ulen
    vvec /= vlen

    # Determine wvec according to paper
    if not are_parallel(uvec, vvec):
        wvec = np.cross(uvec, vvec)
    # vectors are parallel, need to be treated differently
    else:
        orth_vec = np.array([1.0, -1.0, 1.0])
        if not are_parallel(uvec, orth_vec) and not are_parallel(vvec, orth_vec):
            wvec = np.cross(uvec, orth_vec)
        else:
            wvec = np.cross(uvec, np.array([-1.0, 1.0, 1.0]))

    # normalized w-vector
    wlen = np.linalg.norm(wvec)
    wvec /= wlen

    # Compute first derivatives 
    tmpBmat = np.zeros((3,3))
    uwvec   = np.cross(uvec, wvec)
    wvvec   = np.cross(wvec, vvec)

    # For each coordinate of each atom
    for atom in range(3):
        for xyz in range(3):
            tmpBmat[atom, xyz] = zijk(atom, 0, 1) * uwvec[xyz]/ulen + zijk(atom, 2, 1) * wvvec[xyz]/vlen

    # If sine squared is too small, angle is too small
    cos_q = np.dot(uvec, vvec)
    sin_q = np.sqrt(1.0 - cos_q**2)
    if sin_q <= 1.0e-12: 
        return None

    # And the second derivatives
    for idx1, atom1 in enumerate(atom_ids):
        for xyz1 in range(3):
            for idx2, atom2 in enumerate(atom_ids):
                for xyz2 in range(3):
                    # Calculate according to the paper
                    tmpval = zijk(idx1, 0, 1)*zijk(idx2, 0, 1)*(uvec[xyz1]*vvec[xyz2] + uvec[xyz2]*vvec[xyz1] - 3*uvec[xyz1]*uvec[xyz2]*cos_q + dij(xyz1, xyz2)*cos_q) / (ulen**2 * sin_q)
                    tmpval += zijk(idx1, 2, 1)*zijk(idx2, 2, 1)*(uvec[xyz1]*vvec[xyz2] + uvec[xyz2]*vvec[xyz1] - 3*vvec[xyz1]*vvec[xyz2]*cos_q + dij(xyz1, xyz2)*cos_q) / (vlen**2 * sin_q)
                    tmpval += zijk(idx1, 0, 1)*zijk(idx2, 2, 1)*(uvec[xyz1]*uvec[xyz2] + vvec[xyz2]*vvec[xyz1] - uvec[xyz1]*vvec[xyz2]*cos_q - dij(xyz1, xyz2)) / (ulen*vlen*sin_q)
                    tmpval += zijk(idx1, 2, 1)*zijk(idx2, 0, 1)*(vvec[xyz1]*vvec[xyz2] + uvec[xyz2]*uvec[xyz1] - vvec[xyz1]*uvec[xyz2]*cos_q - dij(xyz1, xyz2)) / (ulen*vlen*sin_q)
                    tmpval -= tmpBmat[idx1, xyz1] * tmpBmat[idx2, xyz2] * cos_q / sin_q
                    tmpval *= grad_int_id
                    tmpmat[atom1*3+xyz1, atom2*3+xyz2] = tmpval
    return tmpmat

@njit
def kmat_twist(atom_ids, flatcoords, n_cart, grad_int_id):
    tmpmat  = np.zeros((n_cart, n_cart), dtype = np.float64)
    atom_a, atom_b, atom_c, atom_d = atom_ids
    uvec = flatcoords[atom_a*3:atom_a*3+3] - flatcoords[atom_b*3:atom_b*3+3]
    vvec = flatcoords[atom_d*3:atom_d*3+3] - flatcoords[atom_c*3:atom_c*3+3]
    wvec = flatcoords[atom_c*3:atom_c*3+3] - flatcoords[atom_b*3:atom_b*3+3]
    ulen = np.linalg.norm(uvec)
    vlen = np.linalg.norm(vvec)
    wlen = np.linalg.norm(wvec)
    uvec /= ulen
    vvec /= vlen
    wvec /= wlen

    # Defining sine and cosine
    cos_u = np.dot(uvec, wvec)
    cos_v = -np.dot(vvec, wvec)
    sin_u = np.sqrt(1.0 - cos_u**2)
    sin_v = np.sqrt(1.0 - cos_v**2)
    if sin_u <= 1.0e-12 or sin_v <= 1.0e-12: 
        return None

    uwcross = np.cross(uvec, wvec)
    vwcross = np.cross(vvec, wvec)
    sin4_u  = sin_u**4
    sin4_v  = sin_v**4
    cos3_u  = cos_u**3
    cos3_v  = cos_v**3

    # For all coordinates of all atom pairs
    # in the paper, the nomenclature is
    # idx1 = a, can be m, o, p, n or 0, 1, 2, 3
    # idx2 = b, can be m, o, p, n or 0, 1, 2, 3
    # xyz1 = i
    # xyz2 = j
    for idx1, atom1 in enumerate(atom_ids):
        for idx2, atom2 in enumerate(atom_ids[:idx1+1]):
            for xyz1 in range(3):
                for xyz2 in range(3):
                    tmpval = 0

                    # If statements go through cases, where the expression is not 0 because of the zijk function
                    # Avoiding computation if zijk is 0
                    if (idx1 == 0 and idx2 == 0) or (idx1 == 1 and idx2 == 0) or (idx1 == 1 and idx2 == 1):
                        tmpval += zijk(idx1,0,1)*zijk(idx2,0,1)*(uwcross[xyz1]*(wvec[xyz2]*cos_u - uvec[xyz2]) + uwcross[xyz2]*(wvec[xyz1]*cos_u - uvec[xyz1]))/(sin4_u*ulen**2)

                    if (idx1 ==3 and idx2 == 3) or (idx1 == 3 and idx2 == 2) or (idx1 == 2 and idx2 == 2):
                        tmpval += zijk(idx1,3,2)*zijk(idx2,3,2)*(vwcross[xyz1]*(wvec[xyz2]*cos_v + vvec[xyz2]) + vwcross[xyz2]*(wvec[xyz1]*cos_v + vvec[xyz1]))/(sin4_v*vlen**2)

                    if (idx1 == 1 and idx2 == 1) or (idx1 == 2 and idx2 == 1) or (idx1 == 2 and idx2 == 0) or (idx1 == 1 and idx2 == 0):
                        tmpval += (zijk(idx1,0,1)*zijk(idx2,1,2) + zijk(idx1,2,1)*zijk(idx2,1,0))*(
                                    uwcross[xyz1] * (wvec[xyz2] - 2*uvec[xyz2]*cos_u + wvec[xyz2]*cos_u**2) +
                                    uwcross[xyz2] * (wvec[xyz1] - 2*uvec[xyz1]*cos_u + wvec[xyz1]*cos_u**2))/(2*ulen*wlen*sin4_u)

                    if (idx1 == 3 and idx2 == 2) or (idx1 == 3 and idx2 == 1) or (idx1 == 2 and idx2 == 2) or (idx1 == 2 and idx2 == 1):
                        tmpval += (zijk(idx1,3,2)*zijk(idx2,2,1) + zijk(idx1,1,2)*zijk(idx2,2,3))*(
                                    vwcross[xyz1] * (wvec[xyz2] + 2*vvec[xyz2]*cos_v + wvec[xyz2]*cos_v**2) +
                                    vwcross[xyz2] * (wvec[xyz1] + 2*vvec[xyz1]*cos_v + wvec[xyz1]*cos_v**2))/(2*vlen*wlen*sin4_v)

                    if (idx1 == 1 and idx2 == 1) or (idx1 == 2 and idx2 == 2) or (idx1 == 2 and idx2 == 1):
                        tmpval += zijk(idx1,1,2)*zijk(idx2,2,1)*(uwcross[xyz1]*(uvec[xyz2] + uvec[xyz2]*cos_u**2 - 3*wvec[xyz2]*cos_u + wvec[xyz2]*cos3_u) +
                                                                    uwcross[xyz2]*(uvec[xyz1] + uvec[xyz1]*cos_u**2 - 3*wvec[xyz1]*cos_u + wvec[xyz1]*cos3_u))/(2*sin4_u*wlen**2)

                    if (idx1 == 2 and idx2 == 1) or (idx1 == 2 and idx2 == 2) or (idx1 == 1 and idx2 == 1): # if redundant?
                        tmpval += zijk(idx1,2,1)*zijk(idx2,1,2)*(vwcross[xyz1]*(-vvec[xyz2] - vvec[xyz2]*cos_v**2 - 3*wvec[xyz2]*cos_v + wvec[xyz2]*cos3_v) +
                                                                    vwcross[xyz2]*(-vvec[xyz1] - vvec[xyz1]*cos_v**2 - 3*wvec[xyz1]*cos_v + wvec[xyz1]*cos3_v))/(2*sin4_v*wlen**2)

                    # determine k (the third cartesian direction) for the last two terms
                    if (idx1 != idx2) and (xyz1 != xyz2):
                        if xyz1 != 0 and xyz2 != 0:
                            k = 0
                        elif xyz1 != 1 and xyz2 != 1:
                            k = 1
                        else:
                            k = 2

                        tmppow = (-0.5)**np.absolute(xyz2 - xyz1)
                        if idx1 == 1 and idx2 == 1: # not used at all, because the prefix (1-delta(idx1, idx2)) exists in the formula
                            tmpval += zijk(idx1,0,1)*zijk(idx2,1,2)*(xyz2 - xyz1)*tmppow*(wvec[k]*cos_u - uvec[k])/(ulen*wlen*sin_u**2)

                        if (idx1 == 3 and idx2 == 2) or (idx1 == 3 and idx2 == 1) or (idx1 == 2 and idx2 == 2) or (idx1 == 2 and idx2 == 1):
                            tmpval += zijk(idx1,3,2)*zijk(idx2,2,1)*(xyz2 - xyz1)*tmppow*(-wvec[k]*cos_v - vvec[k])/(vlen*wlen*sin_v**2)

                        if (idx1 == 2 and idx2 == 1) or (idx1 == 2 and idx2 == 0) or (idx1 == 1 and idx2 == 1) or (idx1 == 1 and idx2 == 0):
                            tmpval += zijk(idx1,2,1)*zijk(idx2,1,0)*(xyz2 - xyz1)*tmppow*(-wvec[k]*cos_u + uvec[k])/(ulen*wlen*sin_u**2)

                        if (idx1 == 2 and idx2 == 2): # not used either?
                            tmpval += zijk(idx1,1,2)*zijk(idx2,2,3)*(xyz2 - xyz1)*tmppow*(wvec[k]*cos_v + vvec[k])/(vlen*wlen*sin_v**2)

                    # multiply with internal gradient to get K-matrix elements
                    tmpval *= grad_int_id
                    tmpmat[atom1*3+xyz1, atom2*3+xyz2] = tmpval
                    tmpmat[atom2*3+xyz2, atom1*3+xyz1] = tmpmat[atom1*3+xyz1, atom2*3+xyz2]
    return tmpmat


def calcKmat(flatcoords = None, coord_list = None, grad_int = None):
    """ 
    calcKmat() calculates the second derivatives of the primitive internal coordinates with
    respect to the cartesian coordinates.

    See also: https://doi.org/10.1063/1.1515483
    """
    assert flatcoords is not None
    assert coord_list is not None
    assert grad_int is not None

    # initialize square matrix as K
    n_cart = flatcoords.shape[0]
    newKmat = np.zeros((n_cart, n_cart), dtype = np.float64)
    coord_list = [list(ids) for ids in coord_list]

    # for every internal coordinate, get the type
    for coord_id, atom_ids in enumerate(coord_list):

        # Get the normalized bond vector
        if len(atom_ids) == 2:
            tmpmat = kmat_stretch(atom_ids, flatcoords, n_cart, grad_int[coord_id])

        # handling a bend, get normalized vectors
        elif len(atom_ids) == 3:
            tmpmat = kmat_bend(atom_ids, flatcoords, n_cart, grad_int[coord_id])

        # Handling a twist, definig all vectors and normalizing
        elif len(atom_ids) == 4:
            tmpmat = kmat_twist(atom_ids, flatcoords, n_cart, grad_int[coord_id])
        else:
            raise ValueError("wrong tuple length for atom_ids: expected 2, 3, or 4, got {}".format(len(atom_ids)))
        if tmpmat is not None: 
            newKmat += tmpmat
    return newKmat