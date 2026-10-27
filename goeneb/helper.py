import numpy as np

# Coordinates and vectors

def pvec_to_crdmat(pvec):
    pvec = np.array(pvec)
    natoms = int(len(pvec) / 3)
    return pvec.reshape((natoms, 3))

def crdmat_to_pvec(crdmat):
    crdmat = np.array(crdmat)
    return crdmat.flatten()

def calculate_distmat(pvec, rvec=False):
    """Get the distance matrix of the atoms in the molecule.
    If rvec=True, it also returns the (M,N,3) matrix of the 
    distance vectors."""
    crdmat = pvec_to_crdmat(pvec) # has shape (N,3)
    # Broadcast to new shape (N,N,3)
    rvec_mat = crdmat[np.newaxis, :, :] - crdmat[:, np.newaxis, :]
    distmat = np.linalg.norm(rvec_mat, axis=-1)
    if not rvec:
        return distmat
    else:
        return distmat, rvec_mat

def normalize(vector):
    norm = np.linalg.norm(vector)
    if norm != 0.0:
        vector /= norm
    return vector

def project(vector, axis):
    """assumes vector, axis are 1D numpy arrays, 
    assumes axis is normalized to 1"""
    return axis * np.dot(vector, axis)

def reject(vector, axis):
    """assumes vector, axis are 1D numpy arrays, 
    assumes axis is normalized to 1"""
    return vector - project(vector, axis)

def interpolate_linear(startvec, stopvec, interp_facs):
    """
    Interpolates between startvec and stopvec 
    according to the interpolation factors given.
    
    If an integer is given, it is converted to a linspace 
    with a total of int+2 factors (including 0 and 1)
    """
    if isinstance(interp_facs, int):
        interp_facs = np.linspace(0.0, 1.0, num=interp_facs+2)
    interp_vecs = [startvec + (stopvec - startvec) * fac
                   for fac in interp_facs]
    return np.array(interp_vecs)

# Lists and strings

def bettersplit(text, delim):
    """Splits a string at the specified delimitors."""
    tokens = []
    token = ''
    for char in text:
        if char not in delim:
            token += char
        elif token != '':
            tokens.append(token)
            token = ''
    if token != '':
        tokens.append(token)
    return tokens

def parse_index_list(index_list_conf_entry):
    """Converts a string containing numbers into a list of integers."""
    if index_list_conf_entry is None:
        return []
    else:
        # convert the list string representation
        # into a list of ints
        return [int(token) for token in
                reverse_bettersplit(index_list_conf_entry)]

def reverse_bettersplit(text, not_delims='0123456789'):
    """Extracts all digit sequences from a string, ignoring any non-digit characters used as separators."""
    tokens = []
    token = ''
    for char in text:
        if char in not_delims:
            token += char
        elif token != '':
            tokens.append(token)
            token = ''
    if token != '':
        tokens.append(token)
    return tokens