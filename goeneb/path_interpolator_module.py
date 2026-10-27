import numpy as np
import logging

import struct_aligner_module as sam
import IDPP_module as idpp
import geodesic_module as geo
from neb_exceptions import NEBError
from helper import interpolate_linear, normalize

logger = logging.getLogger(__name__)

# This module contains functions used to generate
# interpolated paths between two end structures.
# Used for starting path generation and reinterpolating
# failed images.


def do_interpolation(start_pvec,
                     end_pvec,
                     labels,
                     interpolations,
                     mode='internal'):
    """
    This function does the basic interpolation,
    with option for cartesian, internal or geodesic.
    Expects:
    - start_pvec and end_pvec: Structures
    - labels: A list with types of atoms
    - mode: internal, cartesian or geodesic
    - interpolations: Either the number of new images, or the specific positions of the interpolation
    """
    # Get the interpolation space
    if isinstance(interpolations, int):
        if mode != 'geodesic':
            interpolations = np.linspace(0.0, 1.0, num=interpolations+2)
    else:
        if mode == 'geodesic':
            mode = 'internal'
            logger.warning('Setting intepolation mode to internal, since geodesic cant handle specific input.')

    if mode == 'cartesian':
        interp_path = interpolate_linear(start_pvec, end_pvec, interpolations)

    elif mode == 'internal':
        # only import now, to keep functionality without ChemChoord
        import ChemCoord_interface as cci

        # construction table based on each end once, then compare
        interp1, ctable1 = cci.zmat_interpolate(start_pvec,
                                                end_pvec,
                                                labels,
                                                interpolations,
                                                use_2nd_constr_table=False)

        interp2, ctable2 = cci.zmat_interpolate(start_pvec,
                                                end_pvec,
                                                labels,
                                                interpolations,
                                                use_2nd_constr_table=True)

        # check if the z-matrix interpolation worked
        if interp1 is not None and interp2 is not None:
            # compare total atom movement to find better path
            tam1 = cci.total_atom_movement(interp1)
            tam2 = cci.total_atom_movement(interp2)
            if tam1 <= tam2:
                interp_path = interp1
            else:
                interp_path = interp2

        elif interp1 is not None and interp2 is None:
            interp_path = interp1
            logger.warning("The z-matrix construction only worked with the start structure. This construction table is used for the interpolation.")

        elif interp2 is not None and interp1 is None:
            interp_path = interp2
            logger.warning("The z-matrix construction only worked with the end structure. This construction table is used for the interpolation.")

        else:
            logger.error("Both structures could not be transformed to z-matrix coordinates. Try again with cartesian or geodesic.")
            raise NEBError('')

    elif mode == 'geodesic':
        level = logger.level
        if logger.level < 30:
            logger.setLevel('WARNING')      # geodesic has some output we don't need
        interp_path = geo.interpolate_geodesic(start_pvec, end_pvec, labels, interpolations)
        logger.setLevel(level)              # set the level back after geodesic use
    else:
        raise ValueError('Error in path_interpolator_module: "' + str(mode) +
                       '" is not a valid interpolation mode.')
    return interp_path


def interpolate_path(start_pvec,
                     end_pvec,
                     n_new_interps,
                     labels,
                     interp_mode,
                     rot_align_mode,
                     settings):
    """
    This function does the interpolation of the starting path (including alignment and IDPP),
    with option for cartesian, internal or geodesic.
    Expects:
    - start_pvec and end_pvec: Structures
    - n_new_interps: How many new interpolation images should be produced
    - labels: A list with types of atoms
    - interp_mode: internal, cartesian or geodesic
    - rot_align_mode: None, full or pairwise
    - settings object (needs the function get_idpp_settings())
    """
    if settings.SIDPP:
        path_pvecs = idpp.do_SIDPP_opt_pass(start_pvec,
                                            end_pvec,
                                            settings)
    else:
        # initial interpolations
        path_pvecs = do_interpolation(start_pvec,
                                      end_pvec,
                                      labels,
                                      n_new_interps,
                                      interp_mode)

        # translational and rotational alignement
        path_pvecs = sam.align_path(path_pvecs, rot_align_mode, settings.translate_mode)

        # IDPP
        if settings.IDPP:
            if interp_mode == 'geodesic':
                logger.warning('IDPP pass should not be selected, when using geodesic interpolation, '
                              + 'IDPP pass will be skipped in favor of geodesic interpolation.')
            else:
                path_pvecs = idpp.do_IDPP_opt_pass(path_pvecs,
                                                   settings)

    # translational and rotational alignement (again after IDPP)
    path_pvecs = sam.align_path(path_pvecs, rot_align_mode, settings.translate_mode)
    return path_pvecs

def interpolate_TS_cubic(coords, energies, grads):
    logger.info("Doing cubic fit to interpolate TS. Suggested by Henkelman and Jonsson.")
    def get_parameters(V_i, V_next, F_i, F_next, R):
        a = 2*(V_i - V_next)/R**3 - (F_i + F_next)/R**2
        b = 3*(V_next - V_i)/R**2 + (2*F_i + F_next)/R
        c = -F_i
        d = V_i
        return a,b,c,d

    def max_cubic_spline(a, b, c, d, R):
        x = np.linspace(0, R, 100)
        y = a*x**3 + b*x**2 + c*x + d
        return x[np.argmax(y)], np.max(y)
    
    energies = np.array(energies)
    hei_index = np.nanargmax(energies)

    if hei_index == len(energies) or hei_index == 0:
        logger.error("HEI is one of the end images. Something went wrong.")

    left_coords = coords[hei_index-1]
    hei_coords = coords[hei_index]
    right_coords = coords[hei_index+1]
    left_e = energies[hei_index-1]
    hei_e = energies[hei_index]
    right_e = energies[hei_index+1]
    left_grads = -grads[hei_index-1]    # forces instead of gradients
    hei_grads = -grads[hei_index]
    right_grads = -grads[hei_index+1]

    # interpolate energies
    R_left = np.linalg.norm(hei_coords - left_coords)
    a,b,c,d = get_parameters(left_e, hei_e, np.linalg.norm(left_grads), np.linalg.norm(hei_grads), R_left)
    s_max_left, max_energy_left = max_cubic_spline(a,b,c,d, R_left)

    R_right = np.linalg.norm(right_coords - hei_coords)
    a,b,c,d = get_parameters(hei_e, right_e, np.linalg.norm(hei_grads), np.linalg.norm(right_grads), R_right)
    s_max_right, max_energy_right = max_cubic_spline(a,b,c,d, R_right)

    if max_energy_left > max_energy_right:
        s_max = s_max_left
        grads1 = left_grads
        grads2 = hei_grads
        coords1 = left_coords
        coords2 = hei_coords
        R = R_left
    else:
        s_max = s_max_right
        grads1 = hei_grads
        grads2 = right_grads
        coords1 = hei_coords
        coords2 = right_coords
        R = R_right

    # modify gradients in end points
    if hei_index == 1 and max_energy_left > max_energy_right:
        grads1 = coords1 - coords2
    elif hei_index == len(energies) -1 and max_energy_left <= max_energy_right:
        grads2 = coords2 - coords1

    # interpolate structures
    a,b,c,d = get_parameters(coords1, coords2, -normalize(grads1), -normalize(grads2), R)
    struct = a*s_max**3 + b*s_max**2 + c*s_max + d
    return struct

