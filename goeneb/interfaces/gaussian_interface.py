import os
import subprocess
import numpy as np
import file_sys_io as io
import logging

from helper import bettersplit
from neb_exceptions import ParsingError

logger = logging.getLogger(__name__)


bohr = 0.529177211  # angs per bohr


# The following is the implementation of the GAUSSIAN interface
# ---------------------------------------------------------


def gaussian_path_engrads(image_pvecs,
                          labels,
                          series_name,
                          workingdir,
                          gaussian,
                          method_keywords,
                          charge,
                          spin,
                          nprocs,
                          mem,
                          title):
    image_data = [[labels, pvec] for pvec in image_pvecs]
    imgage_names = [series_name + str(i+1) for i in range(len(image_pvecs))]
    result = [calculate_gaussian_engrad(image,
                                      name,
                                      workingdir,
                                      gaussian,
                                      method_keywords,
                                      charge,
                                      spin,
                                      nprocs,
                                      mem,
                                      title)
            for image, name in zip(image_data, imgage_names)]
    energies = []
    engrads = []
    # reorganize results the way the new code needs them
    for item in result:
        if item is None:
            # the calculation failed to converge
            energies.append(None)
            engrads.append(np.zeros_like(image_pvecs[0]))
        else:
            [energy, engrad] = item
            energies.append(energy)
            engrads.append(engrad)
    return np.array(energies), np.array(engrads)

def calculate_gaussian_engrad(struct_data,
                              inpfile_name,
                              workingdir,
                              gaussian,
                              method_keywords,
                              charge='0',
                              spin='1',
                              nprocs=1,
                              mem=1000,
                              title='Title Card Required'):
    """
    Calculate the gaussian energy and gradient. This function does:
    1. Delete previous output files
    2. Create input file
    3. Run Gaussian
    4. Read output file\n
    assumes that 'struct_data' is a list containing a list of n string  
    atom labels, and a numpy array of 3*n length containing the atom 
    coordinates
    """
    io.safe_create_dir(workingdir)
    os.environ['GAUSS_SCRDIR'] = str(workingdir)
    input_filename = workingdir / (inpfile_name + '.gjf')
    output_filename = workingdir / (inpfile_name + '.log')

    # make sure no output file is left over. All output files for an
    # image will have the same name, and if one stays around, it might
    # be mistaken for the result of a future calculation that actually failed.
    io.safe_delete_file(output_filename, does_not_exist_ok=True)
    logger.info('Now calculating: ' + inpfile_name)
    write_gaussian_inpfile(input_filename, struct_data, method_keywords,
                           charge, spin, nprocs, mem, title)

    try:
        subprocess.run([str(gaussian) + ' ' + str(input_filename)], shell=True)
    except Exception as e:
        logger.error(f"Unexpected error running Gaussian for {inpfile_name}: {e}")
        return None

    # read results
    try:
        Energy, Gradient = read_gaussian_forcefile(output_filename)
    except ParsingError:
        return None  # engrad calculation didnt converge
    else:
        return [Energy, Gradient]


# Some helper functions

def write_gaussian_inpfile(filename, struct_data, method_keywords,
                           charge='0', spin='1', nprocs=1, mem=1000,
                           title='Title Card Required'):
    """
    Writes a gaussian input file from filename, structure vector, and method keywords.
    """
    coords = struct_data[1].reshape(len(struct_data[0]), 3)
    labels = struct_data[0]
    file = open(filename, 'w')
    try:
        file.write(r'%nprocshared=' + str(nprocs) + '\n')
        file.write(r'%mem=' + str(mem) + 'mb\n')
        file.write('# force ' + method_keywords + '\n\n')
        file.write(title + '\n\n')
        file.write(str(charge) + ' ' + str(spin) + '\n')
        for label, coord in zip(labels, coords):
            line = ' ' + str(label)
            for value in coord:
                line += '\t' + repr(value)
            line += '\n'
            file.write(line)
        file.write('\n')
    finally:
        file.close()


def read_gaussian_forcefile(filename):
    """
    Read the gaussian force file and return the energy and the gradient.
    """
    file = open(filename, 'r')
    lines = file.readlines()
    file.close()

    # read in SPE.
    Energy = None
    for line in reversed(lines):
        if 'E(' in line:
            tokens = bettersplit(line, ' \t\n')
            for token in tokens:
                try:
                    val = float(token)
                except:
                    pass
                else:
                    Energy = val
                    break
    if Energy is None:
        raise ParsingError(f'No SPE found in gaussian result file: {str(filename)}')

    # read in force vector
    try:
        start_ind = lines.index(' Center     Atomic                   Forces (Hartrees/Bohr)\n') + 3
        stop_ind = lines.index(' -------------------------------------------------------------------\n', start_ind)
    except ValueError as e:
        raise ParsingError(f'Invalid gaussian result file: {str(filename)}') from e

    coords = []
    for i in range(start_ind, stop_ind):
        tokens = bettersplit(lines[i], ' \t\n')
        if len(tokens) != 5:
            raise ParsingError(f'Invalid gaussian result file: {str(filename)}')
        try:
            coords += [float(tokens[2]), float(tokens[3]), float(tokens[4])]
        except ValueError as e:
            raise ParsingError(f'Error while trying to read gaussian result file: {str(filename)}') from e
    force_vec = np.array(coords)
    return Energy, force_vec / -bohr  # Eh/bohr forces to Eh/ang gradients

