import numpy as np
import file_sys_io as io
import subprocess
import logging

from helper import bettersplit
from neb_exceptions import ParsingError

logger = logging.getLogger(__name__)

bohr = 0.529177211  # angs per bohr

# The following is the implementation of the ORCA interface
# ---------------------------------------------------------

def orca_path_engrads(image_pvecs,
                      labels,
                      series_name,
                      workingdir,
                      orca,
                      method_keywords,
                      charge,
                      spin,
                      npal=None,
                      method_line2=None):
    """
    This function calculates the ORCA energy and gradient for all images.
    - image_pvecs: lsit of the flattened image vectors
    - labels: list of the atom types
    - series_name: base name for the input and output files
    - workingdir: The directory where the calculation takes place
    - orca: The orca command in CMD
    - method_keywords: Orca method input
    - charge, spin: charge and spin of the system
    - npal: Number of parallel processes
    - method_line2: Additional input for orca
    """
    image_data = [[labels, pvec] for pvec in image_pvecs]
    imgage_names = [series_name + str(i+1) for i in range(len(image_pvecs))]
    results = [calculate_orca_engrad(image,
                                  name,
                                  workingdir,
                                  orca,
                                  method_keywords,
                                  charge,
                                  spin,
                                  npal=npal,
                                  method_line2=method_line2)
               for image, name in zip(image_data, imgage_names)]

    energies = []
    engrads = []
    # reorganize results
    for item in results:
        if item is None:
            # the calculation failed to converge
            energies.append(None)
            engrads.append(np.zeros_like(image_pvecs[0]))
        else:
            [energy, engrad] = item
            energies.append(energy)
            engrads.append(engrad)
    logger.debug('Energies: %s', energies)
    return np.array(energies), np.array(engrads)

def calculate_orca_engrad(struct_data,
                          inpfile_name,
                          workingdir,
                          orca,
                          method_keywords,
                          charge='0',
                          spin='1',
                          npal=None,
                          method_line2=None):
    """
    This function does the complete energy and gradient calculation for one image in ORCA.
    1. Remove any left over output files
    2. Write the ORCA input file
    3. Write the output into the output file
    4. Read the output file and return energy and gradient\n
    Assumes that 'struct_data' is a list containing a list of n string atom labels,
    and a numpy array of 3*n length containing the atom coordinates
    """    
    output_file = None

    input_filename = workingdir / (inpfile_name + '.inp')
    engrad_filename = workingdir / (inpfile_name + '.engrad')
    output_filename = workingdir / (inpfile_name + '.out')

    # make sure no .engrad file is left over. All engrad files for an
    # image will have the same name, and if one stays around, it might
    # be mistaken for the result of a future calculation that actually failed.
    io.safe_delete_file(engrad_filename, does_not_exist_ok=True)
    io.safe_delete_file(output_filename, does_not_exist_ok=True)
    logger.info('Now calculating: ' + inpfile_name)
    write_orca_inpfile(input_filename, struct_data,
                        method_keywords, charge, spin,
                        npal=npal, method_line2=method_line2)

    try:
        with open(output_filename, 'w') as output_file:
            subprocess.run([orca, input_filename], stdout=output_file)
    except Exception as e:
        logger.error(f"Unexpected error running ORCA for {inpfile_name}: {e}")
        return None

    # Read the results
    try:
        V, grad_V = read_orca_engrad(engrad_filename)
    except ParsingError:
        return None  # engrad calculation didnt converge

    return [V, grad_V / bohr]  # convert engrad to Eh/Ang


# Some helper functions


def sanitize_mkwords(method_keywords):
    """
    Modify the method keywords to be a complete Orca input
    line including the keywords: NoAutoStart, EnGrad and Angs as well as the !.
    """
    mk_tokens = bettersplit(method_keywords.lower(), ' \t')
    if 'noautostart' not in mk_tokens:
        method_keywords += ' NoAutoStart'
    if 'engrad' not in mk_tokens:
        method_keywords += ' EnGrad'
    if 'angs' not in mk_tokens:
        method_keywords += ' Angs'
    if method_keywords[0] != '!':
        method_keywords = '!' + method_keywords
    return method_keywords


def write_orca_inpfile(filename, struct_data, method_keywords,
                       charge='0', spin='1', npal=None, method_line2=None):
    """
    Make an inputfile (filename) for an Orca calculation with the specified keywords. 
    The xyz is specified in the input file.
    """
    file = open(filename, 'w')
    coords = struct_data[1].reshape(len(struct_data[0]), 3)

    file.write(sanitize_mkwords(method_keywords) + '\n\n')
    file.write('%maxcore 3000 \n\n')    

    if npal is not None:
        file.write('%PAL NPROCS ' + str(npal) + ' END\n\n')  
    if method_line2 is not None:
        file.write(method_line2 + '\n\n')
    file.write('* xyz ' + str(charge) + ' ' + str(spin) + '\n')

    for label, coord in zip(struct_data[0], coords):
        file.write('\t' + label)
        for number in coord:
            file.write('\t' + str(repr(number)))
        file.write('\n')
    file.write('*')
    file.close()


def read_orca_engrad(filename):
    with open(filename) as file:
        lines = file.readlines()

    try:
        # Find energy
        index = lines.index("# The current total energy in Eh\n") + 2
        Energy = float(lines[index])

        # Find gradient energy
        index_start = lines.index("# The current gradient in Eh/bohr\n") + 2
        index_end = lines.index("# The atomic numbers and current coordinates in Bohr\n") - 1
        grad_vals = [float(line) for line in lines[index_start : index_end]]
        grad_vec = np.array(grad_vals)

    except ValueError:
        raise ParsingError(f'Invalid Orca engrad file {filename}.')

    return Energy, grad_vec
