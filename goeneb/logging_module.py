import numpy as np
import logging
import os

from file_sys_io import qappend
from helper import normalize

# set up the logger for the program progress
def setup_logger(workdir='', name='output', level='debug'):
    """Sets up the logging file (output.log) and mutes 
    some loggers that produce unnecessary output."""
    filepath = os.path.join(workdir, name + '.log')
    logging.basicConfig(filename=filepath, 
                        filemode='w',
                        #format='%(filename)-25s %(message)s',
                        format='%(message)s',
                        level=translate_level(level)[0])
    logger = logging.getLogger()
    mpl_logger = logging.getLogger('matplotlib')
    mpl_logger.setLevel(logging.INFO)
    return logger

def translate_level(str_level):
    """Changes a human readable string into a logging level. 
    Returns the level, as well as a message"""
    translator = {'critical' : logging.CRITICAL,
                  'error' : logging.ERROR,
                  'warning' : logging.WARNING,
                  'info' :  logging.INFO,
                  'debug' : logging.DEBUG}
    str_level = str_level.lower()
    if str_level not in translator:
        message = f'{str_level} is not a valid logging level. Instead INFO is chosen.'
        level = logging.INFO
    else:
        message = f'Logging is set to {str_level}'
        level = translator[str_level]
    return level, message


def project_2d(nebpath_imgs):
    """Project the coordinates of the images into 2 dimensions for 
    visualization.
    1. first coordinate is distance along a line connecting 
    the the ends
    2. second coordinate is the norm off the remainder, 
    the distance between the image and its projected 
    position on the axis above
    """
    # first coordinate
    axvec = normalize(nebpath_imgs[-1] - nebpath_imgs[0])
    first_coords = [np.dot(pvec, axvec) for pvec in nebpath_imgs]

    # second coordinate
    remainders = [pvec - fcrd * axvec for pvec, fcrd in
                  zip(nebpath_imgs, first_coords)]
    second_coords = [np.linalg.norm(rem) for rem in remainders]

    # assemble np array
    return np.array([[crd1, crd2] for crd1, crd2 in
                    zip(first_coords, second_coords)])


class NEBLogger:
    def __init__(self, logfile_path):
        # clear the file in case it already exists
        file = open(logfile_path, 'w')
        file.close()
    
        self.logfile_path = logfile_path
        #self.prev_nebpath_obj = None
        
    def write_to_log(self, nebpath_obj, conv_sig_dict):
        """Writes to logfile (optlog.csv)"""
        path_pvecs = nebpath_obj.get_img_pvecs(include_ends=True)
        path_energies = nebpath_obj.get_energies(include_ends=True)
        path_gradnorms = [np.linalg.norm(grad) for grad in
                          nebpath_obj.get_nebgrads()]
        path_orth_gradnorms = [np.linalg.norm(grad)for grad in
                               nebpath_obj.get_orth_grads()] 
        n_total_imgs = len(path_pvecs)

        # first field in a line is the (currrent) number of images    
        newline = 'nimgs,' + str(n_total_imgs) + ','

        # add the convergence signal values
        # order: name1,val1,name2,val2,...
        for key, val in conv_sig_dict.items():
            newline += str(key) + ',' + str(val) + ','

        # add fields for energies and gradnorms
        newline += 'energies,'
        for energy in path_energies:
            newline += str(energy) + ','
        newline += 'gradnorms,'
        for norm in path_gradnorms:
            newline += str(norm) + ','
        newline += 'orth_gradnorms,'
        for norm in path_orth_gradnorms:
            newline += str(norm) + ','

        # add fields for the 2d path projection
        # order: img1_crd1,img1_crd2,img2_crd1,img2_crd2,...
        coords_2d = project_2d(path_pvecs.copy())
        newline += 'projcoords,'
        for crd_pair in coords_2d:
            newline += str(crd_pair[0]) + ',' + str(crd_pair[1]) + ','

        # add fields for the approx. rxn. coordinates
        rxn_crds = approx_rxn_coordinates(path_pvecs)
        newline += 'approx_rxn_crds,'
        for val in rxn_crds:
            newline += str(val) + ','

        # this is the end of the line, remove last comma
        newline = newline[:-1] + '\n'

        # write to log
        qappend(self.logfile_path, [newline])


def approx_rxn_coordinates(full_path_pvecs):
    """Gets the reaction coordinate (distances) from the pvecs."""
    total_dist = 0.0
    dists = [0.0]
    for i in range(1, len(full_path_pvecs)):
        delta_pos = full_path_pvecs[i] - full_path_pvecs[i-1]
        total_dist += np.linalg.norm(delta_pos)
        dists.append(total_dist)
    dists = np.array(dists)
    dists /= dists[-1]
    return dists


def draw_matrix(matrix, filename):
    """Draws a matrix and saves it."""
    import matplotlib.pyplot as plt
    plt.figure()
    plt.matshow(matrix)
    plt.savefig(filename)
