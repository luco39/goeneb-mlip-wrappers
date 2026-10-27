import numpy as np
import logging

import neb_exceptions as nex
from convsig_module import NEB_RMSF

logger = logging.getLogger(__name__)

class NEBPath:
    def __init__(self,
                 labels, 
                 starting_path, 
                 engrfunc=None, 
                 engrfunc_kwargs=None, 
                 settings=None):

        # with the engrad function, we can calculate the energies of the ends
        # for the nebpath object
        if engrfunc is not None:
            end_pvecs = np.array([starting_path[0], starting_path[-1]])
            end_energies, end_engrads = engrfunc(end_pvecs, **engrfunc_kwargs)
            if NEB_RMSF(end_engrads) > 0.00945:
                logger.warning("One of the end structures had a RMS(gradient) bigger than 0.0189. " + 
                               "Maybe you need to reoptimize the end structures.")
            if None in end_energies:
                # If the ends fail to converge, that usually means
                # there's something wrong with the interface or the
                # Engrad calculation method that the user set up
                raise nex.EngradError('Error: One or more of the end structures ' +
                                    'failed to converge. Check the end structures' +
                                    ' theory level, and other calculation aspects.')
        else:
            end_energies = [None, None]
            logger.warning('No energyfunction for calculation of path ends given!')

        starting_path = [np.array(pvec) for pvec in starting_path]
        self.labels = labels.copy()
        self.start_pvec = starting_path[0].copy()
        self.end_pvec = starting_path[-1].copy()
        self.start_energy = end_energies[0]
        self.end_energy = end_energies[-1]
        self.img_pvecs = starting_path[1:-1].copy()
        self.energies = np.zeros(len(self.img_pvecs))
        self.engrads = np.zeros_like(self.img_pvecs)
        self.springgrads = np.zeros_like(self.img_pvecs)
        self.tanvecs = np.zeros_like(self.img_pvecs)
        self.nebgrads = np.zeros_like(self.img_pvecs)
        self.orth_grads = np.zeros_like(self.img_pvecs)
        self.img_pair_ks = np.zeros(len(self.img_pvecs)+1) + settings.k_const
        self.ci_index = None
        self.misc_data = {}

    def n_failed_images(self):
        count = 0
        for energy in self.energies:
            if energy is None:
                count += 1
        return count

    def n_img_dim(self):
        return len(self.start_pvec)

    def n_images(self, include_ends=False):
        result = len(self.img_pvecs)
        if include_ends:
            result += 2
        return result

    def get_ci_index(self, include_ends=False):
        if self.ci_index is None:
            return None
        elif include_ends:
            return self.ci_index + 1
        else:
            return self.ci_index

    def set_ci_index(self, new_ci_index):
        self.ci_index = new_ci_index

    def update_ci_index(self):
        """update ci index to be the current highest energy image 
        replace Nones in case there are still failed images
        """
        img_energies = [value if value is not None else -np.inf
                        for value in self.energies]
        self.ci_index = np.argmax(img_energies)

    def get_img_pair_ks(self):
        return self.img_pair_ks.copy()

    def set_img_pair_ks(self, new_ks):
        self.img_pair_ks = new_ks.copy()

    def set_img_k_const(self, kval):
        self.img_pair_ks = np.zeros(len(self.img_pvecs)+1) + kval

    def get_img_pvecs(self, include_ends=False):
        if include_ends:
            return np.vstack([self.start_pvec,
                              self.img_pvecs,
                              self.end_pvec])
        else:
            return self.img_pvecs.copy()

    def set_img_pvecs(self, new_img_pvecs):
        self.img_pvecs = new_img_pvecs.copy()

    def get_energies(self, include_ends=False):
        if include_ends:
            return np.concatenate([[self.start_energy],
                                   self.energies,
                                   [self.end_energy]])
        else:
            return self.energies.copy()

    def set_energies(self, new_energies):
        self.energies = new_energies.copy()

    def get_engrads(self):
        return self.engrads.copy()

    def set_engrads(self, new_engrads):
        self.engrads = new_engrads.copy()

    def get_springgrads(self):
        return self.springgrads.copy()

    def set_springgrads(self, new_springgrads):
        self.springgrads = new_springgrads.copy()

    def get_tanvecs(self):
        return self.tanvecs.copy()

    def set_tanvecs(self, new_tanvecs):
        self.tanvecs = new_tanvecs.copy()

    def get_nebgrads(self):
        return self.nebgrads.copy()

    def set_nebgrads(self, new_nebgrads):
        self.nebgrads = new_nebgrads.copy()

    def get_orth_grads(self):
        return self.orth_grads.copy()

    def set_orth_grads(self, new_orth_grads):
        self.orth_grads = new_orth_grads.copy()

    def get_labels(self):
        return self.labels.copy()
    
    def get_path_length(self):
        path_pvecs = self.get_img_pvecs(include_ends=True)
        diffs = np.diff(path_pvecs, axis=0)
        segment_lengths = np.linalg.norm(diffs, axis=1)
        length = np.sum(segment_lengths)
        return length
