import numpy as np
import logging

logger = logging.getLogger(__name__)

# this is dummy interface, producing nonsense energies
# and engrads, meant for testing if the code works
# ---------------------------------------------------------

class IterationState():
    def __init__(self, itercount=0):
        self.itercount = itercount
    def update(self):
        self.itercount += 1

def dummy_path_engrads(image_pvecs, iteration:IterationState, number_of_images):
    # Define some values for energy function and failed images
    number_failed_img = np.min([int(number_of_images/3), 3])
    sim_failed_images = [i for i in range(2, 2 + number_failed_img)]
    max_dummy_energy = 1
    maximum = 0.35 * (number_of_images - 1)

    # Define the energy function dependent on whether ends are given or not
    if number_of_images == len(image_pvecs):
        def dummy_energy(image_nr, max_dummy_energy, maximum):
            return max_dummy_energy - (image_nr - maximum)**2
    elif len(image_pvecs) == 2:
        def dummy_energy(image_nr, max_dummy_energy, maximum):
            return max_dummy_energy - (image_nr*number_of_images - maximum)**2
    elif number_of_images == len(image_pvecs) + 2:
        def dummy_energy(image_nr, max_dummy_energy, maximum):
            return max_dummy_energy - (image_nr - maximum + 1)**2
    else:
        logger.warning('Dummy energy can only be calculated for the whole path.')
        def dummy_energy(image_nr, max_dummy_energy, maximum):
            return 1

    dummy_energies = []
    dummy_engrads = []
    for image_nr in range(len(image_pvecs)):
        energy = dummy_energy(image_nr, max_dummy_energy, maximum)

        # to simulate a converging path, we create exponentially
        # diminishing artificial engrads
        dim_fac = 0.25 * np.power(0.25, iteration.itercount)
        engrad = dim_fac * image_pvecs[image_nr]

        # to simulate failing images, we return none for energies
        if iteration.itercount in [3,4,5] and image_nr in sim_failed_images:
            energy = None
            engrad = np.zeros_like(engrad)

        dummy_energies.append(energy)
        dummy_engrads.append(engrad)

    iteration.update()
    return np.array(dummy_energies), np.array(dummy_engrads)