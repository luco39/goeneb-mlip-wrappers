import neb_exceptions as nex


def replace_failed_images(full_path_pvecs,
                          full_path_energies,
                          interp_func,
                          interp_func_kwargs={}):
    """A function that replaces all images where the energy calculation failed.\\
    The function assumes that failed images are marked with a 'None' for energy\\
    It returns a version of full_path_pvecs where all failed images have been 
    replaced with interpolations based on their nearest non-failed neighbor images.
    """
    # indices which mark the images that neighbor a sequence of failed image (of length 1 or longer)
    left_neighbor = None
    right_neighbor = None

    for i in range(1, len(full_path_pvecs) - 1):
        if full_path_energies[i] is None and full_path_energies[i-1] is not None:
            # we have entered a string of failed images.
            if left_neighbor is not None:
                # this should be impossible
                raise nex.NEBError('Error in failed_img_recalculator: something that ' +
                                   'should be impossible just happened. Probably an ' +
                                   'error in the code.')
            left_neighbor = i - 1

        if full_path_energies[i] is None and full_path_energies[i+1] is not None:
            # we are at the last element in a string of failed images.
            if right_neighbor is not None:
                # this should be impossible
                raise nex.NEBError('Error in failed_img_recalculator: something that ' +
                                   'should be impossible just happened. Probably an ' +
                                   'error in the code.')
            right_neighbor = i + 1

        if left_neighbor is not None and right_neighbor is not None:
            # we have identified a sequence of failed images, and we can replace them.
            # isolate the pvecs of the nearest non-failed neighbors
            left_pvec = full_path_pvecs[left_neighbor]
            right_pvec = full_path_pvecs[right_neighbor]

            n_interp_imgs = (right_neighbor - left_neighbor) - 1

            # It is assumed that interp_func will return a list of images which also contain the two
            # neighbors again at the ends. These should not be replaced.
            new_pvecs = interp_func(left_pvec,
                                    right_pvec,
                                    n_interp_imgs,
                                    **interp_func_kwargs)

            full_path_pvecs[left_neighbor+1 : right_neighbor] = new_pvecs[1:-1]

            left_neighbor = None
            right_neighbor = None

    return full_path_pvecs