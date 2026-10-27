import numpy as np
import logging

logger = logging.getLogger(__name__)

try:
    from scipy.spatial import KDTree
    from scipy.optimize import least_squares
except ImportError:
    KDTree = None
    least_squares = None
    print("Scipy could not be imported. This is important for the geodesic interpolation.")

def interpolate_geodesic(start_pvec, 
                         end_pvec, 
                         labels, n_new_interps, 
                         tol=0.002, 
                         scaling=1.7,
                         dist_cutoff=3,
                         friction=0.01,
                         sweep=False,
                         maxiter=15,
                         microiter=20):

    # Perform first interpolation between the two given images
    new_shape = (int(start_pvec.shape[0]/3), 3)
    start_pvec = start_pvec.reshape(new_shape)
    end_pvec = end_pvec.reshape(new_shape)
    raw = redistribute(labels, [start_pvec, end_pvec], n_new_interps+2, tol=tol*5)

    # Perform smoothing by minimizing distance in Cartesian coordinates with redundant internal metric
    # to find the appropriate geodesic curve on the hyperspace.
    smoother = Geodesic(labels,
                        raw, 
                        scaling, 
                        threshold=dist_cutoff, 
                        friction=friction)

    if (sweep==True) or (len(labels) > 35):
        smoother.sweep(tol=tol, 
                       max_iter=maxiter, 
                       micro_iter=microiter)

    else: 
        smoother.smooth(tol=tol, max_iter=maxiter)

    path = smoother.path.reshape(n_new_interps+2,3*new_shape[0])
    logger.debug('Geodesic path shape is: %s', str(path.shape))
    return path



# ----------------------------------------------------------------------------
# Here are the geodesic_interpolate functions written by Xiaolei Zhu
# to be used in the interpolation 
# https://github.com/virtualzx-nad/geodesic-interpolate

# ----------------------------------------------------------------------------
# Here are the interpolation functions

def mid_point(atoms, geom1, geom2, tol=1e-2, nudge=0.01, threshold=4):
    """Find the Cartesian geometry that has internal coordinate values closest to the average of
    two geometries.

    Simply perform a least-squares minimization on the difference between the current internal
    and the average of the two end points.  This is done twice, using either end point as the
    starting guess.  DON'T USE THE CARTESIAN AVERAGE AS GUESS, THINGS WILL BLOW UP.

    This is used to generate an initial guess path for the later smoothing routine.
    Genenrally, the added point may not be continuous with the both end points, but
    provides a good enough starting guess.

    Random nudges are added to the initial geometry, so running multiple times may not yield
    the same converged geometry. For larger systems, one will never get the same geometry
    twice.  So one may want to perform multiple runs and check which yields the best result.

    Args:
        geom1, geom2:   Cartesian geometry of the end points
        tol:    Convergence tolarnce for the least-squares minimization process
        nudge:  Random nudges added to the initial geometry, which helps to discover different
                solutions.  Also helps in cases where optimal paths break the symmetry.
        threshold:  Threshold for including an atom-pair in the coordinate system

    Returns:
        Optimized mid-point which bisects the two endpoints in internal coordinates
    """
    assert least_squares is not None, "scipy is needed for geodesic interpolation"
    # Process the initial geometries, construct coordinate system and obtain average internals
    geom1, geom2 = np.array(geom1), np.array(geom2)
    add_pair = set()
    geom_list = [geom1, geom2]
    # This loop is for ensuring a sufficient large coordinate system.  The interpolated point may
    # have atom pairs in contact that are far away at both end-points, which may cause collision.
    # One can include all atom pairs, but this may blow up for large molecules.  Here the compromise
    # is to use a screened list of atom pairs first, then add more if additional atoms come into
    # contant, then rerun the minimization until the coordinate system is consistant with the
    # interpolated geometry
    while True:
        rijlist, re = get_bond_list(geom_list, threshold=threshold + 1, enforce=add_pair)
        scaler = morse_scaler(alpha=0.7, re=re)
        w1, _ = compute_wij(geom1, rijlist, scaler)
        w2, _ = compute_wij(geom2, rijlist, scaler)
        w = (w1 + w2) / 2
        d_min, x_min = np.inf, None
        friction = 0.1 / np.sqrt(geom1.shape[0])
        def target_func(X):
            """Squared difference with reference w0"""
            wx, dwdR = compute_wij(X, rijlist, scaler)
            delta_w = wx - w
            val, grad = 0.5 * np.dot(delta_w, delta_w), np.einsum('i,ij->j', delta_w, dwdR)
            logger.info("val=%10.3f  ", val)
            return val, grad

        # The inner loop performs minimization using either end-point as the starting guess.
        for coef in [0.02, 0.98]:
            x0 = (geom1 * coef + (1 - coef) * geom2).ravel()
            x0 += nudge * np.random.random_sample(x0.shape)
            logger.debug('Starting least-squares minimization of bisection point at %7.2f.', coef)
            result = least_squares(lambda x: np.concatenate([compute_wij(x, rijlist, scaler)[0] - w, (x-x0)*friction]), x0,
                                   lambda x: np.vstack([compute_wij(x, rijlist, scaler)[1], np.identity(x.size) * friction]), ftol=tol, gtol=tol)
            x_mid = result['x'].reshape(-1, 3)
            # Take the interpolated geometry, construct new pair list and check for new contacts
            new_list = geom_list + [x_mid]
            new_rij, _ = get_bond_list(new_list, threshold=threshold, min_neighbors=0)
            extras = set(new_rij) - set(rijlist)
            if extras: 
                logger.info('  Screened pairs came into contact. Adding reference point.')
                # Update pair list then go back to the minimization loop if new contacts are found
                geom_list = new_list
                add_pair |= extras
                break
            # Perform local geodesic optimization for the new image.
            smoother = Geodesic(atoms, [geom1, x_mid, geom2], 0.7, threshold=threshold, log_level=logging.DEBUG, friction=1)
            smoother.compute_disps()
            width = max([np.sqrt(np.mean((g - smoother.path[1]) ** 2)) for g in [geom1, geom2]])
            dist, x_mid = width + smoother.length, smoother.path[1]
            logger.debug('  Trial path length: %8.3f after %d iterations', dist, result['nfev'])
            if dist < d_min:
                d_min, x_min = dist, x_mid
        else:   # Both starting guesses finished without new atom pairs.  Minimization successful
            break
    return x_min


def redistribute(atoms, geoms, nimages, tol=1e-2):
    """Add or remove images so that the path length matches the desired number.

    If the number is too few, new points are added by bisecting the largest RMSD. If too numerous,
    one image is removed at a time so that the new merged segment has the shortest RMSD.

    Args:
        geoms:      Geometry of the original path.
        nimages:    The desired number of images
        tol:        Convergence tolerance for bisection.

    Returns:
        An aligned and redistributed path with has the correct number of images.
    """
    _, geoms = align_path(geoms)
    geoms = list(geoms)
    # If there are too few images, add bisection points
    while len(geoms) < nimages:
        dists = [np.sqrt(np.mean((g1 - g2) ** 2)) for g1, g2 in zip(geoms[1:], geoms)]
        max_i = np.argmax(dists)
        logger.info("Inserting image between %d and %d with Cartesian RMSD %10.3f.  New length:%d",
                    max_i, max_i + 1, dists[max_i], len(geoms) + 1)
        insertion = mid_point(atoms, geoms[max_i], geoms[max_i + 1], tol)
        _, insertion = align_geom(geoms[max_i], insertion)
        geoms.insert(max_i + 1, insertion)
        geoms = list(align_path(geoms)[1])
    # If there are too many images, remove points
    while len(geoms) > nimages:
        dists = [np.sqrt(np.mean((g1 - g2) ** 2)) for g1, g2 in zip(geoms[2:], geoms)]
        min_i = np.argmin(dists)
        logger.info("Removing image %d.  Cartesian RMSD of merged section %10.3f",
                    min_i + 1, dists[min_i])
        del geoms[min_i + 1]
        geoms = list(align_path(geoms)[1])
    return geoms


# ----------------------------------------------------------------------------
# Here are some helper functions for coordinate handling


def align_path(path):
    """Rotate and translate images to minimize RMSD movements along the path.
    Also moves the geometric center of all images to the origin.
    """
    path = np.array(path)
    path[0] -= np.mean(path[0], axis=0)
    max_rmsd = 0
    for g, nextg in zip(path, path[1:]):
        rmsd, nextg[:] = align_geom(g, nextg)
        max_rmsd = max(max_rmsd, rmsd)
    return max_rmsd, path


def align_geom(refgeom, geom):
    """Find translation/rotation that moves a given geometry to maximally overlap
    with a reference geometry. Implemented with Kabsch algorithm.

    Args:
        refgeom:    The reference geometry to be rotated to
        geom:       The geometry to be rotated and shifted

    Returns:
        RMSD:       Root-mean-squared difference between the rotated geometry
                    and the reference
        new_geom:   The rotated geometry that maximumally overal with the reference
    """
    center = np.mean(refgeom, axis=0)   # Find the geometric center
    ref2 = refgeom - center
    geom2 = geom - np.mean(geom, axis=0)
    cov = np.dot(geom2.T, ref2)
    v, sv, w = np.linalg.svd(cov)
    if np.linalg.det(v) * np.linalg.det(w) < 0:
        sv[-1] = -sv[-1]
        v[:, -1] = -v[:, -1]
    u = np.dot(v, w)
    new_geom = np.dot(geom2, u) + center
    rmsd = np.sqrt(np.mean((new_geom - refgeom) ** 2))
    return rmsd, new_geom


ATOMIC_RADIUS = dict(H=0.31, He=0.28,
                     Li=1.28, Be=0.96, B=0.84, C=0.76, N=0.71, O=0.66, F=0.57, Ne=0.58,
                     Na=1.66, Mg=1.41, Al=1.21, Si=1.11, P=1.07, S=1.05, Cl=1.02, Ar=1.06)


def get_bond_list(geom, atoms=None, threshold=4, min_neighbors=4, snapshots=30, bond_threshold=1.8,
                  enforce=()):
    """Get the list of all the important atom pairs.
    Samples a number of snapshots from a list of geometries to generate all
    distances that are below a given threshold in any of them.

    Args:
        atoms:      Symbols for each atoms.
        geom:       One or a list of geometries to check for pairs
        threshold:  Threshold for including a bond in the bond list
        min_neighbors: Minimum number of neighbors to include for each atom.
                    If an atom has smaller than this number of bonds, additional
                    distances will be added to reach this number.
        snapshots:  Number of snapshots to be used in the generation, useful
                    for speeding up the process if the path is long and
                    atoms numerous.

    Returns:
        List of all the included interatomic distance pairs.
    """
    assert KDTree is not None, 'scipy is needed for geodesic interpolation'
    # Type casting and value checks on input parameters
    geom = np.asarray(geom)
    if len(geom.shape) < 3:
        # If there is only one geometry or it is flattened, promote to 3d
        geom = geom.reshape(1, -1, 3)
    min_neighbors = min(min_neighbors, geom.shape[1] - 1)

    # Determine which images to be used to determine distances
    snapshots = min(len(geom), snapshots)
    images = [0, len(geom) - 1]
    if snapshots > 2:
        images.extend(np.random.choice(range(1, snapshots - 1), snapshots - 2, replace=False))
    # Get neighbor list for included geometry and merge them
    rijset = set(enforce)
    for image in images:
        tree = KDTree(geom[image])
        pairs = tree.query_pairs(threshold)
        rijset.update(pairs)
        bonded = tree.query_pairs(bond_threshold)
        neighbors = {i: {i} for i in range(geom.shape[1])}
        for i, j in bonded:
            neighbors[i].add(j)
            neighbors[j].add(i)
        for i, j in bonded:
            for ni in neighbors[i]:
                for nj in neighbors[j]:
                    if ni != nj:
                        pair = tuple(sorted([ni, nj]))
                        if pair not in rijset:
                            rijset.add(pair)
    rijlist = sorted(rijset)
    # Check neighbor count to make sure `min_neighbors` is satisfied
    count = np.zeros(geom.shape[1], dtype=int)
    for i, j in rijlist:
        count[i] += 1
        count[j] += 1
    for idx, ct in enumerate(count):
        if ct < min_neighbors:
            _, neighbors = tree.query(geom[-1, idx], k=min_neighbors + 1)
            for i in neighbors:
                if i == idx:
                    continue
                pair = tuple(sorted([i, idx]))
                if pair in rijset:
                    continue
                else:
                    rijset.add(pair)
                    rijlist.append(pair)
                    count[i] += 1
                    count[idx] += 1
    if atoms is None:
        re = np.full(len(rijlist), 2.0)
    else:
        radius = np.array([ATOMIC_RADIUS.get(atom.capitalize(), 1.5) for atom in atoms])
        re = np.array([radius[i] + radius[j] for i, j in rijlist])
    logger.debug("Pair list contain %d pairs", len(rijlist))
    return rijlist, re


def compute_rij(geom, rij_list):
    """Calculate a list of distances and their derivatives

    Takes a set of cartesian geometries then calculate selected distances and their
    cartesian gradients given a list of atom pairs.

    Args:
        geom: Cartesian geometry of all the points.  Must be 2d numpy array or list
            with shape (natoms, 3)
        rij_list: list of indexes of all the atom pairs

    Returns:
        rij (array): Array of all the distances.
        bmat (3d array): Cartesian gradients of all the distances."""
    nrij = len(rij_list)
    rij = np.zeros(nrij)
    bmat = np.zeros((nrij, len(geom), 3))
    for idx, (i, j) in enumerate(rij_list):
        dvec = geom[i] - geom[j]
        rij[idx] = r = np.sqrt(dvec[0] * dvec[0] +
                               dvec[1] * dvec[1] + dvec[2] * dvec[2])
        grad = dvec / r
        bmat[idx, i] = grad
        bmat[idx, j] = -grad
    return rij, bmat


def compute_wij(geom, rij_list, func):
    """Calculate a list of scaled distances and their derivatives

    Takes a set of cartesian geometries then calculate selected distances and their
    cartesian gradients given a list of atom pairs.  The distances are scaled with
    a given scaling function.

    Args:
        geom: Cartesian geometry of all the points.  Must be 2d numpy array or list
            with shape (natoms, 3)
        rij_list: 2d numpy array of indexes of all the atom pairs
        func: A scaling function, which returns both the value and derivative.  Must
            qualify as a numpy Ufunc in order to be broadcasted to array elements.

    Returns:
        wij (array): Array of all the scaled distances.
        bmat (2d array): Cartesian gradients of all the scaled distances, with the
            second dimension flattened (need this to be used in scipy.optimize)."""
    geom = np.asarray(geom).reshape(-1, 3)
    nrij = len(rij_list)
    rij, bmat = compute_rij(geom, rij_list)
    wij, dwdr = func(rij)
    for idx, grad in enumerate(dwdr):
        bmat[idx] *= grad
    return wij, bmat.reshape(nrij, -1)


def morse_scaler(re=1.5, alpha=1.7, beta=0.01):
    """Returns a scaling function that determines the metric of the internal
    coordinates using morse potential

    Takes an internuclear distance, returns the scaled distance, and the
    derivative of the scaled distance with respect to the unscaled one.
    """
    def scaler(x):
        ratio = x / re
        val1 = np.exp(alpha * (1 - ratio))
        val2 = beta / ratio
        dval = -alpha / re * val1 - val2 / x
        return val1 + val2, dval
    return scaler


# ----------------------------------------------------------------------------
# Here is the Geodesic class


class Geodesic(object):
    """Optimizer to obtain geodesic in redundant internal coordinates.  Core part is the calculation
    of the path length in the internal metric."""
    def __init__(self, atoms, path, scaler=1.7, threshold=3, min_neighbors=4, log_level=logging.INFO,
                 friction=1e-3):
        """Initialize the interpolater
        Args:
            atoms:      Atom symbols, used to lookup radii
            path:       Initial geometries of the path, must be of dimension `nimage * natoms * 3`
            scaler:     Either the alpha parameter for morse potential, or an explicit scaling function.
                        It is easier to get smoother paths with small number of data points using small
                        scaling factors, as they have large range, but larger values usually give
                        better energetics because they better represent the (sharp) energy landscape.
            threshold:  Distance cut-off for constructing inter-nuclear distance coordinates.  Note that
                        any atoms linked by three or less bonds will also be added.
            min_neighbors:  Minimum number of neighbors an atom must have in the atom pair list.
            log_level:  Logging level to use.
            friction:   Friction term in the target function which regularizes the optimization step
                        size to prevent explosion.
        """
        rmsd0, self.path = align_path(path)
        logger.log(log_level, "Maximum RMSD change in initial path: %10.2f", rmsd0)
        if self.path.ndim != 3:
            raise ValueError('The path to be interpolated must have 3 dimensions')
        self.nimages, self.natoms, _ = self.path.shape
        # Construct coordinates
        self.rij_list, self.re = get_bond_list(path, atoms, threshold=threshold, min_neighbors=min_neighbors)
        if isinstance(scaler, float):
            self.scaler = morse_scaler(re=self.re, alpha=1.7)
        else:
            self.scaler = scaler
        self.nrij = len(self.rij_list)
        self.friction = friction
        # Initalize interal storages for mid points, internal coordinates and B matrices
        logger.log(log_level, "Performing geodesic smoothing")
        logger.log(log_level, "  Images: %4d  Atoms %4d Rijs %6d", self.nimages, self.natoms, len(self.rij_list))
        self.neval = 0
        self.w = [None] * len(path)
        self.dwdR = [None] * len(path)
        self.X_mid = [None] * (len(path) - 1)
        self.w_mid = [None] * (len(path) - 1)
        self.dwdR_mid = [None] * (len(path) - 1)
        self.disps = self.grad = self.segment = None
        self.conv_path = []

    def update_intc(self):
        """Adjust unknown locations of mid points and compute missing values of internal coordinates
        and their derivatives.  Any missing values will be marked with None values in internal storage,
        and this routine finds and calculates them.  This is to avoid redundant evaluation of value and
        gradients of internal coordinates."""
        for i, (X, w, dwdR) in enumerate(zip(self.path, self.w, self.dwdR)):
            if w is None:
                self.w[i], self.dwdR[i] = compute_wij(X, self.rij_list, self.scaler)
        for i, (X0, X1, w) in enumerate(zip(self.path, self.path[1:], self.w_mid)):
            if w is None:
                self.X_mid[i] = Xm = (X0 + X1) / 2
                self.w_mid[i], self.dwdR_mid[i] = compute_wij(Xm, self.rij_list, self.scaler)

    def update_geometry(self, X, start, end):
        """Update the geometry of a segment of the path, then set the corresponding internal
        coordinate, derivatives and midpoint locations to unknown"""
        X = X.reshape(self.path[start:end].shape)
        if np.array_equal(X, self.path[start:end]):
            return False
        self.path[start:end] = X
        for i in range(start, end):
            self.w_mid[i] = self.w[i] = None
        self.w_mid[start - 1] = None
        return True

    def compute_disps(self, start=1, end=-1, dx=None, friction=1e-3):
        """Compute displacement vectors and total length between two images.
        Only recalculate internal coordinates if they are unknown."""
        if end < 0:
            end += self.nimages
        self.update_intc()
        # Calculate displacement vectors in each segment, and the total length
        vecs_l = [wm - wl for wl, wm in zip(self.w[start - 1:end], self.w_mid[start - 1:end])]
        vecs_r = [wr - wm for wr, wm in zip(self.w[start:end + 1], self.w_mid[start - 1:end])]
        self.length = np.sum(np.linalg.norm(vecs_l, axis=1)) + np.sum(np.linalg.norm(vecs_r, axis=1))
        if dx is None:
            trans = np.zeros(self.path[start:end].size)
        else:
            trans = friction * dx  # Translation from initial geometry.  friction term 
        self.disps = np.concatenate(vecs_l + vecs_r + [trans])
        self.disps0 = self.disps[:len(vecs_l) * 2]

    def compute_disp_grad(self, start, end, friction=1e-3):
        """Compute derivatives of the displacement vectors with respect to the Cartesian coordinates"""
        # Calculate derivatives of displacement vectors with respect to image Cartesians
        l = end - start + 1
        self.grad = np.zeros((l * 2 * self.nrij + 3 * (end - start) * self.natoms, (end - start) * 3 * self.natoms))
        self.grad0 = self.grad[:l * 2 * self.nrij]
        grad_shape = (l, self.nrij, end - start, 3 * self.natoms)
        grad_l = self.grad[:l * self.nrij].reshape(grad_shape)
        grad_r = self.grad[l * self.nrij:l * self.nrij * 2].reshape(grad_shape)
        for i, image in enumerate(range(start, end)):
            dmid1 = self.dwdR_mid[image - 1] / 2
            dmid2 = self.dwdR_mid[image] / 2
            grad_l[i + 1, :, i, :] = dmid2 - self.dwdR[image]
            grad_l[i, :, i, :] = dmid1
            grad_r[i + 1, :, i, :] = -dmid2
            grad_r[i, :, i, :] = self.dwdR[image] - dmid1
        for idx in range((end - start) * 3 * self.natoms):
            self.grad[l * self.nrij * 2 + idx, idx] = friction

    def compute_target_func(self, X=None, start=1, end=-1, log_level=logging.INFO, x0=None, friction=1e-3):
        """Compute the vectorized target function, which is then used for least
        squares minimization."""
        if end < 0:
            end += self.nimages
        if X is not None and not self.update_geometry(X, start, end) and self.segment == (start, end):
            return
        self.segment = start, end
        dx = np.zeros(self.path[start:end].size) if x0 is None else self.path[start:end].ravel() - x0.ravel()
        self.compute_disps(start, end, dx=dx, friction=friction)
        self.compute_disp_grad(start, end, friction=friction)
        self.optimality = np.linalg.norm(np.einsum('i,i...', self.disps, self.grad), ord=np.inf)
        logger.log(log_level, "  Iteration %3d: Length %10.3f |dL|=%7.3e", self.neval, self.length, self.optimality)
        self.conv_path.append(self.path[1].copy())
        self.neval += 1

    def target_func(self, X, **kwargs):
        """Wrapper around `compute_target_func` to prevent repeated evaluation at
        the same geometry"""
        self.compute_target_func(X, **kwargs)
        return self.disps

    def target_deriv(self, X, **kwargs):
        """Wrapper around `compute_target_func` to prevent repeated evaluation at
        the same geometry"""
        self.compute_target_func(X, **kwargs)
        return self.grad

    def smooth(self, tol=1e-3, max_iter=50, start=1, end=-1, log_level=logging.INFO, friction=None,
               xref=None):
        """Minimize the path length as an overall function of the coordinates of all the images.
        This should in principle be very efficient, but may be quite costly for large systems with
        many images.

        Args:
            tol:        Convergence tolerance of the optimality. (.i.e uniform gradient of target func)
            max_iter:   Maximum number of iterations to run.
            start, end: Specify which section of the path to optimize.
            log_level:  Logging level during the optimization

        Returns:
            The optimized path.  This is also stored in self.path
        """
        assert least_squares is not None, 'scipy is needed for geodesic interpolation'
        X0 = np.array(self.path[start:end]).ravel()
        if xref is None:
            xref= X0
        self.disps = self.grad = self.segment = None
        logger.log(log_level, "  Degree of freedoms %6d: ", len(X0))
        if friction is None:
            friction = self.friction
        # Configure the keyword arguments that will be sent to the target function.
        kwargs = dict(start=start, end=end, log_level=log_level, x0=xref, friction=friction)
        self.compute_target_func(**kwargs)  # Compute length and optimality
        if self.optimality > tol:
            result = least_squares(self.target_func, X0, self.target_deriv, ftol=tol, gtol=tol,
                                   max_nfev=max_iter, kwargs=kwargs, loss='soft_l1')
            self.update_geometry(result['x'], start, end)
            logger.log(log_level, "Smoothing converged after %d iterations", result['nfev'])
        else:
            logger.log(log_level, "Skipping smoothing: path already optimal.")
        rmsd, self.path = align_path(self.path)
        logger.log(log_level, "Final path length: %12.5f  Max RMSD in path: %10.2f", self.length, rmsd)
        return self.path

    def sweep(self, tol=1e-3, max_iter=50, micro_iter=20, start=1, end=-1):
        """Minimize the path length by adjusting one image at a time and sweeping the optimization
        side across the chain.  This is not as efficient, but scales much more friendly with the
        size of the system given the slowness of scipy's optimizers.  Also allows more detailed
        control and easy way of skipping nearly optimal points than the overall case.

        Args:
            tol:        Convergence tolerance of the optimality. (.i.e uniform gradient of target func)
            max_iter:   Maximum number of sweeps through the path.
            micro_iter: Number of micro-iterations to be performed when optimizing each image.
            start, end: Specify which section of the path to optimize.
            log_level:  Logging level during the optimization

        Returns:
            The optimized path.  This is also stored in self.path
        """
        if end < 0:
            end = self.nimages + end
        self.neval = 0
        images = range(start, end)
        logger.info("  Degree of freedoms %6d: ", (end - start) * 3 * self.natoms)
        # Microiteration convergence tolerances are adjusted on the fly based on level of convergence.
        curr_tol = tol * 10
        self.compute_disps()    # Compute and print the initial path length
        logger.info("  Initial length: %8.3f", self.length)
        for iteration in range(max_iter):
            max_dL = 0
            X0 = self.path.copy()
            for i in images[:-1]:   # Use self.smooth() to optimize individual images
                xmid = (self.path[i - 1] + self.path[i + 1]) * 0.5
                self.smooth(curr_tol, max_iter=min(micro_iter, iteration + 6),
                            start=i, end=i + 1, log_level=logging.DEBUG,
                            friction=self.friction if iteration else 0.1,
                            xref=xmid)
                max_dL = max(max_dL, self.optimality)
            self.compute_disps()    # Compute final length after sweep
            logger.info("Sweep %3d: L=%7.2f dX=%7.2e tol=%7.3e dL=%7.3e",
                     iteration, self.length, np.linalg.norm(self.path - X0), curr_tol, max_dL)
            if max_dL < tol:    # Check for convergence.
                logger.info("Optimization converged after %d iteartions", iteration)
                break
            curr_tol = max(tol * 0.5, max_dL * 0.2) # Adjust micro-iteration threshold
            images = list(reversed(images))         # Alternate sweeping direction.
        else:
            logger.info("Optimization not converged after %d iteartions", iteration)
        rmsd, self.path = align_path(self.path)
        logger.info("Final path length: %12.5f  Max RMSD in path: %10.2f", self.length, rmsd)
        return self.path