import numpy as np
import logging

from internal_module import calcBmat, calcKmat, convert_gradient

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------
# Hessian Objects

class hessian:
    def __init__(self, mode='diagonal', start=1, labels=[]):
        self.mode = mode
        self.inv_cart_hessian = None                # the hessian stored here is actually the inverse hessian
        self.start = max(start-1, 1)                # cant be less than 1
        self.labels = labels

        self.cart_grads = []
        self.cart_pvecs = []

    # Pointer to the hessian this class is gonna yield
    @property
    def current_inv_hessian(self):
        """
        Pointer to the inverse hessian
        """
        return self.inv_cart_hessian

    @property
    def current_hessian(self):
        """
        Pointer to the hessian
        """
        if self.inv_cart_hessian is None:
            return None
        else:
            return np.linalg.inv(self.inv_cart_hessian)

    @current_inv_hessian.setter
    def current_inv_hessian(self, hessian):
        self.inv_cart_hessian = hessian

    def initialize(self):
        """
        Provides the first initial hessian in based on the provided mode and coordinates:\n
        Mode:
        - diagonal: for a diagonal hessian either in internal or in cartesian coordinates
        - lindh: from internal coordinates
        """
        mode = self.mode
        img_dim = len(self.cart_pvecs[self.start])
        if mode == 'diagonal':
            hessian = Initial_Hess().get_Diag_Hess(img_dim=img_dim)
            hessian = self.scale_cartesian_hessian(hessian)

        elif mode == 'lindh':
            n_cart = len(self.labels)*3
            if n_cart == len(self.cart_pvecs[-1]):
                hessian = Initial_Hess().get_Lindh_Hess(self.cart_pvecs[-1],
                                                        self.labels,
                                                        self.cart_grads[-1])
            else:
                # Global hessian requested
                try:
                    pvecs = np.array(self.cart_pvecs[-1]).reshape((-1, n_cart))
                    grads = np.array(self.cart_grads[-1]).reshape((-1, n_cart))
                except ValueError as e:
                    logger.error(f"Reshape error while building global hessian: check that array size matches labels! {e}")
                    raise
                lindh_hessians = [Initial_Hess().get_Lindh_Hess(pvec, self.labels, grad) for pvec, grad in zip(pvecs, grads)]

                try:
                    import scipy as sp
                except ImportError:
                    logger.error('For global Lindh hessian scipy is needed, but the module can not be imported.')
                    raise ImportError

                global_hessian = sp.linalg.block_diag(*lindh_hessians)
                return global_hessian
        else:
            logger.error('%s is not a valid mode for hessian initialization.', mode)
            raise ValueError('No valid hessian initialization mode selected.')

        # Return the hessian
        return hessian

    def scale_cartesian_hessian(self, hessian):
        """
        The initial scaling of the hessian in the first step.
        Done for the real hessian.
        """
        s = self.cart_pvecs[self.start] - self.cart_pvecs[self.start - 1]
        y = self.cart_grads[self.start] - self.cart_grads[self.start - 1]
        scalar = np.dot(y,y)/np.dot(s,y)            # This is the inverse of the scalar for the inverse hessian update
        logger.debug('Hessian is scaled by %s.', scalar)
        hessian = scalar * hessian
        return hessian

    def update(self):
        """
        Doing the BFGS Update formula for the inverse hessian.
        """
        assert len(self.cart_pvecs) >= 2
        assert len(self.cart_grads) >= 2

        logger.debug('Updating the inverse hessian matrix.')
        initial_hessian = self.initialize()
        Hk = np.linalg.inv(initial_hessian)

        pvecs = self.cart_pvecs
        grads = self.cart_grads

        n = len(pvecs)
        sk = [pvecs[i] - pvecs[i-1] for i in range(self.start, n)]
        yk = [grads[i] - grads[i-1] for i in range(self.start, n)]
        rhos = [1/np.dot(s,y) for s, y in zip(sk, yk)]
        I = np.identity(len(sk[0]))

        # updating the inverse hessian
        for s, y, rho in zip(sk, yk, rhos):
            nu1 = I - rho * np.outer(s, y)
            nu2 = I - rho * np.outer(y, s)
            Hk = np.dot(nu1, np.dot(Hk, nu2)) + rho * np.outer(s, s)

        # Set according to the setter function
        eigvals, eigvecs = np.linalg.eig(np.linalg.inv(Hk))
        logger.debug('Eigenvalues of the current hessian: %s', eigvals)
        self.current_inv_hessian = Hk

    def get_inv_hessian(self):
        if self.current_inv_hessian is None:
            return None
        else:
            return self.current_inv_hessian.copy()

    def get_hessian(self):
        if self.current_hessian is None:
            return None
        else:
            return self.current_hessian.copy()

    def set_hessian(self, hess, is_inverse=None):
        assert is_inverse is not None
        if is_inverse:
            self.inv_cart_hessian = hess
        else:
            self.inv_cart_hessian = np.linalg.inv(hess)

    def reset(self):
        self.inv_cart_hessian = None
        self.cart_grads = []
        self.cart_pvecs = []
        self.start = 1
        logger.warning('Resetting Hessian')

    def soft_reset(self, memory=5):
        self.inv_cart_hessian = None
        self.cart_grads = self.cart_grads[-memory:]
        self.cart_pvecs = self.cart_pvecs[-memory:]
        self.start = 1
        logger.warning('Soft-Resetting Hessian')

# ------------------------------------------------------------------------------------------------------------------

class Initial_Hess:
    """
    Creates and assignes the Hessian as defined by Lindh. Uses a basic definition in which all possible stretches, bends and twists
    except the redundant inverse are used to create the internal coordinates. These coordinates are then used to directly transform
    to a cartesian Hessian.
    NOTE: Bakken and Helgaker actually used the actual coordinate set used in the optimization to determine Lindh's Hessian.
    """
    def __init__(self):
        # Define the constants used for the Lindh model hessian
        # converted to Angstrom and Hartree
        self.BohrinA = 0.52918
        self.lindh_k = np.array((0.45 / (self.BohrinA **2), 0.15, 0.005))  #kr, kφ, kτ
        self.lindh_aij = np.array(((1.0000, 0.3949, 0.3949, 0.3949), #Too redundant for a tuple?
                          (0.3949, 0.2800, 0.2800, 2.800),
                          (0.3949, 0.2800, 0.2800, 2.500),
                          (0.3949, 0.2800, 0.2500, 2.500)))
        self.lindh_rij = np.array(((1.35, 2.10, 2.53, 2.76), #And of course symmetric
                              (2.10, 2.87, 3.40, 3.71),
                              (2.53, 3.40, 3.40, 3.71),
                              (2.76, 3.71, 3.71, 3.80)))
        self.period = {'H' : 1, 'He' : 1,
                       'Li' : 2, 'Be' : 2, 'B' : 2, 'C' : 2,
                       'N' : 2,'O' : 2, 'F' : 2, 'Ne' : 2,
                       'Na' : 3, 'Mg' : 3, 'Al' : 3, 'Si' : 3,
                       'P' : 3, 'S' : 3, 'Cl' : 3, 'Ar' : 3}

    def get_period(self, atom_labels):
        try:
            return [self.period[atom] for atom in atom_labels]
        except KeyError as e:
            raise ValueError(
                f"Atom '{e.args[0]}' could not be assigned to the first three periods. "
                "This is a problem during the creation of the Lindh Hessian.")

    def get_Diag_Hess(self, img_dim=None):
        """ 
        The cartesian digonal hessian is returned.

        Returns:
        - Diagonal hessian in cartesian coordinates
        """
        assert img_dim is not None
        new_hess = np.identity(img_dim)
        logger.debug('Hessian is initialized as identity.')  
        return new_hess

    def get_Lindh_Hess(self, cart_coords, atom_labels, cart_grad):
        """
        Creates and assignes the Hessian as defined by Lindh. 
        Returns:
        - The hessian in internal coordinates
        """
        bonds, angles, dihedrals, k_list = self.get_Lindh_coords(cart_coords,
                                                                 self.get_period(atom_labels))
        internal_hess = np.diag(k_list)
        lindh_coords = bonds + angles + dihedrals
        B = calcBmat(cart_coords, lindh_coords)
        int_grad = convert_gradient(cart_grad, B)
        K = calcKmat(cart_coords, lindh_coords, int_grad)
        cart_hess = B.T @ internal_hess @ B

        # modify cartesian hessian
        eigvals, eigvecs = np.linalg.eigh(cart_hess)
        new_eigvals = np.array([1 if np.abs(val) < 1e-6 else val for val in eigvals])
        cart_hess_mod = eigvecs @ np.diag(new_eigvals) @ eigvecs.T + K
        logger.debug('Hessian is initialized as Lindh Hessian.')
        return cart_hess_mod

    def get_Lindh_coords(self, cart_coords, atom_periods):
        """
        Not needed.
        """
        # Set up of empty lists
        thresh_zero = 1.0e-6
        tmpstretch = []
        tmpbend = []
        tmptwist = []
        tmpks = []
        natoms = len(atom_periods)

        # Stretches
        for a in range(natoms):
            for b in range(a+1, natoms):
                p1  = atom_periods[a]-1
                p2  = atom_periods[b]-1
                val = np.linalg.norm(cart_coords[b*3:b*3+3] - cart_coords[a*3:a*3+3])
                k   = self.lindh_k[0] * np.exp( -self.lindh_aij[p1,p2] * (val**2 - self.lindh_rij[p1,p2]**2))
                if k > thresh_zero:
                    tmpstretch.append((a, b))
                    tmpks.append(k)
        # Bends
        for a in range(natoms):
            for b in range(natoms):
                if a != b:
                    for c in range(a+1, natoms):
                        if b != c:
                            v12 = cart_coords[a*3:a*3+3] - cart_coords[b*3:b*3+3]
                            v23 = cart_coords[c*3:c*3+3] - cart_coords[b*3:b*3+3]
                            dist12 = np.linalg.norm(v12)
                            dist23 = np.linalg.norm(v23)

                            p1  = atom_periods[a]-1
                            p2  = atom_periods[b]-1
                            p3  = atom_periods[c]-1
                            k   = self.lindh_k[1] * np.exp(-self.lindh_aij[p1,p2] * (dist12**2 - self.lindh_rij[p1,p2]**2)) *\
                                            np.exp(-self.lindh_aij[p2,p3] * (dist23**2 - self.lindh_rij[p2,p3]**2))
                            if k > thresh_zero:
                                tmpbend.append((a,b,c))
                                tmpks.append(k)

        # Twists
        for a in range(natoms):
            for b in range(natoms):
                if a != b:
                    for c in range(natoms):
                        if a != c and b != c:
                            for d in range(a+1, natoms):
                                if d != b and d != c:
                                    v12 = cart_coords[a*3:a*3+3] - cart_coords[b*3:b*3+3]
                                    v23 = cart_coords[c*3:c*3+3] - cart_coords[b*3:b*3+3]
                                    v34 = cart_coords[d*3:d*3+3] - cart_coords[c*3:c*3+3]
                                    dist12 = np.linalg.norm(v12)
                                    dist23 = np.linalg.norm(v23)
                                    dist34 = np.linalg.norm(v34)
                                    angle123 = np.arccos(np.dot(v12,v23)/(dist12*dist23))*180.0/np.pi
                                    angle234 = np.arccos(np.dot(v23,v34)/(dist23*dist34))*180.0/np.pi
                                    if angle123 < 175.0 and angle123 > 5.0 and angle234 < 175.0 and angle234 > 5.0:
                                        p1  = atom_periods[a]-1
                                        p2  = atom_periods[b]-1
                                        p3  = atom_periods[c]-1
                                        p4  = atom_periods[d]-1
                                        k   = self.lindh_k[2] * np.exp(-self.lindh_aij[p1,p2] * (dist12**2 - self.lindh_rij[p1,p2]**2)) *\
                                                        np.exp(-self.lindh_aij[p2,p3] * (dist23**2 - self.lindh_rij[p2,p3]**2)) *\
                                                        np.exp(-self.lindh_aij[p3,p4] * (dist34**2 - self.lindh_rij[p3,p4]**2))
                                        if k > thresh_zero:
                                            tmptwist.append((a, b, c, d))
                                            tmpks.append(k)

        return (tmpstretch, tmpbend, tmptwist, tmpks)
