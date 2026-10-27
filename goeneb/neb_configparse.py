import logging

import neb_exceptions as nex
from file_sys_io import qread, get_full_path
from helper import parse_index_list

logger = logging.getLogger(__name__)

class Settings:
    def __init__(self, neb_ini: str = None):
        # First all defaults
        # structures and files
        self.starttraj = None
        self.TS_guess = None
        self.start_structure = None
        self.end_structure = None
        self.tempdir = None

        # general options
        self.maxiter = 500
        self.climbing_image = False
        self.frozen_atom_indices = []
        self.relaxed_neb = False
        self.trajtest = False
        self.n_images = 11

        # springs and tangents
        self.use_vark = False
        self.k_const = 0.003
        self.tangents = 'henkjon'
        self.spring_gradient = 'difference'
        self.vark_min_fac = 0.1
        self.use_analytical_springpos = False

        # step prediction
        self.step_pred_method = 'amgd'
        self.stepsize_fac = 0.2
        self.harmonic_stepsize_fac = 0.01
        self.harmonic_conv_fac = 0.7
        self.AMGD_max_gamma = 0.9
        self.NR_start = 10
        self.BFGS_start = 5
        self.initial_hessian = 'diagonal'
        self.max_step = 0.05
        self.soft_reset = True
        self.soft_reset_memory = 5

        # rotations and translations
        self.rot_align_mode = 'pairwise'
        self.translate_mode = 'individual'
        self.remove_gradtrans = True
        self.remove_gradrot = False

        # IDPP and interpolation
        self.interp_mode = 'internal'
        self.IDPP = True
        self.SIDPP = False
        self.IDPP_maxiter = 1000
        self.IDPP_max_RMSF = 0.00945
        self.IDPP_max_AbsF = 0.0189

        # convergence
        self.Max_RMSF_tol = 0.000945
        self.Max_AbsF_tol = 0.00189
        self.CI_RMSF_tol = 0.000473
        self.CI_AbsF_tol = 0.000945
        self.Relaxed_Max_RMSF_tol = 0.00945
        self.Relaxed_Max_AbsF_tol = 0.0189
        self.failed_img_tol_percent = 1.0

        # interfaces
        self.interface = None
        self.molpro_path = None
        self.gaussian_path = None
        self.orca_path = None
        self.orca_keywords = None
        self.orca_keywords2 = None
        self.gaussian_keywords = None
        self.molpro_keywords = None
        self.tblite_keywords = None
        self.mace_model_path = None
        self.mace_foundation = None
        self.mace_size = 'medium'
        self.mace_device = 'cpu'
        self.mace_dtype = 'float64'
        self.uma_model = 'uma-s-1p1'
        self.uma_device = 'cpu'
        self.uma_task = 'omol'
        self.uma_batch_images = False
        self.mace_ensemble_model_paths = None
        self.mace_ensemble_uncertainty_threshold = None
        self.mace_ensemble_device = 'cpu'
        self.mace_ensemble_dtype = 'float64'
        self.mace_ensemble_uncertainty_metric = 'energy'
        self.mace_ensemble_batch_images = False
        self.mace_ensemble_max_batch_atoms = 0

        # cluster
        self.n_threads = 1
        self.memory = 10000
        self.verbose = 'info'

        # system
        self.charge = 0
        self.spin = 1

        # for case-insensitive checking of user settings
        self.allowed_keywords = set(self.__dict__.keys())
        self.allowed_keywords_map = {k.lower(): k for k in self.allowed_keywords}
        self.case_sensitive_kws = {'molpro_path',
                                   'gaussian_path',
                                   'orca_path',
                                   'orca_keywords',
                                   'orca_keywords2',
                                   'gaussian_keywords',
                                   'molpro_keywords',
                                   'tblite_keywords',
                                   'mace_model_path',
                                   'mace_ensemble_model_paths',
                                   'starttraj',
                                   'ts_guess',
                                   'start_structure',
                                   'end_structure',
                                   'tempdir'}

        if neb_ini is not None:
            user_settings = self.parse_inpfile(neb_ini)
            self.check_values(user_settings)
            self.set_values(user_settings)
        
        self.logic_check()

    def parse_inpfile(self, filepath:str):
        """
        Read the input file, remove any blank spaces, convert the 
        entrys to their respective types and return a dictionary.
        """
        user_settings = {}
        lines = qread(filepath)
        for line in lines:
            # ignore lines with no '=' in them
            if '=' not in line:
                continue
            try:
                clean_line = self.clean_eqsign(line)
                [newkey, newval] = clean_line.split('=', 1)
                user_settings[newkey] = self.convert_entry(newval, newkey)
            except (ValueError, IndexError):
                raise nex.ParsingError("Error in neb_configparse: couldn't parse" +
                                   "input file line: " + str(line))
        return user_settings
    
    def check_values(self, user_settings:dict):
        for key in user_settings:
            if key.lower() not in self.allowed_keywords_map:
                logger.error('Invalid keyword: %s. Please consult the manual for allowed keywords.', key)
                raise ValueError(f"Invalid keyword: {key}")
            
    def set_values(self, user_settings:dict):
        for key, value in user_settings.items():
            correct_key = self.allowed_keywords_map[key.lower()]
            setattr(self, correct_key, value)

    @staticmethod
    def clean_eqsign(text):
        """Remove blank spaces in the beginning, end and around the '='."""
        if '=' not in text:
            logger.warning("No '=' found in the input text.")
            return ""
        eqsign = text.index('=')
        left = text[:eqsign].strip()
        right = text[eqsign+1:].strip()
        return left + '=' + right

    def convert_entry(self, entry, key):
        """
        function for checking if a config entry
        is a special kind of type (e.g. None, bool, int,...)
        and convert them if they are. Only lower case if
        it is a string.
        """
        entry = entry.strip()
        if key == 'frozen_atom_indices':
            return parse_index_list(entry)
        if entry.lower() == 'none':
            return None
        elif entry.lower() == 'true':
            return True
        elif entry.lower() == 'false':
            return False
        else:
            # check for integer
            try:
                parsed_entry = int(entry)
            except ValueError:
                pass
            else:
                return parsed_entry
            # check for float
            try:
                parsed_entry = float(entry)
            except ValueError:
                pass
            else:
                return parsed_entry
            # it is a string
            if key.lower() in self.case_sensitive_kws:
                return entry
            else:
                return entry.lower()
        
    def get_idpp_settings(self):
        # copy the current settings attributes
        idpp_settings = type(self)()
        idpp_settings.__dict__.update(self.__dict__)

        # change some settings IDPP specific
        idpp_settings.maxiter = self.IDPP_maxiter
        idpp_settings.stepsize = 0.1
        idpp_settings.k_const = 1
        return idpp_settings
    
    def get_complete_paths(self):
        assert hasattr(self, 'workdir') and self.workdir is not None, "workdir must be set"

        # Define the path settings that need to be processed
        path_settings = ['start_structure', 'end_structure', 'starttraj', 'TS_guess']
        for path_name in path_settings:
            path_value = getattr(self, path_name)
            if path_value is not None:
                path = get_full_path(path_value)
                setattr(self, path_name, path)

                if path.exists:
                    logger.debug(f"The path {path_name} = {str(path)} exists")
                else:
                    logger.error(f"The path {path_name} = {str(path)} does not exist")
                    raise FileNotFoundError
                
    def logic_check(self):
        """Do some logical checks of the settings and print warnings."""
        # frozen atoms
        if len(self.frozen_atom_indices) > 0 and self.interp_mode != 'cartesian':
            logger.warning("Atoms do not remain frozen whem using internal or geodesic interpolation. "+
                           "You should use cartesian (+ IDPP) or SIDPP as an initial pathway.")
        if len(self.frozen_atom_indices) > 0 and self.rot_align_mode is not None:
            logger.warning("The force on the frozen atoms is zeroed, however the rotation alignment might move the whole image.")
        if len(self.frozen_atom_indices) > 0 and self.translate_mode == 'individual':
            logger.warning("The force on the frozen atoms is zeroed, however the translation alignment might move the whole image. "+
                           "Choose either None or 'collective'.")


    def __str__(self):
        excluded = {'allowed_keywords', 'allowed_keywords_map', 'case_sensitive_kws'}
        return "\n".join(
            f"{key} = {value}" 
            for key, value in self.__dict__.items()
            if key not in excluded)


