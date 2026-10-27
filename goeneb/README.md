The GöNEB code uses the framework of the [Nudged Elastic Band](https://theory.cm.utexas.edu/henkelman/pubs/jonsson98_385.pdf) method described by Jonsson et.al.

> [!NOTE]
> Only tested with python 3.8, and the exact versions of the libraries given in `requirements.txt`!

# Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/cchembio/goeneb.git
   cd goeneb
   ```

---

2. **Create and activate a virtual environment**

    1. **Select your Python version**<br>
       Ensure you have your desired Python version installed on your system (Python 3.8 is recommended):
       ```bash
       python --version
       ```

    2. **Create the virtual environment**
       ```bash
       python -m venv neb-env
       ```
       Replace `neb-env` with any name you prefer for your environment.

    3. **Activate the environment**
       ```bash
       source neb-env/bin/activate
       ```

    4. **Install required Python modules**

       Install the dependencies from the `requirements.txt` file:

       ```bash
       pip install -r requirements.txt
       ```

---

3. **Prepare the shell environment**

   The file `scripts/environment.sh` is used to generate a temporary directory for the calculations, export all environment variables and source the python environment. The lines

   ```bash
   ENV_PATH="/path/to/neb-env/bin/activate"
   NEB_PATH="/path/to/goeneb"
   ```
   need to be changed to the full paths of the python environment and the GöNEB program.

   **Temporary files**<br>
   The `environment.sh` script also takes care of creating a temporary directory. By default this `NEB_TMPDIR` is created inside the folder specified by the environment variable `TMP_DIR`. 

   > [!WARNING]
   > Please make sure that `TMP_DIR` is either already set on your system, or that you are happy with its default value (`"/scr/${USER}"`).

   **Executables**<br>
   If you plan on using an external program for the energies and gradients (ORCA, Gaussian, Molpro), the location of the executable must be exported. 

   > [!NOTE]
   > The `environment.sh` expects the executable of the quantum chemistry program you are planning to use (ORCA, Molpro, Gaussian) to be in the `PATH`. Either permanently, or e.g. by loading the module. If this is not the case, the path to the program needs to be specified here.

---

4. **Test your installation**

   The `scripts/test_NEB.sh` can now be used to start the test module (`testing_module.py`).
   
   ```bash
   bash path/to/goeneb/scripts/test_NEB.sh
   ```

    It tests all the energy calculation interfaces (ORCA, Molpro, Gaussian and tblite) and checks whether the calculated values for energies and gradients are similar to the values calculated on our system.

   Furthermore, it runs the important functions from the NEB routine together with the dummy potential and tests whether all functions return the correct values. The tested functions are:
   - Alignment
   - Interpolation (Cartesian, Internal ([z-matrix](https://doi.org/10.1002/jcc.27029)), [geodesic](https://doi.org/10.1063/1.5090303) (non-deterministic) and [IDPP](https://doi.org/10.1063/1.4878664))
   - NEB gradients (tangents, springs, complete NEB-gradient)
   - Step predictors (SD, AMGD, RFO, SCT)

   The test returns a file `test.log` to your current directory. You should get CHECK for all tested functions (except geodesic).

---

**The GöNEB program is now ready to use!**


# Calculating NEBs
1. **Set up a job directory.**

   **OPTION 1:** Use the whole directory as input for the GöNEB

      - This directory must include exactly *one* `*.ini` file with the options for the NEB
      - It is possible to define start, end and TS structures in the ini-file via their path. This way the xyz-structures may be located elsewhere
      - It is also possible to not specify the input structures. When choosing this option you should name your structures `*1*.xyz` for the start structure, `*2*.xyz` for the end structure and `*TS*.xyz` for the TS if you want to include this structure. If your naming was not recognized by the program, you will get a warning and the structures are sorted alphabetically.

   **OPTION 2:** Use the `*.ini` file as the only input for the GöNEB program. You need to specify at least the start and end structure via their path in the ini file. They may be located anywhere on your system.

> [!IMPORTANT]
> You can find further information on how to write the ini file later in this ReadMe.
> 
> Also: An example for a complete calculation is in the folder `goeneb/testjobs/example`. There you can see a complete
> input and output.

---

2. **Run the program**
   
   Run the `scripts/run_NEB.sh`. This sets up the environment, runs the program and then deletes the temporary files.

   ```bash
   bash path/to/goeneb/scripts/run_NEB.sh
   ```

---

The program will produce:
- `batchlog.out`: The slurm output, preferably this file should stay almost empty
- `output.log`: The NEB output, all info you need to follow the NEB calculation
- `results`:
    - `starttraj.xyz`: The starting trajectory of the NEB
    - `currenttraj.xyz`: The current trajectory during the NEB calculation. Continuously updated while the job is running.
    - `finaltraj.xyz`: The final trajectory (not necessarily converged)
    - `optlog.csv`: A csv file with different properties for tracking the calculation that can be visualized with the `optplot.py` program
    - `HEI.xyz` and `HEI_trj.xyz`: The highest energy image and its trajectory over the course of the NEB optimization
    - `TS_guess_cubic.xyz` and `TS_guess_cubic_trj.xyz`: A [cubic interpolation](https://doi.org/10.1063/1.1323224) of the images with the highest energy, that should be closest to the actual TS, and its trajectory over the course of the NEB optimization.

---

# Keywords
The `neb.ini` file should contain all options of the NEB calculation. It starts with the line:
```ini
[options]
```
Each subsequent line should be structured like: 
```ini
keyword = value
```

All keywords are explained in the following tables, default values are printed **bold**.

## Initial Set-up and interpolation
| Keyword  | Options  | Explanation |
|----------|----------|----------|
| start_structure | `path-to/struct.xyz` | The start structure, only explicitly needed, if the input is only this file, not the whole directory. Can be a relative path. |
| end_structure | `path-to/struct.xyz` | The start structure, only explicitly needed, if the input is only this file, not the whole directory. Can be a relative path.|
| TS_guess | **None**<br>`path-to/struct.xyz` | The TS structure, will be included in initial the interpolation. Can be a relative path.|
| starttraj | **None**<br>`path-to/traj.xyz` | The file with the starting path, from where the NEB should be started. None means, a starting path is generated. Can be a relative path.|
| n_images | **default: 11** <br>integer $>$ 0 | Number of interpolation images. The total number of images (including ends) will be `n_images + 2`|
| frozen_atom_indices | **default: []**<br>integers seperated by comma | Usage: `frozen_atom_indices = 0,4,5`, this will freeze the atoms with index 0, 4 and 5 during the NEB. To also ensure they are not moving during the path initialization one should use cartesian (+ IDPP) or SIDPP as initial path with `rot_align_mode = None` and `translate_mode = collective or None`.|
| interp_mode | **internal**<br>cartesian<br>geodesic | How the initial path is created. Internal uses the [z-matrix](https://doi.org/10.1002/jcc.27029) from the `ChemCoord`-package, [geodesic](https://doi.org/10.1063/1.5090303) uses the code by Zhu et.al. |
|trajtest | **False**<br>True | Whether to stop after the initial trajectory has been created. |
| charge | **default: 0**<br>any integer | The charge of the system.|
|spin| **default: 1**<br>any integer | The spin of the system.|
|failed_img_tol_percent |  **default: 1.0**<br>float between 0 and 1| The maximum fraction of images that can have failed calculations before the NEB aborts. |
| rot_align_mode | **pairwise**<br>single_reference<br>None | How the rotation is removed from the initial path. Rotation is removed via Kabsch algorithm, either pairwise to the neighboring image, or single reference to the start structure. |
| translate_mode | **individual**<br>collective<br>None | How the translation is removed from the initial path. Either each image is individually centered, or they are centered on the starting structure and every other image is shifted accordingly. |
| remove_gradtrans | **True**<br>False | Whether or not translations are included in the calculation of the NEB-gradient.|
| remove_gradrot | **False**<br>True |Whether or not rotations are included in the calculation of the NEB-gradient. |
| verbose | debug<br>**info**<br>warning<br>error<br>critical | The logging level (levels from the logging module). In how much detail should `output.log` be written.

### IDPP keywords
| Keyword  | Options  | Explanation |
|----------|----------|----------|
| IDPP | **True**<br>False | Whether an [IDPP](https://doi.org/10.1063/1.4878664) pre-optimization is done after the initial interpolation. |
| SIDPP | **False**<br>True| Whether a [Sequential IDPP](https://doi.org/10.1021/acs.jctc.3c01111) should be used for the initial path. |
| IDPP_maxiter | **default: 1000**<br>any integer | Maximum iterations during the IDPP or SIDPP pass.|
| IDPP_max_RMSF |**default: 0.00945**<br>any float | RMSF convergence criteria for the IDPP. In $E_h/\AA$.|
|IDPP_max_AbsF|**default: 0.0189**<br>any float | Maximum absolute value convergence criteria for the IDPP. In $E_h/\AA$.|

## Interface selection
| Keyword  | Options  | Explanation |
|----------|----------|----------|
| interface | orca<br>gaussian<br>molpro<br>tblite<br>dummy | What program should be used for the energy calculation. Dummy is a fast to calculate, non-chemical surface. Only for debugging. |
| gaussian_path<br>molpro_path<br>orca_path | **None**<br>`path-to/program` | This will override the GAUSS_EXE, MOL_EXE or ORCA_EXE environment variables used normally by the program. |
| gaussian_keywords<br>molpro_keywords<br>tblite_keywords<br>orca_keywords<br>orca_keywords2 | **None**<br> Any valid combination of keywords for the chosen interface | **Gaussian**: 'force' keyword is added automatically<br>**Molpro**: 'SET,CHARGE=...', 'SET,SPIN=...' and 'force' are added automatically<br>**ORCA**: 'NoAutoStart EnGrad Angs' is added automatically, second line can be added with orca_keywords2 |
| memory | **default: 10000**<br>any integer | Memory keyword, needed for Molpro and Gaussian.<br> MOLPRO: memory in MW, remember: this is assigned to each core.<br>GAUSSIAN: memory in MB, remember: this is the total memory |
|n_threads|**default: 1**<br>any integer | The cores used for parallelizing the energy calculation. **Has to be the same as the number of cores specified in the slurm file!** |

## NEB specific options
### Tangents and Spring Forces
| Keyword  | Options  | Explanation |
|----------|----------|----------|
| maxiter | **default: 500**<br>integer $>$ 0 | Maximum number of iterations of the NEB. |
| relaxed_neb | **False**<br>True | Whether to use relaxed convergence thresholds.|
| climbing_image | **False**<br>True | Whether or not climbing image should be activated (after an initial NEB optimization to relaxed convergence criteria). |
| spring_gradient | **difference**<br>projected<br>raw | The definition of the spring gradient. The difference-based version is the improved definition by [Henkelman and Jonsson](https://doi.org/10.1063/1.1323224), projected is the [original implementation of the NEB](https://doi.org/10.1142/9789812839664_0016), and the raw spring force is not projected onto the tangent direction. |
|k_const | default: **0.003**<br>any float  | The spring force constant in $E_h/\AA^2$. When using variable k, this is the maximum value for the springs.|
| use_vark | **False** | Whether to use the improved variable k scheme based on the Scheme by [Henkelmann and Jonsson](https://doi.org/10.1063/1.1329672).|
| vark_min_fac | **default: 0.1**<br>float between 0 and 1 | The minimal spring constant between images is `vark_min_fac * k_const` |
| use_analytical_springpos | **False**<br>True | Analytical spring positioning scheme. Calculates the ideal position of the images in the 1D case. Does not work with SCT step predictor.
|tangents|**henkjon**<br>simple| The type of tangents used. Most importantantly the [improved tangent](https://doi.org/10.1063/1.1323224) by Henkelman and Jonsson.|


### Step Predictors
| Keyword  | Options  | Explanation |
|----------|----------|----------|
|step_pred_method| **AMGD**<br>SD<br>SCT<br>RFO<br>NR | Adaptive Momentum Gradient Descent<br>Steepest Descent<br>Self Consistent Tangents<br>Rational Function Optimization with global BFGS<br>Newton Raphson with global BFGS|
| stepsize_fac | **default: 0.2**<br>any float | The step size factor used to scale the step in AMGD, SD, NR. Also in the AMGD steps leading up to any of the other methods (SCT, NR, RF) |
| max_step | **default: 0.05**<br>any float | The maximum step size in $\AA$. The step will be scaled back if it is too long.
| AMGD_max_gamma | **default: 0.9**<br>float between 0 and 1 | The maximum factor for including the last step in the AMGD method. |
| initial_hessian | **diagonal**<br>lindh | The initial Hessian is created either as identity or as [Lindh](https://doi.org/10.1016/0009-2614(95)00646-L) hessian. The diagonal identity will be scaled in the first step. The global Lindh hessian is just the diagonal block matrix. |
| BFGS_start | **default: 5**<br>integer $\geq$ 0 | The iteration, in which the BFGS update starts.  | 
| NR_start | **default: 10**<br>integer $\geq$ 0 | The iteration, in which the second order method (NR, RF or SCT) starts. |
| harmonic_stepsize_fac | **default: 0.01**<br>any float | The step size used for the AMGD in the sub-cycle of the harmonic NEB (only in the method SCT)|
| harmonic_conv_fac | **default: 0.7**<br>float between 0 and 1| The factor used to lower the convergence thresholds for the harmonic NEB cycle (only in method SCT)|

## Convergence Thresholds
All convergence Thresholds are the same ones as in ORCA.
| Keyword  | Options  | Explanation |
|----------|----------|----------|
|Max_RMSF_tol| **default: 0.000945**<br>any float | The RMSF convergence criteria for the normal NEB. In $E_h/\AA$.|
|Max_AbsF_tol| **default: 0.00189**<br>any float | The maximum absolute value convergence criteria for the normal NEB. In $E_h/\AA$.|
|CI_RMSF_tol| **default: 0.000473**<br>any float | The RMSF convergence criteria for the climbing image in the NEB. In $E_h/\AA$.|
|CI_AbsF_tol| **default: 0.000945**<br>any float | The maximum absolute value convergence criteria for the climbing image in the NEB. In $E_h/\AA$.|
|Relaxed_Max_RMSF_tol| **default: 0.00945**<br>any float | Relaxed RMSF convergence criteria for the NEB. This has to be reached for CI to start. In $E_h/\AA$.|
|Relaxed_Max_AbsF_tol| **default: 0.0189**<br>any float | Relaxed maximum absolute value convergence criteria for the NEB. This has to be reached for CI to start. In $E_h/\AA$.|

# Visualization with optplot

The optplot.py program transforms the optlog.csv file into nice, readable output. To visualize, follow these steps:

1. Open your NEB environment<br>
```bash
source path/to/neb-env/bin/activate
```
2. Run the program with<br>
```bash
path/to/goeneb/optplot.py -f path/to/optlog.csv
```

The program has the following arguments (can also be accessed by `path-to/goeneb/optplot.py --help`):

| short argument | long argument | Description |
|---|---|---|
|-f |--filename|Enter the file that should be visualized. No file opens the user input.|
|-s|--savefile|Enter the path where the file should be saved. No file means displaying the image directly.|

The optplot software gives you 4 different plots to visualize the progress of the NEB:

1. The progression of the convergence measures over the iterations.
2. The energy profile of the reaction, the energy profiles of previous iterations are also shown in lesser occupacity.
3. A 2-D projection of the image coordinates. The initial pathway is also shown and the movement of the images is tracked.
4. The energy gradient for each image in each iteration as a visualized matrix.

# The 2D example
The GöNEB comes with a two dimensional NEB model. It uses the functions in the real GöNEB but a model PES which is Himmelblaus function. On this 2D surface the NEB runs instantly and produces results that are easy to analyze since the optimization is in only 2 dimension. This can be used for visualization and testing of various functions.

The code for the 2D example can be run by the following command:
```bash
python /path/to/goeneb/2dtest.py -f <filename> -m <step-prediction-method>
```
The other arguments that can be parsed by the program are:

| Short | Long| Description|
|-------|------------------|------------------|
| -s| --stepsize_fac   | Step size factor|
| -m| --step_pred_method | Step predictor method (`AMGD`, `SD`, `RFO`, etc.)|
| -k    | --k_const| Spring constant|
| -t    | --tangents| Tangents calculation method (`henkjon`, `simple`)|
| -i    | --maxiter| Maximum number of iterations|
| -l    | --max_step| Maximum step length|
| -n    | --NR_start| Iteration when to start second order method|
| -b    | --BFGS_start| Iteration when to start BFGS updates|
| -d    | --draw_every_iteration | Draw output after every optimization iteration|
| -h    | --show_hessian   | Show hessian matrix|
| -c    | --caption| Title of the output graph|
| -f    | --filename| Filename of the produced image|
| -ff   | --fileformat| File format of the produced image (`pdf`, `png`, etc.)|

# Contact

This code is currently maintained and developed by Lynn Meeder (lynn.meeder@uni-goettingen.de).<br>
The base code was written by Björn Hein-Janke (bjoern.hein-janke@uni-goettingen.de).

# References

- Jónsson, H.; Mills, G.; Jacobsen, K. W. [Nudged Elastic Band Method for Finding Minimum Energy Paths of Transitions.](https://doi.org/10.1142/9789812839664_0016) In Classical and Quantum Dynamics in Condensed Phase Simulations; WORLD SCIENTIFIC: LERICI, Villa Marigola, 1998; pp 385–404. 
- Weser, O.; Hein-Janke, B.; Mata, R. A. [Automated Handling of Complex Chemical Structures in Z-Matrix Coordinates—the Chemcoord Library.](https://doi.org/10.1002/jcc.27029) J. Comput. Chem. 2022, 44 (5), 710–726.
- Zhu, X.; Thompson, K. C.; Martínez, T. J. [Geodesic Interpolation for Reaction Pathways.](https://doi.org/10.1063/1.5090303) J. Chem. Phys. 2019, 150 (16), 164103. 
- Smidstrup, S.; Pedersen, A.; Stokbro, K.; Jónsson, H. [Improved Initial Guess for Minimum Energy Path Calculations.](https://doi.org/10.1063/1.4878664) J. Chem. Phys. 2014, 140 (21), 214106. 
- Schmerwitz, Y. L. A.; Ásgeirsson, V.; Jónsson, H. [Improved Initialization of Optimal Path Calculations Using Sequential Traversal over the Image-Dependent Pair Potential Surface](https://doi.org/10.1021/acs.jctc.3c01111) J. Chem. Theory Comput. 2024, 20 (1), 155–163.
- Henkelman, G.; Uberuaga, B. P.; Jónsson, H. [A Climbing Image Nudged Elastic Band Method for Finding Saddle Points and Minimum Energy Paths.](https://doi.org/10.1063/1.1329672) J. Chem. Phys. 2000, 113 (22), 9901–9904. 
- Henkelman, G.; Jónsson, H. [Improved Tangent Estimate in the Nudged Elastic Band Method for Finding Minimum Energy Paths and Saddle Points.](https://doi.org/10.1063/1.1323224) J. Chem. Phys. 2000, 113 (22), 9978–9985.
- Lindh, R.; Bernhardsson, A.; Karlström, G.; Malmqvist, P.-Å. [On the Use of a Hessian Model Function in Molecular Geometry Optimizations.](https://doi.org/10.1016/0009-2614(95)00646-L) Chem. Phys. Lett. 1995, 241 (4), 423–428.
