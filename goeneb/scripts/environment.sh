#! /bin/bash -l

# User specified paths for python environment and NEB program path
ENV_PATH="/path/to/venv/bin/activate"
NEB_PATH="/path/to/goeneb"

if [ -z "$TMP_DIR" ]; then
    TMP_DIR="/tmp/${USER}"
fi

# Temp directory
JOB_ID="${SLURM_JOB_ID:-$$}"
NEB_TMPDIR="${TMP_DIR}/${JOB_ID}"
mkdir -p "${NEB_TMPDIR}"
export NEB_TMPDIR

# Executables
if [ -z "$GAUSS_EXE" ]; then
    export GAUSS_EXE="$(command -v g16)"
fi
if [ -z "$MOL_EXE" ]; then
    export MOL_EXE="$(command -v molpro)"
fi
if [ -z "$ORCA_EXE" ]; then
    export ORCA_EXE="$(command -v orca)"
fi
if [ -z "$MPI_PATH" ]; then
    export MPI_PATH="$(command -v mpirun)"
fi
export NEB_PATH

# Stacksize for tblite
export OMP_STACKSIZE=4G

# Python environment
source "${ENV_PATH}"