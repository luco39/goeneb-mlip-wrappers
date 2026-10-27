#! /bin/bash -l
 
source "$(dirname "$0")/environment.sh"
trap 'rm -rf "${NEB_TMPDIR}"' EXIT
python "$NEB_PATH/testing_module.py"