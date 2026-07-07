#!/bin/bash

# Need to capture partition and account for SLURM
while [[ $# -gt 0 ]]
do
    case "$1" in
        --partition=*)
            PARTITION="${1#*=}"
            shift
            ;;
        --account=*)
            ACCOUNT="${1#*=}"
            shift
            ;;
        *)
            POS+=("$1")
            shift
            ;;
    esac
done
set -- "${POS[@]}"

# Bodge Kerberos credentials
# These duplicates are removed later by the workflow process
KERB_CACHE_PATH=$(klist -l | awk -F"FILE:" '{printf (NF>1)? $NF : ""}')
if [[ ! -d $HOME/.tmp_cache ]]; then
    mkdir $HOME/.tmp_cache
fi
cp $KERB_CACHE_PATH $HOME/.tmp_cache/kerbcache
echo $?
export KRB5CCNAME="FILE:${HOME}/.tmp_cache/kerbcache"

LUTE_BIN_PATH="$(cd "$(dirname ${BASH_SOURCE[0]})" &> /dev/null && pwd)"

# Detect install type
if [[ -f "${LUTE_BIN_PATH}/activate" ]]; then
    # Virtual environment install
    source "${LUTE_BIN_PATH}/activate"
    if [[ -n "${VIRTUAL_ENV}" ]]; then
        export LUTE_VIRTUAL_ENV="${VIRTUAL_ENV}"
    fi

    # Search for all Python version virtual environments at standard locations
    # Convention: lute_env_py<MAJORMINOR>/bin/activate (e.g. lute_env_py39, lute_env_py311)
    LUTE_ENVS_PARENT="$(dirname "$(dirname "${LUTE_BIN_PATH}")")" 
    for env_dir in "${LUTE_ENVS_PARENT}"/lute_env_py*/bin/activate; do
        if [[ -f "${env_dir}" ]]; then
            env_name="$(basename "$(dirname "$(dirname "${env_dir}")")")" 
            # Extract version: "lute_env_pyXYZ" → "XYZ"
            py_ver="${env_name#lute_env_py}"
            export "LUTE_VIRTUAL_ENV_PY${py_ver}"="$(dirname "$(dirname "${env_dir}")")" 
        fi
    done
else
    # Standard (meson/prefix) installation
    source "${LUTE_BIN_PATH}/activate_installation"
fi

CMD="${@}"
CMD="${CMD} --partition=${PARTITION} --account=${ACCOUNT}"
echo $CMD
SLURM_ARGS="--partition=${PARTITION} --account=${ACCOUNT} --ntasks=1"
echo "Running ${CMD} with ${SLURM_ARGS}"
sbatch $SLURM_ARGS --wrap "${CMD}"
export KRB5CCNAME="FILE:${KERB_CACHE_PATH}"
