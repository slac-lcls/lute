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

CMD="${@}"
CMD="${CMD} --partition=${PARTITION} --account=${ACCOUNT}"
echo $CMD
CMD="/sdf/group/lcls/ds/tools/lute/lute_launcher ${CMD}"
SLURM_ARGS="--partition=${PARTITION} --account=${ACCOUNT} --ntasks=1"
echo "Running ${CMD} with ${SLURM_ARGS}"
sbatch $SLURM_ARGS --wrap "${CMD}"
