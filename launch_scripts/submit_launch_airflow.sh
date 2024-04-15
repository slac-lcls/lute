#!/bin/bash
CMD="${@}"
echo $CMD
CMD="/sdf/group/lcls/ds/tools/lute/lute_launcher ${CMD}"
SLURM_ARGS="--partition=milano --account=lcls:data --ntasks=1"
echo "Running ${CMD} with ${SLURM_ARGS}"
sbatch $SLURM_ARGS --wrap "${CMD}"
