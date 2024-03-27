#!/bin/bash
usage()
{
    cat << EOF
$(basename "$0"):
    Submit a LUTE managed Task using SLURM on S3DF.
    Options:
        -c|--config
          ABSOLUTE path to the LUTE configuration YAML. Must be absolute.
        -h|--help
          Display this message.
        -t|--taskname
          Name of the LUTE managed Task to run.

    NOTE: This script does not parse SLURM arguments, but a number of them are
          mandatory. All additional arguments are transparently passed to SLURM.
          You will need to provide at least the queue and account using, e.g.:
                  --partition=milano --account=lcls:<experiment>
EOF
}

POSITIONAL=()
while [[ $# -gt 0 ]]
do
    flag="$1"

    case $flag in
    -c|--config)
      CONFIGPATH="$2"
      shift
      shift
      ;;
    -h|--help)
        usage
        exit
        ;;
    -t|--taskname)
        TASK="$2"
        shift
        shift
        ;;
    --debug)
        DEBUG=1
        shift
        ;;
    *)
        POS+=("$1")
        shift
        ;;
    esac
done
set -- "${POS[@]}"

if [[ -z ${CONFIGPATH} || -z ${TASK} ]]; then
    echo "Path to LUTE config and Task name are required!"
    usage
    exit
fi

# Assume all other arguments are for SLURM
SLURM_ARGS=$@

# Setup logfile names - $EXPERIMENT and $RUN will be available if ARP submitted
FORMAT_RUN=$(printf "%04d" ${RUN:-0})
LOG_FILE="${TASK}_${EXPERIMENT:-$EXP}_r${FORMAT_RUN}_$(date +'%Y-%m-%d_%H-%M-%S')"
SLURM_ARGS+=" --output=${LOG_FILE}.out"
SLURM_ARGS+=" --error=${LOG_FILE}.err"

export LUTE_SOCKET="/tmp/lute_${RANDOM}.sock"

# By default source the psana environment since most Tasks will use it.
source /sdf/group/lcls/ds/ana/sw/conda1/manage/bin/psconda.sh

export LUTE_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd | sed s/launch_scripts//g )"
EXECUTABLE="${LUTE_PATH}run_task.py"

if [[ ${DEBUG} ]]; then
    echo "Running in debug mode - verbose logging."
    CMD="python -B ${EXECUTABLE} -c ${CONFIGPATH} -t ${TASK}"
else
    echo "Running in standard mode."
    CMD="python -OB ${EXECUTABLE} -c ${CONFIGPATH} -t ${TASK}"
fi

echo "Submitting task ${TASK}"
if [[ $DEBUG ]]; then
    echo "Running ${TASK} with SLURM arguments: ${SLURM_ARGS}"
    echo "Using socket ${LUTE_SOCKET}"
    echo "${CMD}"
fi

sbatch $SLURM_ARGS --wrap "${CMD}"
