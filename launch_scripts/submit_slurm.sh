#!/bin/bash
usage()
{
    cat << EOF
$(basename "$0"):
    Submit a LUTE managed Task using SLURM on S3DF.
    Options:
        -c|--config
          ABSOLUTE path to the LUTE configuration YAML. Must be absolute.
        --debug
          Whether to run in debug mode (verbose printing).
        -h|--help
          Display this message.
        -t|--taskname
          Name of the LUTE managed Task to run.

    NOTE: This script does not parse SLURM arguments, but a number of them are
          mandatory. All additional arguments are transparently passed to SLURM.
          You will need to provide at least the queue and account using, e.g.:
                  --partition=milano --account=lcls:<experiment>

    Additional options:
    You can also optionally provide experiment and run number. Do this only when
    NOT running from the eLog using the ARP. These options set environment
    variables EXPERIMENT and RUN, which can alternatively be set directly.
        -e|--experiment
          Experiment name.
        -r|--run
          Run number.
        -K|--KERB
          Kerberos cache file variable. This should NOT be set manually!
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
    -e|--experiment)
        EXP_PARAM="$2"
        shift
        shift
        ;;
    -r|--run)
        RUN_PARAM="$2"
        shift
        shift
        ;;
    -K|--KERB)
        KERB_CACHE="$2"
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

if [[ -v KERB_CACHE ]]; then
    export KRB5CCNAME=$KERB_CACHE
fi

if [[ -v EXP_PARAM ]]; then
    EXPERIMENT=$EXP_PARAM
fi
export EXPERIMENT
# Setup logfile names - $EXPERIMENT and $RUN_NUM will be available if ARP submitted
# RUN_NUM is actually in format RUN_DATETIME
RUN_TIME_ARR=(${RUN_NUM//_/ })
RUN="${RUN_TIME_ARR[0]}"
if [[ -v RUN_PARAM ]]; then
    RUN_NUM=$RUN_PARAM
    RUN=$RUN_NUM
fi
export RUN_NUM=$RUN
FORMAT_RUN=$(printf "%04d" ${RUN:-0})
LOG_FILE="${TASK}_${EXPERIMENT:-$EXP}_r${FORMAT_RUN}_$(date +'%Y-%m-%d_%H-%M-%S')"
SLURM_ARGS+=" --output=${LOG_FILE}_%J.out"
SLURM_ARGS+=" --error=${LOG_FILE}_%J.out"

# If LUTE_USE_TCP is unset use TCP
if [[ -z ${LUTE_USE_TCP} || ${LUTE_USE_TCP} != 0 ]]; then
    echo "Using TCP"
    export LUTE_USE_TCP=1
else
    echo "Using Unix sockets"
    unset LUTE_USE_TCP
    export LUTE_SOCKET="/tmp/lute_${RANDOM}.sock"
fi

# By default source the psana environment since most Tasks will use it.
source /sdf/group/lcls/ds/ana/sw/conda1/manage/bin/psconda.sh

# activate LUTE
EXECUTABLE="run_task.py"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)"
# LUTE_PATH is really only used for clones not installations
export LUTE_PATH="$(echo $SCRIPT_DIR | sed s/launch_scripts//g | sed s/bin//g)"

if [[ $SCRIPT_DIR == *"launch_scripts"* ]]; then
    # Running from a clone: /path/to/lute/launch_scripts
    EXECUTABLE="${LUTE_PATH}run_task.py"
else
    # Running from an installation: /path/to/lute/bin
    activate_installation
    EXECUTABLE=$(which run_task.py)
    # ideally would use run_task instead of python run_task.py but need
    # a non-hacky way to provide python arguments to shebang (e.g. -O)
fi

DEBUG_PRINT="Running in debug mode -verbose logging."
NON_DEBUG_PRINT="Running in standard mode."

if [[ ${DEBUG} ]]; then
    echo $DEBUG_PRINT
    CMD="python -B ${EXECUTABLE} -c ${CONFIGPATH} -t ${TASK}"
else
    echo $NON_DEBUG_PRINT
    CMD="python -OB ${EXECUTABLE} -c ${CONFIGPATH} -t ${TASK}"
fi

echo "Submitting task ${TASK}"
if [[ $DEBUG ]]; then
    echo "Running ${TASK} with SLURM arguments: ${SLURM_ARGS}"
    echo "Using socket ${LUTE_SOCKET}"
    echo "${CMD}"
fi

sbatch $SLURM_ARGS --wrap "${CMD}"
