#!/bin/bash

#!/bin/bash
usage()
{
    cat << EOF
run_functional.py:
    Run a series of functional tests for LUTE.
    Options:
        -a|--admin
          Use an administrator account for Airflow authentication. Default: False
        --git_pr_id
          Checkout a PR branch to run LUTE against based on GitHub ID. (Optional)
        --git_tag
          Checkout a specific tag (e.g. release) of LUTE. (Optional)
        -h|--help
          Display this message.
        --run_dir
          Directory to install LUTE in, and setup the output folder. (Required)
        --test
          Use the test Airflow instance.
EOF
}

while [[ $# -gt 0 ]]
do
    flag="$1"

    case $flag in
        -h|--help)
            usage
            exit
            ;;
        -r|--run_dir)
            POS+=("$1")
            POS+=("$2")
            RUN_DIR="$2"
            shift
            shift
            ;;
        *)
            POS+=("$1")
            shift
            ;;
    esac
done
set -- "${POS[@]}"

if [[ -z ${RUN_DIR} ]]; then
    echo "Must provide a run directory!"
    usage
    exit
fi

# Bodge Kerberos credentials
# These duplicates are removed later by the workflow process
KERB_CACHE_PATH=$(klist -l | awk -F"FILE:" '{printf (NF>1)? $NF : ""}')
if [[ ! -d $HOME/.tmp_cache ]]; then
    mkdir $HOME/.tmp_cache
fi
cp $KERB_CACHE_PATH $HOME/.tmp_cache/kerbcache
export KRB5CCNAME="FILE:${HOME}/.tmp_cache/kerbcache"

TEST_SCRIPT_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
source /sdf/group/lcls/ds/ana/sw/conda1/manage/bin/psconda.sh
echo "Sourced latest psconda.sh"
CMD="python -B ${TEST_SCRIPT_PATH}/run_functional.py ${@}"
echo $CMD
SLURM_ARGS="--partition=milano --account=lcls:data --ntasks=1"
echo "Running ${CMD} with ${SLURM_ARGS}"
sbatch $SLURM_ARGS --wrap "${CMD}"
export KRB5CCNAME="FILE:${KERB_CACHE_PATH}"
