#!/usr/bin/env bash
mkdir -p install

if [[ $HOSTNAME =~ "sdf" ]]; then
    source /sdf/group/lcls/ds/ana/sw/conda1/manage/bin/psconda.sh
fi

BASE_DIR="$( readlink -f "$( dirname "${BASH_SOURCE[0]}" )" )"
INSTALL_DIR="${BASE_DIR}/install"

echo "Will build an installation of LUTE at ${INSTALL_DIR}"
pip install . --prefix="${INSTALL_DIR}" --no-dependencies
