#!/usr/bin/env bash
mkdir -p install

source /sdf/group/lcls/ds/ana/sw/conda2/manage/bin/psconda.sh

BASE_DIR="$( readlink -f "$( dirname "${BASH_SOURCE[0]}" )" )"
INSTALL_DIR="${BASE_DIR}/install"

echo "Will build an editable installation of LUTE at ${INSTALL_DIR}"
pip install --editable . --prefix="${INSTALL_DIR}" --no-dependencies
