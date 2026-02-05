#!/usr/bin/env bash
set -e

## This build script will create an isolated build environment, and cache it for reuse
## By caching, we can use `meson` (fast) to actually build C++ code.
## `pip` gets used only at the very end to install the entry-points.

# Bunch of functions to pretty print...
center_text() {
    local TEXT="$1"
    local WIDTH="$2"
    local LEN=${#TEXT}
    if (( LEN >= WIDTH )); then
        echo "===== ${TEXT} ====="
    else
        # Calculate spaces on each side
        local PAD_SIZE=$(( (WIDTH - LEN) / 2 + 5))
        NEW_LINE=$(printf "%*s%s%*s\n" "${PAD_SIZE}" "" "${TEXT}" "$((WIDTH - LEN - PAD_SIZE))" " ")
        echo "===== ${NEW_LINE} ====="
    fi
}

## Create a banner that looks like:
## ===============================
## =====         Line 1      =====
## =====         Line 2      =====
## =====         Line 3      =====
## ===============================
## For the LINES that are provided
print_banner() {
    local LINES=("$@")
    local MAXLEN=0
    local NEWLINES=()
    for LINE in "${LINES[@]}"; do
        NEWLINES+=("$LINE")
        (( ${#LINE} > MAXLEN )) && MAXLEN=${#LINE}
    done

    local BORDERLEN=$((MAXLEN+12))
    if (( BORDERLEN < 80)); then
        BORDERLEN=80
        MAXLEN=$((BORDERLEN - 12))
    fi
    # Print top border
    printf '%*s\n' "${BORDERLEN}" '' | tr ' ' '='

    # Print centered lines
    for LINE in "${NEWLINES[@]}"; do
        center_text "${LINE}" "${MAXLEN}"
    done

    printf '%*s\n' "${BORDERLEN}" '' | tr ' ' '='
}


# Determine build directories, install directory, and where to put the build env
BASE_DIR="$( readlink -f "$( dirname "${BASH_SOURCE[0]}" )" )"
BUILD_DIR="${BASE_DIR}/_build"
INSTALL_DIR="${BASE_DIR}/install"

# Virtual environment needs to be outside the source tree
# Otherwise you get some path errors with meson
# At least I couldn't figure out any other way to do it...
BUILD_ENV="${HOME}/.cache/lute_build_env_$(echo ${BASE_DIR} | md5sum | cut -d' ' -f1)"

mkdir -p "${INSTALL_DIR}"
mkdir -p "${BUILD_DIR}"
mkdir -p "${HOME}/.cache"

# On S3DF get a standard Python3 - otherwise you're on your own
if [[ $HOSTNAME =~ "sdf" ]]; then
    LINES=("Sourcing the Psana1 environment (for Python3)")
    print_banner "${LINES[@]}"
    source /sdf/group/lcls/ds/ana/sw/conda1/manage/bin/psconda.sh
fi

# Save host/conda env Python for later
HOST_PYTHON=$(which python3)

# Create a build environment if it doesn't yet exist
if [ ! -d "${BUILD_ENV}" ]; then
    LINES=("Creating isolated build environment in ${BUILD_ENV}...")
    print_banner "${LINES[@]}"
    python3 -m venv "${BUILD_ENV}"
    # Source before installing dependencies
    source "${BUILD_ENV}/bin/activate"
    pip install --upgrade pip
    # Install the relevant build dependencies and nothing else
    pip install "meson>=1.10.1" "meson-python" "ninja" "numpy" "pybind11" "setuptools"
else
    # If it already exists (you've run build.sh before) just activate it
    LINES=("Activating build environment at ${BUILD_ENV}")
    print_banner "${LINES[@]}"
    source "${BUILD_ENV}/bin/activate"
fi

LINES=(
    "Will build and installation of LUTE at ${INSTALL_DIR}"
    "(Build cache available at: ${BUILD_DIR})"
    "(Build environment available at: ${BUILD_ENV})"
)

print_banner "${LINES[@]}"

# Run meson configure/setup if it hasn't be done yet.
if [ ! -d "${BUILD_DIR}" ]; then
    LINES=("Running meson setup for build configuration")
    print_banner "${LINES[@]}"
    meson setup "${BUILD_DIR}" --prefix="${INSTALL_DIR}" -Dbuildtype=release
else
    LINES=("Running meson setup configuration")
    print_banner "${LINES[@]}"
    # Reconfigure in case prefix or options changed, but keep cache
    meson setup "${BUILD_DIR}" --reconfigure --prefix="${INSTALL_DIR}"
fi

# Build... This is mostly for `maestro` and C/C++ extensions
LINES=("Compiling...")
print_banner "${LINES[@]}"
meson compile -C "${BUILD_DIR}"

LINES=("Installing files in ${INSTALL_DIR}...")
print_banner "${LINES[@]}"
meson install -C "${BUILD_DIR}"

# Run `pip` at the end - this gets the entrypoints defined in `pyproject.toml`
# It can be pointed at the build directory to prevent `pip` from trying to rebuild
# the rest of the stuff from scratch.

# We will also use the underlying Python (e.g. from psconda.sh)
# This way, the build env can be kept small, and it can be deleted as well.
# Otherwise, the Python scripts would end up pointing to the Python from that env.
LINES=("Creating Python entry points")
print_banner "${LINES[@]}"

BUILD_VENV_SITE_PACKAGES=(${BUILD_ENV}/lib/python*/site-packages)
PYTHONPATH="${BUILD_VENV_SITE_PACKAGES}:${PYTHONPATH}" \
PATH="${BUILD_ENV}/bin:${PATH}" \
${HOST_PYTHON} -m pip install . \
    --prefix="${INSTALL_DIR}" \
    --no-dependencies \
    --no-build-isolation \
    --config-settings=build-dir="${BUILD_DIR}"
