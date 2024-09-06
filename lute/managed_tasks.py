"""LUTE Managed Tasks.

Executor-managed Tasks with specific environment specifications are defined
here.
"""

from lute.execution.executor import *
from lute.io.config import *
from lute.tasks.tasklets import compare_hkl_fom_summary

# Tests
#######
Tester: Executor = Executor("Test")
"""Runs a basic test of a first-party Task."""

BinaryTester: Executor = Executor("TestBinary")
"""Runs a basic test of a multi-threaded third-party Task."""

BinaryErrTester = Executor("TestBinaryErr")
"""Runs a test of a third-party task that fails."""

SocketTester: Executor = Executor("TestSocket")
"""Runs a test of socket-based communication."""

WriteTester: Executor = Executor("TestWriteOutput")
"""Runs a test to confirm database writing."""

ReadTester: Executor = Executor("TestReadOutput")
"""Runs a test to confirm database reading."""

MultiNodeCommunicationTester: MPIExecutor = MPIExecutor("TestMultiNodeCommunication")
"""Runs a test to confirm communication works between multiple nodes."""

# SmallData-related
###################
SmallDataProducer: Executor = Executor("SubmitSMD")
"""Runs the production of a smalldata HDF5 file."""

SmallDataXSSAnalyzer: MPIExecutor = MPIExecutor("AnalyzeSmallDataXSS")
"""Process scattering results from a Small Data HDF5 file."""

SmallDataXASAnalyzer: MPIExecutor = MPIExecutor("AnalyzeSmallDataXAS")
"""Process XAS results from a Small Data HDF5 file."""

SmallDataXESAnalyzer: MPIExecutor = MPIExecutor("AnalyzeSmallDataXES")
"""Process XES results from a Small Data HDF5 file."""

# SFX
#####
CCTBXIndexer: Executor = Executor("IndexCCTBXXFEL")
"""Runs crystallographic indexing using cctbx.xfel."""
CCTBXIndexer.shell_source("/sdf/group/lcls/ds/tools/cctbx/setup.sh")

CrystFELIndexer: Executor = Executor("IndexCrystFEL")
"""Runs crystallographic indexing using CrystFEL."""
CrystFELIndexer.update_environment(
    {
        "PATH": (
            "/sdf/group/lcls/ds/tools/XDS-INTEL64_Linux_x86_64:"
            "/sdf/group/lcls/ds/tools:"
            "/sdf/group/lcls/ds/tools/crystfel/0.10.2/bin"
        )
    }
)

StreamFileConcatenator: Executor = Executor("ConcatenateStreamFiles")
"""Concatenate output stream files."""

CCTBXMerger: Executor = Executor("MergeCCTBXXFEL")
"""Runs crystallographic merging using cctbx.xfel."""
CCTBXMerger.shell_source("/sdf/group/lcls/ds/tools/cctbx/setup.sh")

PartialatorMerger: Executor = Executor("MergePartialator")
"""Runs crystallographic merging using CrystFEL's partialator."""

HKLComparer: Executor = Executor("CompareHKL")  # For figures of merit
"""Runs analysis on merge results for statistics/figures of merit.."""
HKLComparer.add_tasklet(
    compare_hkl_fom_summary,
    ["{{ shell_file }}", "test/rsplit"],
    when="after",
    set_result=False,
    set_summary=True,
)

HKLManipulator: Executor = Executor("ManipulateHKL")  # For hkl->mtz, but can do more
"""Performs format conversions (among other things) of merge results."""

DimpleSolver: Executor = Executor("DimpleSolve")
"""Solves a crystallographic structure using molecular replacement."""
DimpleSolver.shell_source("/sdf/group/lcls/ds/tools/ccp4-8.0/bin/ccp4.setup-sh")

PeakFinderPyAlgos: MPIExecutor = MPIExecutor("FindPeaksPyAlgos")
"""Performs Bragg peak finding using the PyAlgos algorithm."""

SHELXCRunner: Executor = Executor("RunSHELXC")
"""Runs CCP4 SHELXC - needed for crystallographic phasing."""
SHELXCRunner.shell_source("/sdf/group/lcls/ds/tools/ccp4-8.0/bin/ccp4.setup-sh")

PeakFinderPsocake: Executor = Executor("FindPeaksPsocake")
"""Performs Bragg peak finding using psocake - *DEPRECATED*."""
