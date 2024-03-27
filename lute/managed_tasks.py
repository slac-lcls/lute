from typing import Dict

from .execution.executor import *
from .io.config import *

# Tests
#######
Tester: Executor = Executor("Test")
BinaryTester: Executor = Executor("TestBinary")
BinaryErrTester = Executor("TestBinaryErr")
SocketTester: Executor = Executor("TestSocket")
WriteTester: Executor = Executor("TestWriteOutput")
ReadTester: Executor = Executor("TestReadOutput")


# SmallData-related
###################
SmallDataProducer: Executor = Executor("SubmitSMD")

# SFX
#####
CrystFELIndexer: Executor = Executor("IndexCrystFEL")
CrystFELIndexer.update_environment(
    {
        "PATH": (
            "/sdf/group/lcls/ds/tools/XDS-INTEL64_Linux_x86_64:"
            "/sdf/group/lcls/ds/tools:"
            "/sdf/group/lcls/ds/tools/crystfel/0.10.2/bin"
        )
    }
)
PartialatorMerger: Executor = Executor("MergePartialator")
HKLComparer: Executor = Executor("CompareHKL")  # For figures of merit
HKLManipulator: Executor = Executor("ManipulateHKL")  # For hkl->mtz, but can do more
DimpleSolver: Executor = Executor("DimpleSolve")
DimpleSolver.shell_source("/sdf/group/lcls/ds/tools/ccp4-8.0/bin/ccp4.setup-sh")
PeakFinderPyAlgos: MPIExecutor = MPIExecutor("FindPeaksPyAlgos")
SHELXCRunner: Executor = Executor("RunSHELXC")
SHELXCRunner.shell_source("/sdf/group/lcls/ds/tools/ccp4-8.0/bin/ccp4.setup-sh")
PeakFinderPsocake: Executor = Executor("FindPeaksPsocake")
