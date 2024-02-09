from .io.config import *
from .execution.executor import *

# Tests
#######
Tester = Executor("Test")
BinaryTester = Executor("TestBinary")
SocketTester = Executor("TestSocket")

# SmallData-related
###################
SmallDataProducer = Executor("SubmitSMD")
