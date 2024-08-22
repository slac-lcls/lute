"""Base classes and functions for handling `Task` execution.

Executors run a `Task` as a subprocess and handle all communication with other
services, e.g., the eLog. They accept specific handlers to override default
stream parsing.

Event handlers/hooks are implemented as standalone functions which can be added
to an Executor.


Classes:
    BaseExecutor: Abstract base class from which all Executors are derived.

    Executor: Default Executor implementing all basic functionality and IPC.

    MPIExecutor: Runs exactly as the Executor but submits the Task using MPI.

Exceptions
----------

"""

__all__ = ["BaseExecutor", "Executor", "MPIExecutor"]
__author__ = "Gabriel Dorlhiac"

import sys
import _io
import logging
import subprocess
import time
import os
import signal
from typing import Dict, Callable, List, Optional, Any
from typing_extensions import Self
from abc import ABC, abstractmethod
import warnings
import copy

from lute.execution.ipc import *
from lute.tasks.task import *
from lute.tasks.dataclasses import *
from lute.io.models.base import TaskParameters, TemplateParameters
from lute.io.db import record_analysis_db
from lute.io.elog import post_elog_run_status

if __debug__:
    warnings.simplefilter("default")
    os.environ["PYTHONWARNINGS"] = "default"
    logging.basicConfig(level=logging.DEBUG)
    logging.captureWarnings(True)
else:
    logging.basicConfig(level=logging.INFO)
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"

logger: logging.Logger = logging.getLogger(__name__)


class BaseExecutor(ABC):
    """ABC to manage Task execution and communication with user services.

    When running in a workflow, "tasks" (not the class instances) are submitted
    as `Executors`. The Executor manages environment setup, the actual Task
    submission, and communication regarding Task results and status with third
    party services like the eLog.

    Attributes:

    Methods:
        add_hook(event: str, hook: Callable[[None], None]) -> None: Create a
            new hook to be called each time a specific event occurs.

        add_default_hooks() -> None: Populate the event hooks with the default
            functions.

        update_environment(env: Dict[str, str], update_path: str): Update the
            environment that is passed to the Task subprocess.

        execute_task(): Run the task as a subprocess.
    """

    class Hooks:
        """A container class for the Executor's event hooks.

        There is a corresponding function (hook) for each event/signal. Each
        function takes two parameters - a reference to the Executor (self) and
        a reference to the Message (msg) which includes the corresponding
        signal.
        """

        def no_pickle_mode(self: Self, msg: Message): ...

        def task_started(self: Self, msg: Message): ...

        def task_failed(self: Self, msg: Message): ...

        def task_stopped(self: Self, msg: Message): ...

        def task_done(self: Self, msg: Message): ...

        def task_cancelled(self: Self, msg: Message): ...

        def task_result(self: Self, msg: Message): ...

    def __init__(
        self,
        task_name: str,
        communicators: List[Communicator],
        poll_interval: float = 0.05,
    ) -> None:
        """The Executor will manage the subprocess in which `task_name` is run.

        Args:
            task_name (str): The name of the Task to be submitted. Must match
                the Task's class name exactly. The parameter specification must
                also be in a properly named model to be identified.

            communicators (List[Communicator]): A list of one or more
                communicators which manage information flow to/from the Task.
                Subclasses may have different defaults, and new functionality
                can be introduced by composing Executors with communicators.

            poll_interval (float): Time to wait between reading/writing to the
                managed subprocess. In seconds.
        """
        result: TaskResult = TaskResult(
            task_name=task_name, task_status=TaskStatus.PENDING, summary="", payload=""
        )
        task_parameters: Optional[TaskParameters] = None
        task_env: Dict[str, str] = os.environ.copy()
        self._communicators: List[Communicator] = communicators
        communicator_desc: List[str] = []
        for comm in self._communicators:
            comm.stage_communicator()
            communicator_desc.append(str(comm))

        self._analysis_desc: DescribedAnalysis = DescribedAnalysis(
            task_result=result,
            task_parameters=task_parameters,
            task_env=task_env,
            poll_interval=poll_interval,
            communicator_desc=communicator_desc,
        )

    def add_hook(self, event: str, hook: Callable[[Self, Message], None]) -> None:
        """Add a new hook.

        Each hook is a function called any time the Executor receives a signal
        for a particular event, e.g. Task starts, Task ends, etc. Calling this
        method will remove any hook that currently exists for the event. I.e.
        only one hook can be called per event at a time. Creating hooks for
        events which do not exist is not allowed.

        Args:
            event (str): The event for which the hook will be called.

            hook (Callable[[None], None]) The function to be called during each
                occurrence of the event.
        """
        if event.upper() in LUTE_SIGNALS:
            setattr(self.Hooks, event.lower(), hook)

    @abstractmethod
    def add_default_hooks(self) -> None:
        """Populate the set of default event hooks."""

        ...

    def update_environment(
        self, env: Dict[str, str], update_path: str = "prepend"
    ) -> None:
        """Update the stored set of environment variables.

        These are passed to the subprocess to setup its environment.

        Args:
            env (Dict[str, str]): A dictionary of "VAR":"VALUE" pairs of
                environment variables to be added to the subprocess environment.
                If any variables already exist, the new variables will
                overwrite them (except PATH, see below).

            update_path (str): If PATH is present in the new set of variables,
                this argument determines how the old PATH is dealt with. There
                are three options:
                * "prepend" : The new PATH values are prepended to the old ones.
                * "append" : The new PATH values are appended to the old ones.
                * "overwrite" : The old PATH is overwritten by the new one.
                "prepend" is the default option. If PATH is not present in the
                current environment, the new PATH is used without modification.
        """
        if "PATH" in env:
            sep: str = os.pathsep
            if update_path == "prepend":
                env["PATH"] = (
                    f"{env['PATH']}{sep}{self._analysis_desc.task_env['PATH']}"
                )
            elif update_path == "append":
                env["PATH"] = (
                    f"{self._analysis_desc.task_env['PATH']}{sep}{env['PATH']}"
                )
            elif update_path == "overwrite":
                pass
            else:
                raise ValueError(
                    (
                        f"{update_path} is not a valid option for `update_path`!"
                        " Options are: prepend, append, overwrite."
                    )
                )
        os.environ.update(env)
        self._analysis_desc.task_env.update(env)

    def shell_source(self, env: str) -> None:
        """Source a script.

        Unlike `update_environment` this method sources a new file.

        We prepend a token to each environment variable. This allows the initial
        part of the Task to be run using the appropriate environment.

        The environment variables containing the token will be swapped in using
        their appropriate form prior to the actual execution of Task code.

        Args:
            env (str): Path to the script to source.
        """
        if not os.path.exists(env):
            logger.info(f"Cannot source environment from {env}!")
            return

        script: str = (
            f"set -a\n"
            f'source "{env}" >/dev/null\n'
            f'{sys.executable} -c "import os; print(dict(os.environ))"\n'
        )
        logger.info(f"Sourcing file {env}")
        o, e = subprocess.Popen(
            ["bash", "-c", script], stdout=subprocess.PIPE
        ).communicate()
        tmp_environment: Dict[str, str] = eval(o)
        new_environment: Dict[str, str] = {}
        for key, value in tmp_environment.items():
            new_environment[f"LUTE_TENV_{key}"] = value
            if key == "PATH":
                new_environment[key] = value
        self._analysis_desc.task_env = new_environment

    def _pre_task(self) -> None:
        """Any actions to be performed before task submission.

        This method may or may not be used by subclasses. It may be useful
        for logging etc.
        """
        # This prevents the Executors in managed_tasks.py from all acquiring
        # resources like sockets.
        for communicator in self._communicators:
            communicator.delayed_setup()
            # Not great, but experience shows we need a bit of time to setup
            # network.
            time.sleep(0.1)
        # Propagate any env vars setup by Communicators - only update LUTE_ vars
        tmp: Dict[str, str] = {
            key: os.environ[key] for key in os.environ if "LUTE_" in key
        }
        self._analysis_desc.task_env.update(tmp)

    def _submit_task(self, cmd: str) -> subprocess.Popen:
        proc: subprocess.Popen = subprocess.Popen(
            cmd.split(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=self._analysis_desc.task_env,
        )
        os.set_blocking(proc.stdout.fileno(), False)
        os.set_blocking(proc.stderr.fileno(), False)
        return proc

    @abstractmethod
    def _task_loop(self, proc: subprocess.Popen) -> None:
        """Actions to perform while the Task is running.

        This function is run in the body of a loop until the Task signals
        that its finished.
        """
        ...

    @abstractmethod
    def _finalize_task(self, proc: subprocess.Popen) -> None:
        """Any actions to be performed after the Task has ended.

        Examples include a final clearing of the pipes, retrieving results,
        reporting to third party services, etc.
        """
        ...

    def _submit_cmd(self, executable_path: str, params: str) -> str:
        """Return a formatted command for launching Task subprocess.

        May be overridden by subclasses.

        The default submission uses the Executor environment. This ensures that
        all necessary packages (e.g. Pydantic for validation) are available to
        the startup scripts. If a Task has a different environment it will be
        swapped prior to execution.

        Args:
            executable_path (str): Path to the LUTE subprocess script.

            params (str): String of formatted command-line arguments.

        Returns:
            cmd (str): Appropriately formatted command for this Executor.
        """
        cmd: str = ""
        if __debug__:
            cmd = f"{sys.executable} -B {executable_path} {params}"
        else:
            cmd = f"{sys.executable} -OB {executable_path} {params}"

        return cmd

    def execute_task(self) -> None:
        """Run the requested Task as a subprocess."""
        self._pre_task()
        lute_path: Optional[str] = os.getenv("LUTE_PATH")
        if lute_path is None:
            logger.debug("Absolute path to subprocess_task.py not found.")
            lute_path = os.path.abspath(f"{os.path.dirname(__file__)}/../..")
            self.update_environment({"LUTE_PATH": lute_path})
        executable_path: str = f"{lute_path}/subprocess_task.py"
        config_path: str = self._analysis_desc.task_env["LUTE_CONFIGPATH"]
        params: str = f"-c {config_path} -t {self._analysis_desc.task_result.task_name}"

        cmd: str = self._submit_cmd(executable_path, params)
        proc: subprocess.Popen = self._submit_task(cmd)

        while self._task_is_running(proc):
            self._task_loop(proc)
            time.sleep(self._analysis_desc.poll_interval)

        os.set_blocking(proc.stdout.fileno(), True)
        os.set_blocking(proc.stderr.fileno(), True)

        self._finalize_task(proc)
        proc.stdout.close()
        proc.stderr.close()
        proc.wait()
        if ret := proc.returncode:
            logger.info(f"Task failed with return code: {ret}")
            self._analysis_desc.task_result.task_status = TaskStatus.FAILED
            self.Hooks.task_failed(self, msg=Message())
        elif self._analysis_desc.task_result.task_status == TaskStatus.RUNNING:
            # Ret code is 0, no exception was thrown, task forgot to set status
            self._analysis_desc.task_result.task_status = TaskStatus.COMPLETED
            logger.debug(f"Task did not change from RUNNING status. Assume COMPLETED.")
            self.Hooks.task_done(self, msg=Message())
        self._store_configuration()
        for comm in self._communicators:
            comm.clear_communicator()

        if self._analysis_desc.task_result.task_status == TaskStatus.FAILED:
            logger.info("Exiting after Task failure. Result recorded.")
            sys.exit(-1)

        self.process_results()

    def _store_configuration(self) -> None:
        """Store configuration and results in the LUTE database."""
        record_analysis_db(copy.deepcopy(self._analysis_desc))

    def _task_is_running(self, proc: subprocess.Popen) -> bool:
        """Whether a subprocess is running.

        Args:
            proc (subprocess.Popen): The subprocess to determine the run status
                of.

        Returns:
            bool: Is the subprocess task running.
        """
        # Add additional conditions - don't want to exit main loop
        # if only stopped
        task_status: TaskStatus = self._analysis_desc.task_result.task_status
        is_running: bool = task_status != TaskStatus.COMPLETED
        is_running &= task_status != TaskStatus.CANCELLED
        is_running &= task_status != TaskStatus.TIMEDOUT
        return proc.poll() is None and is_running

    def _stop(self, proc: subprocess.Popen) -> None:
        """Stop the Task subprocess."""
        os.kill(proc.pid, signal.SIGTSTP)
        self._analysis_desc.task_result.task_status = TaskStatus.STOPPED

    def _continue(self, proc: subprocess.Popen) -> None:
        """Resume a stopped Task subprocess."""
        os.kill(proc.pid, signal.SIGCONT)
        self._analysis_desc.task_result.task_status = TaskStatus.RUNNING

    def _set_result_from_parameters(self) -> None:
        """Use TaskParameters object to set TaskResult fields.

        A result may be defined in terms of specific parameters. This is most
        useful for ThirdPartyTasks which would not otherwise have an easy way of
        reporting what the TaskResult is. There are two options for specifying
        results from parameters:
            1. A single parameter (Field) of the model has an attribute
               `is_result`. This is a bool indicating that this parameter points
               to a result. E.g. a parameter `output` may set `is_result=True`.
            2. The `TaskParameters.Config` has a `result_from_params` attribute.
               This is an appropriate option if the result is determinable for
               the Task, but it is not easily defined by a single parameter. The
               TaskParameters.Config.result_from_param can be set by a custom
               validator, e.g. to combine the values of multiple parameters into
               a single result. E.g. an `out_dir` and `out_file` parameter used
               together specify the result. Currently only string specifiers are
               supported.

        A TaskParameters object specifies that it contains information about the
        result by setting a single config option:
                        TaskParameters.Config.set_result=True
        In general, this method should only be called when the above condition is
        met, however, there are minimal checks in it as well.
        """
        # This method shouldn't be called unless appropriate
        # But we will add extra guards here
        if self._analysis_desc.task_parameters is None:
            logger.debug(
                "Cannot set result from TaskParameters. TaskParameters is None!"
            )
            return
        if (
            not hasattr(self._analysis_desc.task_parameters.Config, "set_result")
            or not self._analysis_desc.task_parameters.Config.set_result
        ):
            logger.debug(
                "Cannot set result from TaskParameters. `set_result` not specified!"
            )
            return

        # First try to set from result_from_params (faster)
        if self._analysis_desc.task_parameters.Config.result_from_params is not None:
            result_from_params: str = (
                self._analysis_desc.task_parameters.Config.result_from_params
            )
            logger.info(f"TaskResult specified as {result_from_params}.")
            self._analysis_desc.task_result.payload = result_from_params
        else:
            # Iterate parameters to find the one that is the result
            schema: Dict[str, Any] = self._analysis_desc.task_parameters.schema()
            for param, value in self._analysis_desc.task_parameters.dict().items():
                if isinstance(value, TemplateParameters):
                    # Extract TemplateParameters if needed
                    value = value.params
                param_attrs: Dict[str, Any] = schema["properties"][param]
                if "is_result" in param_attrs:
                    is_result: bool = param_attrs["is_result"]
                    if isinstance(is_result, bool) and is_result:
                        logger.info(f"TaskResult specified as {value}.")
                        self._analysis_desc.task_result.payload = value
                    else:
                        logger.debug(
                            (
                                f"{param} specified as result! But specifier is of "
                                f"wrong type: {type(is_result)}!"
                            )
                        )
                    break  # We should only have 1 result-like parameter!

        # If we get this far and haven't changed the payload we should complain
        if self._analysis_desc.task_result.payload == "":
            task_name: str = self._analysis_desc.task_result.task_name
            logger.debug(
                (
                    f"{task_name} specified result be set from {task_name}Parameters,"
                    " but no result provided! Check model definition!"
                )
            )
        # Now check for impl_schemas and pass to result.impl_schemas
        # Currently unused
        impl_schemas: Optional[str] = (
            self._analysis_desc.task_parameters.Config.impl_schemas
        )
        self._analysis_desc.task_result.impl_schemas = impl_schemas
        # If we set_result but didn't get schema information we should complain
        if self._analysis_desc.task_result.impl_schemas is None:
            task_name: str = self._analysis_desc.task_result.task_name
            logger.debug(
                (
                    f"{task_name} specified result be set from {task_name}Parameters,"
                    " but no schema provided! Check model definition!"
                )
            )

    def process_results(self) -> None:
        """Perform any necessary steps to process TaskResults object.

        Processing will depend on subclass. Examples of steps include, moving
        files, converting file formats, compiling plots/figures into an HTML
        file, etc.
        """
        self._process_results()

    @abstractmethod
    def _process_results(self) -> None: ...


class Executor(BaseExecutor):
    """Basic implementation of an Executor which manages simple IPC with Task.

    Attributes:

    Methods:
        add_hook(event: str, hook: Callable[[None], None]) -> None: Create a
            new hook to be called each time a specific event occurs.

        add_default_hooks() -> None: Populate the event hooks with the default
            functions.

        update_environment(env: Dict[str, str], update_path: str): Update the
            environment that is passed to the Task subprocess.

        execute_task(): Run the task as a subprocess.
    """

    def __init__(
        self,
        task_name: str,
        communicators: List[Communicator] = [
            PipeCommunicator(Party.EXECUTOR),
            SocketCommunicator(Party.EXECUTOR),
        ],
        poll_interval: float = 0.05,
    ) -> None:
        super().__init__(
            task_name=task_name,
            communicators=communicators,
            poll_interval=poll_interval,
        )
        self.add_default_hooks()

    def add_default_hooks(self) -> None:
        """Populate the set of default event hooks."""

        def no_pickle_mode(self: Executor, msg: Message):
            for idx, communicator in enumerate(self._communicators):
                if isinstance(communicator, PipeCommunicator):
                    self._communicators[idx] = PipeCommunicator(
                        Party.EXECUTOR, use_pickle=False
                    )

        self.add_hook("no_pickle_mode", no_pickle_mode)

        def task_started(self: Executor, msg: Message):
            if isinstance(msg.contents, TaskParameters):
                self._analysis_desc.task_parameters = msg.contents
                # Maybe just run this no matter what? Rely on the other guards?
                # Perhaps just check if ThirdPartyParameters?
                # if isinstance(self._analysis_desc.task_parameters, ThirdPartyParameters):
                if hasattr(self._analysis_desc.task_parameters.Config, "set_result"):
                    # Third party Tasks may mark a parameter as the result
                    # If so, setup the result now.
                    self._set_result_from_parameters()
            logger.info(
                f"Executor: {self._analysis_desc.task_result.task_name} started"
            )
            self._analysis_desc.task_result.task_status = TaskStatus.RUNNING
            elog_data: Dict[str, str] = {
                f"{self._analysis_desc.task_result.task_name} status": "RUNNING",
            }
            post_elog_run_status(elog_data)

        self.add_hook("task_started", task_started)

        def task_failed(self: Executor, msg: Message):
            elog_data: Dict[str, str] = {
                f"{self._analysis_desc.task_result.task_name} status": "FAILED",
            }
            post_elog_run_status(elog_data)

        self.add_hook("task_failed", task_failed)

        def task_stopped(self: Executor, msg: Message):
            elog_data: Dict[str, str] = {
                f"{self._analysis_desc.task_result.task_name} status": "STOPPED",
            }
            post_elog_run_status(elog_data)

        self.add_hook("task_stopped", task_stopped)

        def task_done(self: Executor, msg: Message):
            elog_data: Dict[str, str] = {
                f"{self._analysis_desc.task_result.task_name} status": "COMPLETED",
            }
            post_elog_run_status(elog_data)

        self.add_hook("task_done", task_done)

        def task_cancelled(self: Executor, msg: Message):
            elog_data: Dict[str, str] = {
                f"{self._analysis_desc.task_result.task_name} status": "CANCELLED",
            }
            post_elog_run_status(elog_data)

        self.add_hook("task_cancelled", task_cancelled)

        def task_result(self: Executor, msg: Message):
            if isinstance(msg.contents, TaskResult):
                self._analysis_desc.task_result = msg.contents
                logger.info(self._analysis_desc.task_result.summary)
                logger.info(self._analysis_desc.task_result.task_status)
            elog_data: Dict[str, str] = {
                f"{self._analysis_desc.task_result.task_name} status": "COMPLETED",
            }
            post_elog_run_status(elog_data)

        self.add_hook("task_result", task_result)

    def _task_loop(self, proc: subprocess.Popen) -> None:
        """Actions to perform while the Task is running.

        This function is run in the body of a loop until the Task signals
        that its finished.
        """
        for communicator in self._communicators:
            while True:
                msg: Message = communicator.read(proc)
                if msg.signal is not None and msg.signal.upper() in LUTE_SIGNALS:
                    hook: Callable[[Executor, Message], None] = getattr(
                        self.Hooks, msg.signal.lower()
                    )
                    hook(self, msg)
                if msg.contents is not None:
                    if isinstance(msg.contents, str) and msg.contents != "":
                        logger.info(msg.contents)
                    elif not isinstance(msg.contents, str):
                        logger.info(msg.contents)
                if not communicator.has_messages:
                    break

    def _finalize_task(self, proc: subprocess.Popen) -> None:
        """Any actions to be performed after the Task has ended.

        Examples include a final clearing of the pipes, retrieving results,
        reporting to third party services, etc.
        """
        self._task_loop(proc)  # Perform a final read.

    def _process_results(self) -> None:
        """Performs result processing.

        Actions include:
        - For `ElogSummaryPlots`, will save the summary plot to the appropriate
            directory for display in the eLog.
        """
        task_result: TaskResult = self._analysis_desc.task_result
        self._process_result_payload(task_result.payload)
        self._process_result_summary(task_result.summary)

    def _process_result_payload(self, payload: Any) -> None:
        if self._analysis_desc.task_parameters is None:
            logger.debug("Please run Task before using this method!")
            return
        if isinstance(payload, ElogSummaryPlots):
            # ElogSummaryPlots has figures and a display name
            # display name also serves as a path.
            expmt: str = self._analysis_desc.task_parameters.lute_config.experiment
            base_path: str = f"/sdf/data/lcls/ds/{expmt[:3]}/{expmt}/stats/summary"
            full_path: str = f"{base_path}/{payload.display_name}"
            if not os.path.isdir(full_path):
                os.makedirs(full_path)

            # Preferred plots are pn.Tabs objects which save directly as html
            # Only supported plot type that has "save" method - do not want to
            # import plot modules here to do type checks.
            if hasattr(payload.figures, "save"):
                payload.figures.save(f"{full_path}/report.html")
            else:
                ...
        elif isinstance(payload, str):
            # May be a path to a file...
            schemas: Optional[str] = self._analysis_desc.task_result.impl_schemas
            # Should also check `impl_schemas` to determine what to do with path

    def _process_result_summary(self, summary: str) -> None: ...


class MPIExecutor(Executor):
    """Runs first-party Tasks that require MPI.

    This Executor is otherwise identical to the standard Executor, except it
    uses `mpirun` for `Task` submission. Currently this Executor assumes a job
    has been submitted using SLURM as a first step. It will determine the number
    of MPI ranks based on the resources requested. As a fallback, it will try
    to determine the number of local cores available for cases where a job has
    not been submitted via SLURM. On S3DF, the second determination mechanism
    should accurately match the environment variable provided by SLURM indicating
    resources allocated.

    This Executor will submit the Task to run with a number of processes equal
    to the total number of cores available minus 1. A single core is reserved
    for the Executor itself. Note that currently this means that you must submit
    on 3 cores or more, since MPI requires a minimum of 2 ranks, and the number
    of ranks is determined from the cores dedicated to Task execution.

    Methods:
        _submit_cmd: Run the task as a subprocess using `mpirun`.
    """

    def _submit_cmd(self, executable_path: str, params: str) -> str:
        """Override submission command to use `mpirun`

        Args:
            executable_path (str): Path to the LUTE subprocess script.

            params (str): String of formatted command-line arguments.

        Returns:
            cmd (str): Appropriately formatted command for this Executor.
        """
        py_cmd: str = ""
        nprocs: int = max(
            int(os.environ.get("SLURM_NPROCS", len(os.sched_getaffinity(0)))) - 1, 1
        )
        mpi_cmd: str = f"mpirun -np {nprocs}"
        if __debug__:
            py_cmd = f"python -B -u -m mpi4py.run {executable_path} {params}"
        else:
            py_cmd = f"python -OB -u -m mpi4py.run {executable_path} {params}"

        cmd: str = f"{mpi_cmd} {py_cmd}"
        return cmd
