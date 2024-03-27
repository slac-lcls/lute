"""Base classes for implementing analysis tasks.

Classes:
    Task: Abstract base class from which all analysis tasks are derived.

    BinaryTask: Class to run a third-party executable binary as a `Task`.
"""

__all__ = ["Task", "TaskResult", "TaskStatus", "DescribedAnalysis", "BinaryTask"]
__author__ = "Gabriel Dorlhiac"

import time
from abc import ABC, abstractmethod
from typing import Any, List, Dict, Union, Type, TextIO, Optional
import os
import warnings
import signal
import types

from ..io.models.base import (
    TaskParameters,
    ThirdPartyParameters,
    TemplateConfig,
    AnalysisHeader,
)
from ..execution.ipc import *
from .dataclasses import *

if __debug__:
    warnings.simplefilter("default")
    os.environ["PYTHONWARNINGS"] = "default"

    def lute_warn(
        message: Union[str, Warning],
        category: Type[Warning],
        filename: str,
        lineno: int,
        file: Union[TextIO, None] = None,
        line: Union[str, None] = None,
    ) -> None:
        formatted_warning: str = warnings.formatwarning(
            message, category=category, filename=filename, lineno=lineno, line=line
        )
        msg: Message = Message(contents=formatted_warning)
        communicator: PipeCommunicator = PipeCommunicator()

        communicator.write(msg)

    warnings.showwarning = lute_warn
else:
    warnings.simplefilter("ignore")


class Task(ABC):
    """Abstract base class for analysis tasks.

    Attributes:
        name (str): The name of the Task.
    """

    def __init__(self, *, params: TaskParameters) -> None:
        """Initialize a Task.

        Args:
            params (TaskParameters): Parameters needed to properly configure
                the analysis task. These are NOT related to execution parameters
                (number of cores, etc), except, potentially, in case of binary
                executable sub-classes.
        """
        self.name: str = str(type(self)).split("'")[1].split(".")[-1]
        self._result: TaskResult = TaskResult(
            task_name=self.name,
            task_status=TaskStatus.PENDING,
            summary="PENDING",
            payload="",
        )
        self._task_parameters: TaskParameters = params
        timeout: int = self._task_parameters.lute_config.task_timeout
        signal.setitimer(signal.ITIMER_REAL, timeout)

    def run(self) -> None:
        """Calls the analysis routines and any pre/post task functions.

        This method is part of the public API and should not need to be modified
        in any subclasses.
        """
        self._signal_start()
        self._pre_run()
        self._run()
        self._post_run()
        self._signal_result()

    @abstractmethod
    def _run(self) -> None:
        """Actual analysis to run. Overridden by subclasses.

        Separating the calling API from the implementation allows `run` to
        have pre and post task functionality embedded easily into a single
        function call.
        """
        ...

    def _pre_run(self) -> None:
        """Code to run BEFORE the main analysis takes place.

        This function may, or may not, be employed by subclasses.
        """
        ...

    def _post_run(self) -> None:
        """Code to run AFTER the main analysis takes place.

        This function may, or may not, be employed by subclasses.
        """
        ...

    @property
    def result(self) -> TaskResult:
        """TaskResult: Read-only Task Result information."""
        return self._result

    def __call__(self) -> None:
        self.run()

    def _signal_start(self) -> None:
        """Send the signal that the Task will begin shortly."""
        start_msg: Message = Message(
            contents=self._task_parameters, signal="TASK_STARTED"
        )
        self._result.task_status = TaskStatus.RUNNING
        self._report_to_executor(start_msg)

    def _signal_result(self) -> None:
        """Send the signal that results are ready along with the results."""
        signal: str = "TASK_RESULT"
        results_msg: Message = Message(contents=self.result, signal=signal)
        self._report_to_executor(results_msg)
        time.sleep(0.1)

    def _report_to_executor(self, msg: Message) -> None:
        """Send a message to the Executor.

        Details of `Communicator` choice are hidden from the caller. This
        method may be overriden by subclasses with specialized functionality.

        Args:
            msg (Message): The message object to send.
        """
        communicator: Communicator
        if isinstance(msg.contents, str) or msg.contents is None:
            communicator = PipeCommunicator()
        else:
            communicator = SocketCommunicator()

        communicator.write(msg)

    def clean_up_timeout(self) -> None:
        """Perform any necessary cleanup actions before exit if timing out."""
        ...


class BinaryTask(Task):
    """A `Task` interface to analysis with binary executables."""

    def __init__(self, *, params: TaskParameters) -> None:
        """Initialize a Task.

        Args:
            params (TaskParameters): Parameters needed to properly configure
                the analysis task. `Task`s of this type MUST include the name
                of a binary to run and any arguments which should be passed to
                it (as would be done via command line). The binary is included
                with the parameter `executable`. All other parameter names are
                assumed to be the long/extended names of the flag passed on the
                command line:
                    * `arg_name = 3` is converted to `--arg_name 3`
                Positional arguments can be included with `p_argN` where `N` is
                any integer:
                    * `p_arg1 = 3` is converted to `3`
        """
        super().__init__(params=params)
        self._cmd = self._task_parameters.executable
        self._args_list: List[str] = [self._cmd]
        self._template_context: Dict[str, Any] = {}

    def _add_to_jinja_context(self, param_name: str, value: Any) -> None:
        """Store a parameter as a Jinja template variable.

        Variables are stored in a dictionary which is used to fill in a
        premade Jinja template for a third party configuration file.

        Args:
            param_name (str): Name to store the variable as. This should be
                the name defined in the corresponding pydantic model. This name
                MUST match the name used in the Jinja Template!
            value (Any): The value to store. If possible, large chunks of the
                template should be represented as a single dictionary for
                simplicity; however, any type can be stored as needed.
        """
        context_update: Dict[str, Any] = {param_name: value}
        if __debug__:
            msg: Message = Message(contents=f"ThirdPartyParameters: {context_update}")
            self._report_to_executor(msg)
        self._template_context.update(context_update)

    def _template_to_config_file(self) -> None:
        """Convert a template file into a valid configuration file.

        Uses Jinja to fill in a provided template file with variables supplied
        through the LUTE config file. This facilitates parameter modification
        for third party tasks which use a separate configuration, in addition
        to, or instead of, command-line arguments.
        """
        from jinja2 import Environment, FileSystemLoader, Template

        out_file: str = self._task_parameters.lute_template_cfg.output_path
        template_name: str = self._task_parameters.lute_template_cfg.template_name

        lute_path: Optional[str] = os.getenv("LUTE_PATH")
        template_dir: str
        if lute_path is None:
            warnings.warn(
                "LUTE_PATH is None in Task process! Using relative path for templates!",
                category=UserWarning,
            )
            template_dir: str = "../../config/templates"
        else:
            template_dir = f"{lute_path}/config/templates"
        environment: Environment = Environment(loader=FileSystemLoader(template_dir))
        template: Template = environment.get_template(template_name)

        with open(out_file, "w", encoding="utf-8") as cfg_out:
            cfg_out.write(template.render(self._template_context))

    def _pre_run(self) -> None:
        """Parse the parameters into an appropriate argument list.

        Arguments are identified by a `flag_type` attribute, defined in the
        pydantic model, which indicates how to pass the parameter and its
        argument on the command-line. This method parses flag:value pairs
        into an appropriate list to be used to call the executable.

        Note:
        ThirdPartyParameter objects are returned by custom model validators.
        Objects of this type are assumed to be used for a templated config
        file used by the third party executable for configuration. The parsing
        of these parameters is performed separately by a template file used as
        an input to Jinja. This method solely identifies the necessary objects
        and passes them all along. Refer to the template files and pydantic
        models for more information on how these parameters are defined and
        identified.
        """
        super()._pre_run()
        full_schema: Dict[str, Union[str, Dict[str, Any]]] = (
            self._task_parameters.schema()
        )
        short_flags_use_eq: bool
        long_flags_use_eq: bool
        if hasattr(self._task_parameters.Config, "short_flags_use_eq"):
            short_flags_use_eq: bool = self._task_parameters.Config.short_flags_use_eq
            long_flags_use_eq: bool = self._task_parameters.Config.long_flags_use_eq
        else:
            short_flags_use_eq = False
            long_flags_use_eq = False
        for param, value in self._task_parameters.dict().items():
            # Clunky test with __dict__[param] because compound model-types are
            # converted to `dict`. E.g. type(value) = dict not AnalysisHeader
            if (
                param == "executable"
                or value is None  # Cannot have empty values in argument list for execvp
                or value == ""  # But do want to include, e.g. 0
                or isinstance(self._task_parameters.__dict__[param], TemplateConfig)
                or isinstance(self._task_parameters.__dict__[param], AnalysisHeader)
            ):
                continue
            if isinstance(self._task_parameters.__dict__[param], ThirdPartyParameters):
                # ThirdPartyParameters objects have a single parameter `params`
                self._add_to_jinja_context(param_name=param, value=value.params)
                continue

            param_attributes: Dict[str, Any] = full_schema["properties"][param]
            # Some model params do not match the commnad-line parameter names
            param_repr: str
            if "rename_param" in param_attributes:
                param_repr = param_attributes["rename_param"]
            else:
                param_repr = param
            if "flag_type" in param_attributes:
                flag: str = param_attributes["flag_type"]
                if flag:
                    # "-" or "--" flags
                    if flag == "--" and isinstance(value, bool) and not value:
                        continue
                    constructed_flag: str = f"{flag}{param_repr}"
                    if flag == "--" and isinstance(value, bool) and value:
                        # On/off flag, e.g. something like --verbose: No Arg
                        self._args_list.append(f"{constructed_flag}")
                        continue
                    if (flag == "-" and short_flags_use_eq) or (
                        flag == "--" and long_flags_use_eq
                    ):  # Must come after above check! Otherwise you get --param=True
                        # Flags following --param=value or -param=value
                        constructed_flag = f"{constructed_flag}={value}"
                        self._args_list.append(f"{constructed_flag}")
                        continue
                    self._args_list.append(f"{constructed_flag}")
            else:
                warnings.warn(
                    "Model parameters should be defined using Field(...,flag_type='') in the future.",
                    category=PendingDeprecationWarning,
                )
                if len(param) == 1:  # Single-dash flags
                    if short_flags_use_eq:
                        self._args_list.append(f"-{param_repr}={value}")
                        continue
                    self._args_list.append(f"-{param_repr}")
                elif "p_arg" in param:  # Positional arguments
                    pass
                else:  # Double-dash flags
                    if isinstance(value, bool) and not value:
                        continue
                    if long_flags_use_eq:
                        self._args_list.append(f"--{param_repr}={value}")
                        continue
                    self._args_list.append(f"--{param_repr}")
                    if isinstance(value, bool) and value:
                        continue
            if isinstance(value, str) and " " in value:
                for val in value.split():
                    self._args_list.append(f"{val}")
            else:
                self._args_list.append(f"{value}")
        if (
            hasattr(self._task_parameters, "lute_template_cfg")
            and self._template_context
        ):
            self._template_to_config_file()

    def _run(self) -> None:
        """Execute the new program by replacing the current process."""
        if __debug__:
            time.sleep(0.1)
            msg: Message = Message(contents=self._formatted_command())
            self._report_to_executor(msg)
        os.execvp(file=self._cmd, args=self._args_list)

    def _formatted_command(self) -> str:
        """Returns the command as it would passed on the command-line."""
        formatted_cmd: str = "".join(f"{arg} " for arg in self._args_list)
        return formatted_cmd

    def _signal_start(self) -> None:
        """Override start signal method to switch communication methods."""
        super()._signal_start()
        time.sleep(0.05)
        signal: str = "NO_PICKLE_MODE"
        msg: Message = Message(signal=signal)
        self._report_to_executor(msg)
