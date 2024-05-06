"""Base classes for describing Task parameters.

Classes:
    AnalysisHeader(BaseModel): Model holding shared configuration across Tasks.
        E.g. experiment name, run number and working directory.

    TaskParameters(BaseSettings): Base class for Task parameters. Subclasses
        specify a model of parameters and their types for validation.

    ThirdPartyParameters(TaskParameters): Base class for Third-party, binary
        executable Tasks.

    TemplateParameters: Dataclass to represent parameters of binary
        (third-party) Tasks which are used for additional config files.

    TemplateConfig(BaseModel): Class for holding information on where templates
        are stored in order to properly handle ThirdPartyParameter objects.
"""

__all__ = [
    "TaskParameters",
    "AnalysisHeader",
    "TemplateConfig",
    "TemplateParameters",
    "ThirdPartyParameters",
]
__author__ = "Gabriel Dorlhiac"

import os
from typing import Dict, Any, Union, Optional

from pydantic import (
    BaseModel,
    BaseSettings,
    PositiveInt,
    Field,
    root_validator,
    validator,
)
from pydantic.dataclasses import dataclass


class AnalysisHeader(BaseModel):
    """Header information for LUTE analysis runs."""

    title: str = Field(
        "LUTE Task Configuration",
        description="Description of the configuration or experiment.",
    )
    experiment: str = Field("", description="Experiment.")
    run: Union[str, int] = Field("", description="Data acquisition run.")
    date: str = Field("1970/01/01", description="Start date of analysis.")
    lute_version: Union[float, str] = Field(
        0.1, description="Version of LUTE used for analysis."
    )
    task_timeout: PositiveInt = Field(
        600,
        description=(
            "Time in seconds until a task times out. Should be slightly shorter"
            " than job timeout if using a job manager (e.g. SLURM)."
        ),
    )
    work_dir: str = Field("", description="Main working directory for LUTE.")

    @validator("work_dir", always=True)
    def validate_work_dir(cls, work_dir: str, values: Dict[str, Any]) -> str:
        if work_dir == "":
            work_dir = (
                f"/sdf/data/lcls/ds/{values['experiment'][:3]}/"
                f"{values['experiment']}/scratch"
            )
        # Check existence and permissions
        if not os.path.exists(work_dir):
            raise ValueError(f"Working Directory: {work_dir} does not exist!")
        if not os.access(work_dir, os.W_OK):
            # Need write access for database, files etc.
            raise ValueError(f"Not write access for working directory: {work_dir}!")
        return work_dir

    @validator("run", always=True)
    def validate_run(
        cls, run: Union[str, int], values: Dict[str, Any]
    ) -> Union[str, int]:
        if run == "":
            # From Airflow RUN_NUM should have Format "RUN_DATETIME" - Num is first part
            run_time: str = os.environ.get("RUN_NUM", "")
            if run_time != "":
                return int(run_time.split("_")[0])
        return run

    @validator("experiment", always=True)
    def validate_experiment(cls, experiment: str, values: Dict[str, Any]) -> str:
        if experiment == "":
            arp_exp: str = os.environ.get("EXPERIMENT", "EXPX00000")
            return arp_exp
        return experiment


class TaskParameters(BaseSettings):
    """Base class for models of task parameters to be validated.

    Parameters are read from a configuration YAML file and validated against
    subclasses of this type in order to ensure that both all parameters are
    present, and that the parameters are of the correct type.

    Note:
        Pydantic is used for data validation. Pydantic does not perform "strict"
        validation by default. Parameter values may be cast to conform with the
        model specified by the subclass definition if it is possible to do so.
        Consider whether this may cause issues (e.g. if a float is cast to an
        int).
    """

    class Config:
        env_prefix = "LUTE_"
        underscore_attrs_are_private: bool = True
        copy_on_model_validation: str = "deep"
        allow_inf_nan: bool = False

    lute_config: AnalysisHeader


@dataclass
class TemplateParameters:
    """Class for representing parameters for third party configuration files.

    These parameters can represent arbitrary data types and are used in
    conjunction with templates for modifying third party configuration files
    from the single LUTE YAML. Due to the storage of arbitrary data types, and
    the use of a template file, a single instance of this class can hold from a
    single template variable to an entire configuration file. The data parsing
    is done by jinja using the complementary template.
    All data is stored in the single model variable `params.`

    The pydantic "dataclass" is used over the BaseModel/Settings to allow
    positional argument instantiation of the `params` Field.
    """

    params: Any


class ThirdPartyParameters(TaskParameters):
    """Base class for third party task parameters.

    Contains special validators for extra arguments and handling of parameters
    used for filling in third party configuration files.
    """

    class Config(TaskParameters.Config):
        """Configuration for parameters model.

        The Config class holds Pydantic configuration. In the case of
        ThirdPartyParameters it also holds some LUTE-specific configuration for
        interpreting command-line arguments and determining TaskResults from the
        set of parameters.

        Attributes:
            extra (str): "allow". Pydantic configuration. Allow (or ignore) extra
                arguments.

            short_flags_use_eq (bool): False. If True, "short" command-line args
                are passed as `-x=arg`.

            long_flags_use_eq (bool): False. If True, "long" command-line args
                are passed as `--long=arg`.

            set_result (bool). True. If True, the model has information about
                setting the TaskResult object from the parameters it contains.
                E.g. it has an `output` parameter which is marked as the result.

            impl_schemas (str). Specifies a the schemas the output/results
                conform to. Only used if set_result is True.
        """

        extra: str = "allow"
        short_flags_use_eq: bool = False
        """Whether short command-line arguments are passed like `-x=arg`."""
        long_flags_use_eq: bool = False
        """Whether long command-line arguments are passed like `--long=arg`."""
        set_result: bool = True  # Could consider moving this to the base Config...
        """Whether the Executor should mark a specified parameter as a result."""
        # result_summary: Optional[str] = None
        # """Format a TaskResult.summary from output."""
        # result_from_params: str = ""
        # """Defines a result from the parameters. Use a validator to do so."""
        impl_schemas: str = ""
        """Schema specification for output result. Will be passed to TaskResult."""

    # lute_template_cfg: TemplateConfig

    @root_validator(pre=False)
    def extra_fields_to_thirdparty(cls, values: Dict[str, Any]):
        for key in values:
            if key not in cls.__fields__:
                values[key] = TemplateParameters(values[key])

        return values


class TemplateConfig(BaseModel):
    """Parameters used for templating of third party configuration files.

    Attributes:
        template_name (str): The name of the template to use. This template must
            live in `config/templates`.

        output_path (str): The FULL path, including filename to write the
            rendered template to.
    """

    template_name: str
    output_path: str
