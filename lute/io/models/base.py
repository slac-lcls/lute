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
from typing import (
    Set,
    Dict,
    Any,
    Union,
    Optional,
    ClassVar,
    no_type_check,
    cast,
)

import pydantic
from pydantic import BaseModel, PositiveInt, PrivateAttr
from pydantic import Field as PydanticField
from pydantic.dataclasses import dataclass

PYDANTIC_V2 = True if pydantic.__version__[0] == "2" else False

if PYDANTIC_V2:
    # Ignore mypy and ruff for now since type checking against pydantic 1.10
    from pydantic import model_validator, field_validator  # type: ignore
    from pydantic_core import PydanticUndefined  # type: ignore
    from pydantic_settings import SettingsConfigDict, BaseSettings  # type: ignore

    @no_type_check  # This function causes many headaches with mypy... Ignore
    def Field(
        default: "Any" = PydanticUndefined,
        *,
        default_factory: "Callable[[], Any] | Callable[[dict[str, Any]], Any] | None" = PydanticUndefined,  # noqa: F821
        alias: "str | None" = PydanticUndefined,
        alias_priority: "int | None" = PydanticUndefined,
        validation_alias: "str | AliasPath | AliasChoices | None" = PydanticUndefined,  # noqa: F821
        serialization_alias: "str | None" = PydanticUndefined,
        title: "str | None" = PydanticUndefined,
        field_title_generator: "Callable[[str, FieldInfo], str] | None" = PydanticUndefined,  # noqa: F821
        description: "str | None" = PydanticUndefined,
        examples: "list[Any] | None" = PydanticUndefined,
        exclude: "bool | None" = PydanticUndefined,
        discriminator: "str | types.Discriminator | None" = PydanticUndefined,  # noqa: F821
        deprecated: "Deprecated | str | bool | None" = PydanticUndefined,  # noqa: F821
        json_schema_extra: "JsonDict | Callable[[JsonDict], None] | None" = PydanticUndefined,  # noqa: F821
        frozen: "bool | None" = PydanticUndefined,
        validate_default: "bool | None" = PydanticUndefined,
        repr: "bool" = PydanticUndefined,
        init: "bool | None" = PydanticUndefined,
        init_var: "bool | None" = PydanticUndefined,
        kw_only: "bool | None" = PydanticUndefined,
        pattern: "str | typing.Pattern[str] | None" = PydanticUndefined,  # noqa: F821
        strict: "bool | None" = PydanticUndefined,
        coerce_numbers_to_str: "bool | None" = PydanticUndefined,
        gt: "annotated_types.SupportsGt | None" = PydanticUndefined,  # noqa: F821
        ge: "annotated_types.SupportsGe | None" = PydanticUndefined,  # noqa: F821
        lt: "annotated_types.SupportsLt | None" = PydanticUndefined,  # noqa: F821
        le: "annotated_types.SupportsLe | None" = PydanticUndefined,  # noqa: F821
        multiple_of: "float | None" = PydanticUndefined,
        allow_inf_nan: "bool | None" = PydanticUndefined,
        max_digits: "int | None" = PydanticUndefined,
        decimal_places: "int | None" = PydanticUndefined,
        min_length: "int | None" = PydanticUndefined,
        max_length: "int | None" = PydanticUndefined,
        union_mode: "Literal['smart', 'left_to_right']" = PydanticUndefined,  # noqa: F821
        fail_fast: "bool | None" = PydanticUndefined,
        **extra: "Unpack[_EmptyKwargs]",  # noqa: F821
    ) -> "Any":
        return PydanticField(
            default=default,
            default_factory=default_factory,
            alias=alias,
            alias_priority=alias_priority,
            validation_alias=validation_alias,
            serialization_alias=serialization_alias,
            title=title,
            field_title_generator=field_title_generator,
            description=description,
            examples=examples,
            exclude=exclude,
            discriminator=discriminator,
            deprecated=deprecated,
            json_schema_extra=extra,  ## Changed vs v1 and will be removed in v3
            frozen=frozen,
            validate_default=validate_default,
            repr=repr,
            init=init,
            init_var=init_var,
            kw_only=kw_only,
            pattern=pattern,
            strict=strict,
            coerce_numbers_to_str=coerce_numbers_to_str,
            gt=gt,
            ge=ge,
            lt=lt,
            le=le,
            multiple_of=multiple_of,
            allow_inf_nan=allow_inf_nan,
            max_digits=max_digits,
            decimal_places=decimal_places,
            min_length=min_length,
            max_length=max_length,
            union_mode=union_mode,
            fail_fast=fail_fast,
        )

else:
    from pydantic import root_validator, validator
    from pydantic import BaseSettings  # type: ignore[no-redef]

    Field = PydanticField

LUTE_PARAMETER_CONFIG_KEYS: Set[str] = {
    "run_directory",
    "set_result",
    "result_from_params",
    "result_summary",
    "impl_schemas",
    "short_flags_use_eq",
    "long_flags_use_eq",
}


class AnalysisHeader(BaseModel):
    """Header information for LUTE analysis runs."""

    title: str = Field(
        "LUTE Task Configuration",
        description="Description of the configuration or experiment.",
    )
    experiment: str = (
        Field("", description="Experiment.")
        if PYDANTIC_V2
        else Field("", description="Experiment.")
    )
    run: Union[str, int] = (
        Field("", description="Data acquisition run.", validate_default=True)
        if PYDANTIC_V2
        else Field("", description="Data acquisition run.")
    )
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
    work_dir: str = (
        Field("", description="Main working directory for LUTE.", validate_default=True)
        if PYDANTIC_V2
        else Field("", description="Main working directory for LUTE.")
    )

    if PYDANTIC_V2:
        work_dir_validator: ClassVar = field_validator("work_dir")
        run_validator: ClassVar = field_validator("run")
        experiment_validator: ClassVar = field_validator("experiment")
    else:
        work_dir_validator = validator("work_dir", always=True)
        run_validator = validator("run", always=True)
        experiment_validator = validator("experiment", always=True)

    @work_dir_validator
    @classmethod
    def validate_work_dir(cls, directory: str, values: Dict[str, Any]) -> str:
        work_dir: str
        if directory == "":
            std_work_dir = (
                f"/sdf/data/lcls/ds/{values['experiment'][:3]}/"
                f"{values['experiment']}/scratch"
            )
            work_dir = std_work_dir
        else:
            work_dir = directory
        # Check existence and permissions
        if not os.path.exists(work_dir):
            raise ValueError(f"Working Directory: {work_dir} does not exist!")
        if not os.access(work_dir, os.W_OK):
            # Need write access for database, files etc.
            raise ValueError(f"Not write access for working directory: {work_dir}!")
        os.environ["LUTE_WORK_DIR"] = work_dir
        return work_dir

    @run_validator
    @classmethod
    def validate_run(
        cls, run: Union[str, int], values: Dict[str, Any]
    ) -> Union[str, int]:
        if run == "":
            # From Airflow RUN_NUM should have Format "RUN_DATETIME" - Num is first part
            run_time: str = os.environ.get("RUN_NUM", "")
            if run_time != "":
                return int(run_time.split("_")[0])
        return run

    @experiment_validator
    @classmethod
    def validate_experiment(cls, experiment: str, values: Dict[str, Any]) -> str:
        if experiment == "":
            arp_exp: str = os.environ.get("EXPERIMENT", "EXPX00000")
            return arp_exp
        return experiment


class TaskParametersConfig(dict):
    """Configuration for parameters model.

    The Config class holds Pydantic configuration. A number of LUTE-specific
    configuration has also been placed here.

    Attributes:
        env_prefix (str): Pydantic configuration. Will set parameters from
            environment variables containing this prefix. E.g. a model
            parameter `input` can be set with an environment variable:
            `{env_prefix}input`, in LUTE's case `LUTE_input`.

        underscore_attrs_are_private (bool): Pydantic configuration. Whether
            to hide attributes (parameters) prefixed with an underscore. This
            is present in Pydantic v1.10 only!

        copy_on_model_validation (str): Pydantic configuration. How to copy
            the input object passed to the class instance for model
            validation. Set to perform a deep copy. This is present in
            Pydantic v1.10 only!

        allow_inf_nan (bool): Pydantic configuration. Whether to allow
            infinity or NAN in float fields.

        run_directory (Optional[str]): None. If set, it should be a valid
            path. The `Task` will be run from this directory. This may be
            useful for some `Task`s which rely on searching the working
            directory.

        set_result (bool). False. If True, the model has information about
            setting the TaskResult object from the parameters it contains.
            E.g. it has an `output` parameter which is marked as the result.
            The result can be set with a field value of `is_result=True` on
            a specific parameter, or using `result_from_params` and a
            validator.

        result_from_params (Optional[str]): None. Optionally used to define
            results from information available in the model using a custom
            validator. E.g. use a `outdir` and `filename` field to set
            `result_from_params=f"{outdir}/{filename}`, etc. Only used if
            `set_result==True`

        result_summary (Optional[str]): None. Defines a result summary that
            can be known after processing the Pydantic model. Use of summary
            depends on the Executor running the Task. All summaries are
            stored in the database, however. Only used if `set_result==True`

        impl_schemas (Optional[str]). Specifies a the schemas the
            output/results conform to. Only used if `set_result==True`.
    """

    env_prefix = "LUTE_"
    if not PYDANTIC_V2:
        underscore_attrs_are_private: bool = True
        copy_on_model_validation: str = "deep"
    allow_inf_nan: bool = False

    run_directory: Optional[str] = None
    """Set the directory that the Task is run from."""
    set_result: bool = False
    """Whether the Executor should mark a specified parameter as a result."""
    result_from_params: Optional[str] = None
    """Defines a result from the parameters. Use a validator to do so."""
    result_summary: Optional[str] = None
    """Format a TaskResult.summary from output."""
    impl_schemas: Optional[str] = None
    """Schema specification for output result. Will be passed to TaskResult."""


class ThirdPartyParametersConfig(TaskParametersConfig):
    """Configuration for parameters model.

    The Config class holds Pydantic configuration and inherited configuration
    from the base `TaskParameters.Config` class. A number of values are also
    overridden, and there are some specific configuration options to
    ThirdPartyParameters. A full list of options (with TaskParameters options
    repeated) is described below.

    Attributes:
        env_prefix (str): Pydantic configuration. Will set parameters from
            environment variables containing this prefix. E.g. a model
            parameter `input` can be set with an environment variable:
            `{env_prefix}input`, in LUTE's case `LUTE_input`.

        underscore_attrs_are_private (bool): Pydantic configuration. Whether
            to hide attributes (parameters) prefixed with an underscore.

        copy_on_model_validation (str): Pydantic configuration. How to copy
            the input object passed to the class instance for model
            validation. Set to perform a deep copy.

        allow_inf_nan (bool): Pydantic configuration. Whether to allow
            infinity or NAN in float fields.

        run_directory (Optional[str]): None. If set, it should be a valid
            path. The `Task` will be run from this directory. This may be
            useful for some `Task`s which rely on searching the working
            directory.

        set_result (bool). True. If True, the model has information about
            setting the TaskResult object from the parameters it contains.
            E.g. it has an `output` parameter which is marked as the result.
            The result can be set with a field value of `is_result=True` on
            a specific parameter, or using `result_from_params` and a
            validator.

        result_from_params (Optional[str]): None. Optionally used to define
            results from information available in the model using a custom
            validator. E.g. use a `outdir` and `filename` field to set
            `result_from_params=f"{outdir}/{filename}`, etc.

        result_summary (Optional[str]): None. Defines a result summary that
            can be known after processing the Pydantic model. Use of summary
            depends on the Executor running the Task. All summaries are
            stored in the database, however.

        impl_schemas (Optional[str]). Specifies a the schemas the
            output/results conform to. Only used if set_result is True.

        -----------------------
        ThirdPartyTask-specific:

        extra (str): "allow". Pydantic configuration. Allow (or ignore) extra
            arguments.

        short_flags_use_eq (bool): False. If True, "short" command-line args
            are passed as `-x=arg`. ThirdPartyTask-specific.

        long_flags_use_eq (bool): False. If True, "long" command-line args
            are passed as `--long=arg`. ThirdPartyTask-specific.
    """

    extra: str = "allow"
    short_flags_use_eq: bool = False
    """Whether short command-line arguments are passed like `-x=arg`."""
    long_flags_use_eq: bool = False
    """Whether long command-line arguments are passed like `--long=arg`."""
    set_result: bool = True
    """Whether the Executor should mark a specified parameter as a result."""


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

    if PYDANTIC_V2:
        model_config = SettingsConfigDict(
            **{
                key: val
                for key, val in TaskParametersConfig()
                if key not in LUTE_PARAMETER_CONFIG_KEYS
            }
        )
        Config: ClassVar = TaskParametersConfig()
    else:
        Config = TaskParametersConfig

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

    if PYDANTIC_V2:
        model_config = SettingsConfigDict(
            **{
                key: val
                for key, val in ThirdPartyParametersConfig()
                if key not in LUTE_PARAMETER_CONFIG_KEYS
            }
        )
        Config: ClassVar = ThirdPartyParametersConfig()
    else:
        Config = ThirdPartyParametersConfig

    _unknown_template_params: Dict[str, Any] = PrivateAttr()
    # lute_template_cfg: TemplateConfig

    if PYDANTIC_V2:
        extra_fields_validator: ClassVar = model_validator(mode="after")
    else:
        # Strictly only need pre=False for running, but it doesn't match overload
        # variants so mypy complains when using pydantic v2. This is functionally
        # the same for our purposes
        extra_fields_validator = root_validator(
            pre=False, skip_on_failure=True, allow_reuse=True
        )

    @extra_fields_validator
    @classmethod
    def extra_fields_to_thirdparty(cls, values: Dict[str, Any]):
        cls._unknown_template_params = {}
        param_schema_template: Dict[str, Any] = {
            "title": "",
            "description": "Unknown template parameters.",
            "type": "object",
            "properties": {
                "params": "",
                "type": "object",
            },
        }
        new_values: Dict[str, Any] = {}
        fields: Dict[str, Any]
        if PYDANTIC_V2:
            fields = cls.model_fields
        else:
            # For pydantic v2 mypy reports cls.__fields__ as callable it is dict
            # for both versions of pydantic (deprecated in v2)
            fields = cast(dict, cls.__fields__)
        for key in values:
            if key not in fields:
                new_values[key] = TemplateParameters(params=values[key])
                param_schema: Dict[str, Any] = param_schema_template.copy()
                param_schema["title"] = key
                param_schema["properties"]["params"] = values[key]
                cls._unknown_template_params[key] = param_schema
            else:
                new_values[key] = values[key]
        return new_values


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
