"""Machinary for the IO of configuration YAML files and their validation.

Functions:
    parse_config(taskname: str, config_path: str) -> TaskParameters: Parse a
        configuration file and return a TaskParameters object of validated
        parameters for a specific Task. Raises an exception if the provided
        configuration does not match the expected model.

Exceptions:
    ValidationError: Error raised by pydantic during data validation. (From
        Pydantic)
"""

__all__ = ["parse_config"]
__author__ = "Gabriel Dorlhiac"

import re
import warnings
from typing import List, Dict, Iterator, Dict, Any

import yaml
import yaml
from pydantic import (
    BaseModel,
    BaseSettings,
    HttpUrl,
    PositiveInt,
    NonNegativeInt,
    Field,
    conint,
    root_validator,
    validator,
)
from pydantic.dataclasses import dataclass

from .models import *


def substitute_variables(
    config: Dict[str, Any], curr_key: Optional[str] = None
) -> None:
    """Performs variable substitutions on a dictionary read from config YAML file.

    Can be used to define input parameters in terms of other input parameters.
    This is similar to functionality employed by validators for parameters in
    the specific Task models, but is intended to be more accessible to users.
    Variable substitutions are defined using a minimal syntax from Jinja:
                               {{ experiment }}
    defines a substitution of the variable `experiment`. The characters `{{ }}`
    can be escaped if the literal symbols are needed in place.

    For example, a path to a file can be defined in terms of experiment and run
    values in the config file:
        MyTask:
          experiment: myexp
          run: 2
          special_file: /path/to/{{ experiment }}/{{ run }}/file.inp

    Acceptable variables for substitutions are values defined elsewhere in the
    YAML file. Environment variables can also be used if prefaced with a `$`
    character. E.g. to get the experiment from an environment variable:
        MyTask:
          run: 2
          special_file: /path/to/{{ $EXPERIMENT }}/{{ run }}/file.inp

    Args:
        config (Dict[str, Any]):  A dictionary of parsed configuration.

        curr_key (Optional[str]): Used to keep track of recursion level when scanning
            through iterable items in the config dictionary.

    Returns:
        subbed_config (Dict[str, Any]): The config dictionary after substitutions
            have been made. May be identical to the input if no substitutions are
            needed.
    """
    _sub_pattern = "\{\{.*\}\}"
    iterable: Dict[str, Any]
    if curr_key is not None:
        # Need to handle nested levels by interpreting curr_key
        iterable = config[curr_key]
    else:
        iterable = config
    for param, value in iterable.items():
        if isinstance(value, dict):
            substitute_variables(config, curr_key=param)
        elif isinstance(value, list):
            ...
        # Scalars str - we skip numeric types
        elif isinstance(value, str):
            matches: List[str] = re.findall(_sub_pattern, value)
            for m in matches:
                key_to_sub: str = m[2:-2].strip()
                sub: str
                if key_to_sub[0] == "$":
                    sub = os.environ.get(key_to_sub[1:], "")
                else:
                    sub = config[key_to_sub]
                pattern: str = m.replace("{{", "\{\{").replace("}}", "\}\}")
                iterable[param] = re.sub(pattern, sub, value)


def parse_config(task_name: str = "test", config_path: str = "") -> TaskParameters:
    """Parse a configuration file and validate the contents.

    Args:
        task_name (str): Name of the specific task that will be run.

        config_path (str): Path to the configuration file.

    Returns:
        params (TaskParameters): A TaskParameters object of validated
            task-specific parameters. Parameters are accessed with "dot"
            notation. E.g. `params.param1`.

    Raises:
        ValidationError: Raised if there are problems with the configuration
            file. Passed through from Pydantic.
    """
    task_config_name: str = f"{task_name}Parameters"

    with open(config_path, "r") as f:
        docs: Iterator[Dict[str, Any]] = yaml.load_all(stream=f, Loader=yaml.FullLoader)
        header: Dict[str, Any] = next(docs)
        config: Dict[str, Any] = next(docs)

    lute_config: Dict[str, AnalysisHeader] = {"lute_config": AnalysisHeader(**header)}
    try:
        task_config: Dict[str, Any] = dict(config[task_name])
        lute_config.update(task_config)
    except KeyError as err:
        warnings.warn(
            (
                f"{task_name} has no parameter definitions in YAML file."
                " Attempting default parameter initialization."
            )
        )
    parsed_parameters: TaskParameters = globals()[task_config_name](**lute_config)

    return parsed_parameters
