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

import os
import warnings
from abc import ABC
from typing import List, Dict, Iterator, Dict, Any, Union, Optional

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
