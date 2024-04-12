import sys
import argparse
import logging
from typing import Dict, Optional, List, Set, Tuple
from typing_extensions import TypedDict

import pprint

import lute.io.models
from lute.io.models.base import TaskParameters


class PropertyDict(TypedDict):
    default: str
    description: str
    title: str
    type: Optional[str]
    anyOf: Optional[List[Dict[str, str]]]  # Either an anyOf or type per property
    # Generally only for BinaryTasks
    rename_param: Optional[str]
    flag_type: Optional[str]
    # Other additional field attributes
    env_names: Optional[Set[str]]
    maxLength: Optional[int]
    minLength: Optional[int]
    format: Optional[str]
    exclusiveMinimum: Optional[int]
    minimum: Optional[int]


class ObjectDefintion(TypedDict):
    description: str
    properties: Dict[str, PropertyDict]
    title: str
    type: str


class ModelSchema(TypedDict):
    definitions: Optional[Dict[str, ObjectDefintion]]
    description: str
    properties: Dict[str, PropertyDict]
    required: Optional[List[str]]
    title: str
    type: str


logging.basicConfig(level=logging.INFO)
logger: logging.Logger = logging.getLogger(__name__)

parser: argparse.ArgumentParser = argparse.ArgumentParser(
    prog="Task parameters help utility.",
    description="Display parameter descriptions and types for a specified Task.",
    epilog="Refer to https://github.com/slac-lcls/lute for more information.",
)
parser.add_argument("-T", "--Task", type=str, help="Name of the Task to inspect.")
parser.add_argument(
    "--full_schema",
    action="store_true",
    help="Dump an unformated full model schema. Has more information.",
)


def _format_parameter_row(param: str, param_description: PropertyDict) -> str:
    """Take a property dictionary for a parameter and format it for printing."""
    typeinfo: str
    if "type" in param_description:
        typeinfo = param_description["type"]
    elif "anyOf" in param_description:  # anyOf is present instead
        typeinfo = " | ".join(_["type"] for _ in param_description["anyOf"])
    else:
        typeinfo = "No type information"
    typeinfo = f"({typeinfo})"

    msg: str = f"{param} {typeinfo}"
    default: str
    if "default" in param_description:
        default = param_description["default"]
        msg = f"{msg} - Default: {default}"

    description: str
    if "description" in param_description:
        description = param_description["description"]
    else:
        description = "Unknown description."

    msg = f"{msg}\n\t{description}\n\n"
    return msg


if __name__ == "__main__":
    args: argparse.Namespace = parser.parse_args()
    task_name: str = args.Task
    model_name: str = f"{task_name}Parameters"

    if hasattr(lute.io.models, model_name):
        parameter_model: TaskParameters = getattr(lute.io.models, model_name)
        logger.info(f"Fetching parameter information for {task_name}.")
    else:
        logger.info(f"No Task named {task_name} found! Exiting!")
        sys.exit(-1)

    # For types need to check for key `type` or a list of dicts `anyOf=[{'type': ...}, {'type': ...}]`
    parameter_schema: ModelSchema = parameter_model.schema()

    if args.full_schema:
        pprint.pprint(parameter_schema)
        sys.exit(0)

    task_description: str = parameter_schema["description"]
    required_parameters: Optional[List[Tuple[str, PropertyDict]]] = None
    if (
        "required" in parameter_schema.keys()
        and parameter_schema["required"] is not None
    ):
        required_parameters = [
            (param, parameter_schema["properties"][param])
            for param in parameter_schema["required"]
        ]

    out_msg: str = f"{task_name}\n{'-'*len(task_name)}\n"
    out_msg = f"{out_msg}{task_description}\n\n\n"
    if required_parameters is not None:
        out_msg = f"{out_msg}Required Parameters:\n--------------------\n"
        for param in required_parameters:
            out_msg = f"{out_msg}{_format_parameter_row(param[0], param[1])}"
        out_msg = f"{out_msg}\n\n"

    out_msg = f"{out_msg}All Parameters:\n-------------\n"
    for param in parameter_schema["properties"]:
        out_msg = f"{out_msg}{_format_parameter_row(param, parameter_schema['properties'][param])}"

    print(out_msg)
