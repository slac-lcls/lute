import sys
import argparse
import logging
from typing import Dict, Optional, List, Set, Tuple, Any, Callable
from typing_extensions import TypedDict

import pprint

import lute.io.models
from lute.io.models.base import TaskParameters
from lute import managed_tasks


class PropertyDict(TypedDict):
    default: str
    description: str
    title: str
    type: Optional[str]
    anyOf: Optional[List[Dict[str, str]]]  # Either an anyOf or type per property
    # Generally only for ThirdPartyTasks
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
parser.add_argument("-l", "--list", action="store_true", help="List out all Tasks")
parser.add_argument(
    "-T", "--Task", type=str, help="Name of the Task to inspect.", required=False
)
parser.add_argument(
    "--full_schema",
    action="store_true",
    help="Dump an unformated full model schema. Has more information.",
)


def _format_parameter_row(
    param: str,
    param_description: PropertyDict,
    validators: Optional[List[Callable]] = None,
) -> str:
    """Take a property dictionary for a parameter and format it for printing."""
    typeinfo: str
    if "type" in param_description:
        typeinfo = param_description["type"]
    elif "anyOf" in param_description:  # anyOf is present instead
        typeinfo = " | ".join(_["type"] for _ in param_description["anyOf"])
    elif "allOf" in param_description and "$ref" in param_description["allOf"][0]:
        typeinfo = param_description["allOf"][0]["$ref"].split("/")[-1]
    else:
        typeinfo = "No type information"
    typeinfo = f"({typeinfo})"

    msg: str = f"{param} {typeinfo}"
    default: str
    if "default" in param_description:
        default = param_description["default"]
        if default == "":
            default = "<Empty String> - May be populated by validator"
        msg = f"{msg} - Default: {default}"

    description: str
    if "description" in param_description:
        description = param_description["description"]
    else:
        description = "Unknown description."

    msg = f"{msg}\n\t{description}"
    if validators is not None:
        msg = f"{msg}\n\tValidators:"
        for validator in validators:
            msg = f"{msg}\n\t\t- {validator.func.__name__}"
    msg = f"{msg}\n\n"
    return msg


if __name__ == "__main__":
    args: argparse.Namespace = parser.parse_args()
    if args.list:
        logger.info(f"Fetching Task list.")
        # Construct Task <-> Executor mapping
        # [task_name, [managed_task1, managed_task2, ...]
        managed_task_map: Dict[str, List[str]] = {}
        for key in dir(managed_tasks):
            obj: Any = getattr(managed_tasks, key)
            if isinstance(obj, managed_tasks.BaseExecutor):
                task_name: str = obj._analysis_desc.task_result.task_name
                if task_name in managed_task_map:
                    managed_task_map[task_name].append(key)
                else:
                    managed_task_map[task_name] = [key]
        task_list_msg: str = "Task List"
        task_list_msg = f"{task_list_msg}\n{'-'*len(task_list_msg)}"
        for key in dir(lute.io.models):
            if "Parameters" in key and key not in (
                "ThirdPartyParameters",
                "TaskParameters",
                "TemplateParameters",
            ):
                task_name: str = key.replace("Parameters", "")
                task_list_msg = f"{task_list_msg}\n\n- {task_name}\n"
                task_list_msg = f"{task_list_msg}  Current Managed Tasks:"
                if task_name in managed_task_map:
                    for mgd_task in managed_task_map[task_name]:
                        task_list_msg = f"{task_list_msg} {mgd_task},"
                obj: TaskParameters = getattr(lute.io.models, key)
                parameter_schema: ModelSchema = obj.schema()
                description: str = parameter_schema["description"]
                for line in description.split("\n"):
                    new_line: str
                    if line:
                        new_line = f"\n\t{line}"
                    else:
                        new_line = f"\n{line}"
                    task_list_msg = f"{task_list_msg}{new_line}"

        print(task_list_msg)

    task_name: str = args.Task
    if task_name:
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

        validators: Optional[List[Callable]] = None
        out_msg: str = f"{task_name}\n{'-'*len(task_name)}\n"
        out_msg = f"{out_msg}{task_description}\n\n\n"
        if required_parameters is not None:
            out_msg = f"{out_msg}Required Parameters:\n--------------------\n"
            for param in required_parameters:
                validators = (
                    parameter_model.__validators__[param[0]]
                    if param[0] in parameter_model.__validators__
                    else None
                )
                out_msg = (
                    f"{out_msg}{_format_parameter_row(param[0], param[1], validators)}"
                )
            out_msg = f"{out_msg}\n\n"

        out_msg = f"{out_msg}All Parameters:\n---------------\n"
        for param in parameter_schema["properties"]:
            validators = (
                parameter_model.__validators__[param]
                if param in parameter_model.__validators__
                else None
            )
            out_msg = f"{out_msg}{_format_parameter_row(param, parameter_schema['properties'][param], validators)}"

        if "definitions" in parameter_schema and parameter_schema["definitions"]:
            definitions: List[str] = [
                defn
                for defn in parameter_schema["definitions"]
                if defn not in ("AnalysisHeader", "TemplateConfig")
            ]
            if len(definitions) > 0:
                out_msg = f"{out_msg}Template Parameters:\n--------------------\n"
                for defn in definitions:
                    out_msg = f"{out_msg}{defn}:\n"
                    for param in parameter_schema["definitions"][defn]["properties"]:
                        row: str = _format_parameter_row(
                            param,
                            parameter_schema["definitions"][defn]["properties"][param],
                        )
                        out_msg = f"{out_msg}{row}"
        print(out_msg)
