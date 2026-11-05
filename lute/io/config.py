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

# flake8: noqa: F403,F405

__all__ = ["parse_config"]
__author__ = "Gabriel Dorlhiac"

import copy
import os
import re
import warnings
from typing import Any, ClassVar, Dict, Iterator, List, Optional, Tuple, Union

import pprint
import yaml

from lute.io.db import read_latest_db_entry
from lute.io.models import *
from lute.execution.debug_utils import LUTE_DEBUG_EXIT


def _isfloat(string: str) -> bool:
    try:
        float(string)
        return True
    except ValueError:
        return False


def _check_str_numeric(string: str) -> Union[str, int, float]:
    """Check if a string is an integer or float and return it as such.

    Args:
        string (str): Input string to check.

    Returns:
        output (str | int | float): Returns an int or float if the string can be
            converted to one. Otherwise returns the original string.
    """
    if string.isnumeric():
        return int(string)
    elif _isfloat(string):
        return float(string)
    else:
        return string


def _run_group_specificity(
    current_run: int, applies_to: Dict[str, Any]
) -> Optional[Tuple[int, int]]:
    """Determine whether, and how specifically, a run matches a group's rule.

    Lower values are more specific. A match via an explicit `runs` list always
    beats a match via a `range`.

    NOTE: Within definitions of each type (i.e., a range or list of runs), the more
    specific definition wins. I.e., a shorter list of runs, or narrower range.

    Args:
        current_run (int): The run to check.

        applies_to (Dict[str, Any]): The group's `applies_to` rule. May contain
            `runs` (a list of runs) and/or `range` (an inclusive [min, max]).

    Returns:
        specificity (Optional[Tuple[int, int]]): `None` if the run does not match.
            `(0, len(runs))` for a list match, `(1, r_max - r_min)` for a range
            match.

    Raises:
        ValueError: If `range` is not a two-element sequence.
    """
    runs: Optional[List[int]] = applies_to.get("runs")
    if runs is not None and current_run in runs:
        return (0, len(runs))

    run_range: Optional[List[int]] = applies_to.get("range")
    if run_range is not None:
        if len(run_range) != 2:
            raise ValueError(f"RUN_GROUPS `range` must be [min, max], got: {run_range}")
        r_min: int
        r_max: int
        r_min, r_max = sorted(int(r) for r in run_range)
        if r_min <= current_run <= r_max:
            return (1, r_max - r_min)

    return None


_MISSING: Any = object()


def _get_nested(data: Any, path: List[str]) -> Any:
    """Return the value a `path` for nested dicts, or the _MISSING sentinel."""
    current: Any = data
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return _MISSING
        current = current[key]
    return current


def _set_nested(data: Dict[str, Any], path: List[str], value: Any) -> None:
    """Set the value at `path` in a set of nested dicts.

    NOTE: Any intermediate dictionaries are created as necessary.
    """
    current: Dict[str, Any] = data
    for key in path[:-1]:
        if not isinstance(current.get(key), dict):
            current[key] = {}
        current = current[key]
    current[path[-1]] = value


def resolve_run_group_directives(
    task_name: str,
    task_config: Dict[str, Any],
    global_groups: Dict[str, Any],
    work_dir: str,
    current_run: Optional[Union[int, str]],
) -> Dict[str, Any]:
    """Resolves `@` directives to look up parameters based on matching RUN_GROUPS rules.

    Possible directives include the following three options:
    1. `@db:current`: Use the parameter value from the last valid database entry
      for the experiment run of the current execution.
    2. `@db:run:NNNN`: Use the last valid database entry for the specified run `NNNN`.
    3. `@literal`: Use the literal value included in the YAML file.

    Args:
        task_name (str): The current Task to process configuration for.

        task_config (Dict[str, Any]): The configuration YAML data.

        global_groups (Dict[str, Any]): Globally defined RUN GROUPS from the YAML.

        work_dir (str): The LUTE working directory.

        current_run (Optional[Union[int, str]]): The current run.

    Returns:
        updated_config (Dict[str, Any]): Parameter configuration after directive
            resolution.
    """
    # Always need to remove RUN_GROUPS so its not processed by pydantic
    task_groups: Dict[str, Any] = task_config.pop("RUN_GROUPS", None) or {}
    global_groups = global_groups or {}

    # This shouldn't happen, but if the run is not set, just leave YAML unmodified.
    if current_run is None:
        return task_config
    try:
        current_run_int: int = int(current_run)
    except (ValueError, TypeError):
        return task_config

    if not global_groups and not task_groups:
        return task_config

    # Task-level listed first so they win over global
    all_group_names: List[str] = list(task_groups.keys()) + [
        g for g in global_groups if g not in task_groups
    ]

    # Construct candidate definitions and policies in order of specificity
    # - Task groups trump global groups (ranges and policies, including default)
    # - More specific ranges/lists trump less-specific ranges
    candidates: List[Tuple[Tuple[int, int], int, str, Dict[str, Any]]] = []
    for order, group_name in enumerate(all_group_names):
        task_grp: Dict[str, Any] = task_groups.get(group_name) or {}
        global_grp: Dict[str, Any] = global_groups.get(group_name) or {}

        # A task-level redifnition of the range/list is not recommended but
        # always overrides the global one
        applies_to: Dict[str, Any] = task_grp.get(
            "applies_to", global_grp.get("applies_to", {})
        )
        if not applies_to:
            continue
        specificity: Optional[Tuple[int, int]] = _run_group_specificity(
            current_run_int, applies_to
        )
        if specificity is not None:
            candidates.append(
                (specificity, order, group_name, {**global_grp, **task_grp})
            )

    if not candidates:
        return task_config

    # Always use most specific group
    # If there are still ties, then definition order is used
    candidates.sort(key=lambda candidate: (candidate[0], candidate[1]))
    best_specificity: Tuple[int, int]
    active_name: str
    active_policy: Dict[str, Any]
    best_specificity, _, active_name, active_policy = candidates[0]
    tied: List[str] = [
        candidate[2] for candidate in candidates if candidate[0] == best_specificity
    ]
    if len(tied) > 1:
        warnings.warn(
            f"Run {current_run_int} matches RUN_GROUPS {tied} with equal specificity. "
            f"Using '{active_name}'."
        )

    default_directive: str = active_policy.get("default", "@literal")
    # Map parameters to the directives handling nesting.
    # Nesting is done with dot-delimiters (like elsewhere)
    targets: Dict[str, str] = {name: default_directive for name in task_config}
    # Directive *should* be a string. But we double check below
    directive: str
    for key, directive in active_policy.items():
        if key not in ("applies_to", "default"):
            targets[key] = directive

    # Keep the values so when @literal is used, the values can be applied
    yaml_literals: Dict[str, Any] = copy.deepcopy(task_config)
    # Handle the parameter sweep suffix
    db_task_name: str = re.sub(r"_\d+$", "", task_name)
    db_cache: Dict[Tuple[str, int], Any] = {}

    for param_path in sorted(targets, key=lambda p: p.count(".")):
        directive = targets[param_path]
        path: List[str] = param_path.split(".")
        literal_val: Any = _get_nested(yaml_literals, path)

        if not isinstance(directive, str) or not directive.startswith("@db:"):
            # @literal -- restore the YAML value (no-op unless a parent was replaced)
            if literal_val is not _MISSING and len(path) > 1:
                _set_nested(task_config, path, copy.deepcopy(literal_val))

            continue

        target_run: Optional[int] = None
        if directive == "@db:current":
            target_run = current_run_int
        elif directive.startswith("@db:run:"):
            try:
                target_run = int(directive.split(":")[-1])
            except ValueError:
                warnings.warn(f"Invalid run directive format: '{directive}'")
                continue
        if target_run is None:
            warnings.warn(f"Unrecognized directive: '{directive}'")
            continue

        # Walk down the path to get the requested nested value
        cache_key: Tuple[str, int] = (path[0], target_run)
        if cache_key not in db_cache:
            db_cache[cache_key] = read_latest_db_entry(
                db_dir=work_dir,
                task_name=db_task_name,
                param=path[0],
                for_run=target_run,
            )
        db_val: Any = _get_nested({path[0]: db_cache[cache_key]}, path)

        if db_val is not _MISSING and db_val is not None:
            _set_nested(task_config, path, copy.deepcopy(db_val))
        else:
            fallback: str = (
                "<not set in YAML>" if literal_val is _MISSING else f"{literal_val}"
            )
            warnings.warn(
                f"No DB entry found for {task_name}.{param_path} (Run: {target_run}). "
                f"Fallbacking back to the YAML literal: {fallback}."
            )

    return task_config


def substitute_variables(
    header: Dict[str, Any], config: Dict[str, Any], curr_key: Optional[str] = None
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
    _sub_pattern = r"\{\{[^}{]*\}\}"
    iterable: Dict[str, Any] = config
    if curr_key is not None:
        # Need to handle nested levels by interpreting curr_key
        keys_by_level: List[str] = curr_key.split(".")
        for key in keys_by_level:
            iterable = iterable[key]
    else:
        ...
        # iterable = config
    for param, value in iterable.items():
        if isinstance(value, dict):
            new_key: str
            if curr_key is None:
                new_key = param
            else:
                new_key = f"{curr_key}.{param}"
            substitute_variables(header, config, curr_key=new_key)
        elif isinstance(value, list):
            for idx, item in enumerate(value):
                if not isinstance(item, str):
                    continue
                matches_list: List[str] = re.findall(_sub_pattern, item)
                for m in matches_list:
                    item_key_to_sub_maybe_with_fmt = m[2:-2].strip().split(":")
                    item_key_to_sub = item_key_to_sub_maybe_with_fmt[0]
                    item_fmt = None
                    if len(item_key_to_sub_maybe_with_fmt) == 2:
                        item_fmt = item_key_to_sub_maybe_with_fmt[1]
                    item_sub: Any
                    if item_key_to_sub[0] == "$":
                        item_sub = os.getenv(item_key_to_sub[1:], None)
                        if item_sub is None:
                            # Check if we use a different env - substitution
                            # happens before environment reset
                            item_sub = os.getenv(f"LUTE_TENV_{item_key_to_sub[1:]}")
                        if item_sub is None:
                            print(
                                f"Environment variable {item_key_to_sub[1:]} not found! Cannot substitute in YAML config!",
                                flush=True,
                            )
                            continue
                        # substitutions from env vars will be strings, so
                        # convert back to numeric for formatting (e.g. {var:04d})
                        item_sub = _check_str_numeric(item_sub)
                    else:
                        try:
                            item_sub = config
                            for key in item_key_to_sub.split("."):
                                item_sub = item_sub[key]
                        except KeyError:
                            item_sub = header[item_key_to_sub]
                    item_pattern = (
                        m.replace("{{", r"\{\{")
                        .replace("}}", r"\}\}")
                        .replace("$", r"\$")
                    )
                    if item_fmt is not None:
                        item_sub = f"{item_sub:{item_fmt}}"
                    else:
                        item_sub = f"{item_sub}"
                    item = re.sub(item_pattern, item_sub, item)
                iterable[param][idx] = _check_str_numeric(item)
        # Scalars str - we skip numeric types
        elif isinstance(value, str):
            matches: List[str] = re.findall(_sub_pattern, value)
            for m in matches:
                key_to_sub_maybe_with_fmt: List[str] = m[2:-2].strip().split(":")
                key_to_sub: str = key_to_sub_maybe_with_fmt[0]
                fmt: Optional[str] = None
                if len(key_to_sub_maybe_with_fmt) == 2:
                    fmt = key_to_sub_maybe_with_fmt[1]
                sub: Any
                if key_to_sub[0] == "$":
                    sub = os.getenv(key_to_sub[1:], None)
                    if sub is None:
                        # Check if we use a different env - substitution happens
                        # before environment reset
                        sub = os.getenv(f"LUTE_TENV_{key_to_sub[1:]}")
                    if sub is None:
                        print(
                            f"Environment variable {key_to_sub[1:]} not found! Cannot substitute in YAML config!",
                            flush=True,
                        )
                        continue
                    # substitutions from env vars will be strings, so convert back
                    # to numeric in order to perform formatting later on (e.g. {var:04d})
                    sub = _check_str_numeric(sub)
                else:
                    try:
                        sub = config
                        for key in key_to_sub.split("."):
                            sub = sub[key]
                    except KeyError:
                        sub = header[key_to_sub]
                pattern: str = (
                    m.replace("{{", r"\{\{").replace("}}", r"\}\}").replace("$", r"\$")
                )
                if fmt is not None:
                    sub = f"{sub:{fmt}}"
                else:
                    sub = f"{sub}"
                iterable[param] = re.sub(pattern, sub, iterable[param])
            # Reconvert back to numeric values if needed...
            iterable[param] = _check_str_numeric(iterable[param])


class RunGroupsPolicyLoader(yaml.SafeLoader):
    _RUN_GROUP_DIRECTIVES: ClassVar[Tuple[str, ...]] = (
        "@db:run:",
        "@db:current",
        "@literal",
    )
    """The set of prefixes used by the RUN_GROUPS policy adapters."""

    def check_plain(self):
        """Allow presence of unquoted directives used by RUN_GROUPS.

        Must include support for:
        - @db:run:NNN
        - @db:current
        - @literal
        """
        if self.peek() == "@":
            return any(
                self.prefix(len(directive)) == directive
                for directive in RunGroupsPolicyLoader._RUN_GROUP_DIRECTIVES
            )

        return super().check_plain()


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
    cleaned_task_name: str = re.sub(r"_\d+$", "", task_name)
    task_config_name: str = f"{cleaned_task_name}Parameters"

    with open(config_path, "r") as f:
        docs: Iterator[Dict[str, Any]] = yaml.load_all(
            stream=f, Loader=RunGroupsPolicyLoader
        )
        header: Dict[str, Any] = next(docs)
        config: Dict[str, Any] = next(docs)

    substitute_variables(header, header)
    substitute_variables(header, config)

    LUTE_DEBUG_EXIT("LUTE_DEBUG_EXIT_AT_YAML", pprint.pformat(config))

    # Check for global RUN_GROUPS
    global_run_groups: Dict[str, Any] = header.pop("RUN_GROUPS", {})

    lute_config: Dict[str, AnalysisHeader] = {"lute_config": AnalysisHeader(**header)}

    try:
        task_config: Dict[str, Any] = dict(config[task_name])

        work_dir: str = header.get("work_dir", "")
        current_run: Optional[Union[str, int]] = header.get("run") or os.getenv("RUN")

        task_config = resolve_run_group_directives(
            task_name=task_name,
            task_config=task_config,
            global_groups=global_run_groups,
            work_dir=work_dir,
            current_run=current_run,
        )
        lute_config.update(task_config)
    except KeyError:
        warnings.warn(
            (
                f"{task_name} has no parameter definitions in YAML file."
                " Attempting default parameter initialization."
            )
        )
    parsed_parameters: TaskParameters = globals()[task_config_name](**lute_config)

    # Determine and record Task version information dynamically
    version_specifier: Optional[int] = getattr(
        parsed_parameters.Config, "version_specifier", None
    )
    if version_specifier is not None and version_specifier > 0:
        import json

        from lute.io.version_utils import record_version

        location: Optional[str] = getattr(
            parsed_parameters.Config, "version_location", None
        )
        diff_args: Optional[List[str]] = getattr(
            parsed_parameters.Config, "version_diff_args", None
        )
        version_info: Dict[str, str] = record_version(
            version_specifier=version_specifier, location=location, diff_args=diff_args
        )

        # Store where version information was taken from
        version_info["version-location"] = json.dumps(location)
        version_info["version-diff-args"] = json.dumps(diff_args)

        if version_info:
            parsed_parameters.Config.task_version = json.dumps(version_info)

    return parsed_parameters
