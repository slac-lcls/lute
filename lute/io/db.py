"""Version agnostic database API module.

This module serves as the import point for functions defined for various versions
of the database API. Functions can be imported from this module instead of worrying
about handling imports from the various API sub-packages.
"""

import importlib
import logging
import os
import warnings
from functools import lru_cache
from types import ModuleType
from typing import Any, Callable, List, Optional

import lute.io._db.common_sqlite as common_sqlite
from lute.execution.logging import get_logger

if __debug__:
    logging.basicConfig(level=logging.DEBUG)
else:
    logging.basicConfig(level=logging.INFO)

logger: logging.Logger = get_logger(__name__, is_task=False)

LUTE_DB_CURRENT_SPEC_VERSION: int = 0x000002
LUTE_DB_DEFAULT_SPEC_VERSION: int = 0x000002

LUTE_DB_SPEC_VERSION: int = int(
    os.getenv("LUTE_DB_SPEC_VERSION", LUTE_DB_DEFAULT_SPEC_VERSION)
)


def lazy_import(func_name: str, api_version: int) -> Callable:
    """Return a lazily loaded version of the function."""

    def wrapper(*args, **kwargs) -> Callable:
        func: Callable = import_function(func_name=func_name, api_version=api_version)
        return func(*args, **kwargs)

    return wrapper


@lru_cache
def import_function(func_name: str, api_version: int) -> Callable:
    """Import a database function from the appropriate API version.

    Args:
        func_name (str): The name of the function to import.

        api_version (int): The API version. Currently either 0x000001 or 0x000002.

    Returns:
        func (Callable): The requested function.

    Raises:
        DatabaseError: Raised if the api_version is not supported.

        AttributeError: Raised if the function requested does not exist.
    """
    if api_version not in (1, 2):
        raise common_sqlite.DatabaseError(
            "Unrecognized database specification version! Set LUTE_DB_SPEC_VERSION appropriately! "
            "Supported versions: 0x000001 and 0x000002"
        )
    api_mod: ModuleType = importlib.import_module(f"lute.io._db.v{api_version}.api")
    try:
        func: Callable = getattr(api_mod, func_name)
    except AttributeError:
        logging.error(
            f"Attempting to retrieve database API non-existent function {func_name}!"
        )
        raise
    return func


def api_unavailable(func_name: str, api_version: int) -> Callable:
    """Raise an error if function was not supported by the API version."""

    def wrapper(*args, **kwargs) -> None:
        raise NotImplementedError(
            f"Function {func_name} not available with DB version {api_version}"
        )

    return wrapper


record_analysis_db: Callable = lazy_import("record_analysis_db", LUTE_DB_SPEC_VERSION)
read_latest_db_entry: Callable = lazy_import(
    "read_latest_db_entry", LUTE_DB_SPEC_VERSION
)

import_or_raise: Callable
if LUTE_DB_SPEC_VERSION == 0x000002:
    import_or_raise = lazy_import
elif LUTE_DB_SPEC_VERSION == 0x000001:
    import_or_raise = api_unavailable
else:
    import_or_raise = api_unavailable

record_parameters_db: Callable = import_or_raise(
    "record_parameters_db", LUTE_DB_SPEC_VERSION
)
update_analysis_db: Callable = import_or_raise(
    "update_analysis_db", LUTE_DB_SPEC_VERSION
)
get_executions_summary: Callable = import_or_raise(
    "get_executions_summary", LUTE_DB_SPEC_VERSION
)
get_task_parameters_summary: Callable = import_or_raise(
    "get_task_parameters_summary", LUTE_DB_SPEC_VERSION
)
get_task_parameters_defn_and_params: Callable = import_or_raise(
    "get_task_parameters_defn_and_params", LUTE_DB_SPEC_VERSION
)


def _gather_run_results(
    runs: List[int],
    selector_desc: str,
    upstream_task_name: str,
    param: str,
    work_dir: str,
) -> List[Any]:
    """Gather one DB result per run in an already-resolved run list.

    Shared body of `gather_tagged_run_results` and
    `gather_sampled_run_results` - the two differ only in how they turn a
    selector (an eLog tag or a sample name) into `runs`, not in what they do
    with them. `selector_desc` is a lowercase human-readable description of
    that selector (e.g. "tag 'Lyso'") used only in warnings/errors, and
    capitalized where it starts a sentence.

    Args:
        runs (List[int]): Run numbers to gather results for.

        selector_desc (str): Lowercase description of the selector that
            produced `runs`, for messages (e.g. "sample 'Thermolysin'").

        upstream_task_name (str): Task class name whose DB results to look
            up (e.g. "ScaleCCTBXXFEL", "IndexCrystFEL").

        param (str): Name of the parameter/result field to retrieve from
            each run's DB entry (e.g. "result.payload", "out_file").

        work_dir (str): LUTE working directory containing the database.

    Returns:
        resolved (List[Any]): The retrieved values, one per run that had a
            valid DB entry, in the order of `runs`.

    Raises:
        ValueError: If none of the runs have a valid DB entry for
            `upstream_task_name`/`param`.
    """
    resolved: List[Any] = []
    for run in runs:
        entry: Optional[Any] = read_latest_db_entry(
            work_dir, upstream_task_name, param, for_run=run
        )
        if entry:
            resolved.append(entry)
        else:
            warnings.warn(
                f"No {upstream_task_name} DB result for run {run} "
                f"({selector_desc}) - excluding from "
                f"{upstream_task_name}.{param} gather."
            )
    if not resolved:
        raise ValueError(
            f"{selector_desc[0].upper()}{selector_desc[1:]} resolved to {runs} "
            f"but none had a valid {upstream_task_name} DB entry for "
            f"'{param}'."
        )
    return resolved


def gather_tagged_run_results(
    experiment: str,
    tag: str,
    upstream_task_name: str,
    param: str,
    work_dir: str,
) -> List[Any]:
    """Resolve every run carrying `tag` in the eLog, and gather one DB result
    each.

    For each run, looks up `param` from `upstream_task_name`'s most recent
    valid DB entry for that run. This is the generic form of the run-gathering
    pattern used to combine results from multiple tagged runs into one
    non-run-dependent submission (see `--tag` on submit_slurm/launch_slurm).
    Runs without a valid DB entry for `upstream_task_name`/`param` are skipped
    with a warning.

    Args:
        experiment (str): Experiment name, used to query the eLog.

        tag (str): eLog tag identifying the set of runs to gather.

        upstream_task_name (str): Task class name whose DB results to look
            up (e.g. "ScaleCCTBXXFEL", "IndexCrystFEL").

        param (str): Name of the parameter/result field to retrieve from
            each run's DB entry (e.g. "result.payload", "out_file").

        work_dir (str): LUTE working directory containing the database.

    Returns:
        resolved (List[Any]): The retrieved values, one per run that had a
            valid DB entry, in the order returned by the eLog for `tag`.

    Raises:
        ValueError: If no runs are found for `tag`, or none of the resolved
            runs have a valid DB entry for `upstream_task_name`/`param`.
    """
    from lute.io.elog import get_elog_runs_by_tag

    runs: List[int] = get_elog_runs_by_tag(experiment, tag)
    if not runs:
        raise ValueError(
            f"No runs found for tag '{tag}' in '{experiment}' - cannot "
            f"gather {upstream_task_name}.{param}."
        )
    return _gather_run_results(
        runs, f"tag '{tag}'", upstream_task_name, param, work_dir
    )


def gather_sampled_run_results(
    experiment: str,
    sample_name: str,
    upstream_task_name: str,
    param: str,
    work_dir: str,
) -> List[Any]:
    """Resolve every run associated with `sample_name` in the eLog, and gather
    one DB result each.

    The sample-selector counterpart of `gather_tagged_run_results` - identical
    in every respect except that runs are resolved via
    `get_elog_runs_by_sample` (the `sample` field on each run document)
    instead of `get_elog_runs_by_tag` (tags on eLog entries). See `--sample`
    on submit_slurm/launch_slurm.

    Args:
        experiment (str): Experiment name, used to query the eLog.

        sample_name (str): eLog sample name identifying the set of runs to
            gather.

        upstream_task_name (str): Task class name whose DB results to look
            up (e.g. "ScaleCCTBXXFEL", "IndexCrystFEL").

        param (str): Name of the parameter/result field to retrieve from
            each run's DB entry (e.g. "result.payload", "out_file").

        work_dir (str): LUTE working directory containing the database.

    Returns:
        resolved (List[Any]): The retrieved values, one per run that had a
            valid DB entry, in the order returned by the eLog for
            `sample_name`.

    Raises:
        ValueError: If no runs are found for `sample_name`, or none of the
            resolved runs have a valid DB entry for
            `upstream_task_name`/`param`.
    """
    from lute.io.elog import get_elog_runs_by_sample

    runs: List[int] = get_elog_runs_by_sample(experiment, sample_name)
    if not runs:
        raise ValueError(
            f"No runs found for sample '{sample_name}' in '{experiment}' - "
            f"cannot gather {upstream_task_name}.{param}."
        )
    return _gather_run_results(
        runs, f"sample '{sample_name}'", upstream_task_name, param, work_dir
    )
