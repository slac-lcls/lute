"""Tasklet functions to be run before or after a Task.

"Tasklets" are simple processing steps which are intended to be run by an
Executor after the main Task has finished. These functions help provide a way to
handle some common operations which may need to be performed, such as examining
an output text file, or doing simple file conversions. These operations can be
also be imported for use in first-party Tasks. In particular for
`ThirdPartyTask`s, it is not easy to append operations to a Task which is why
the Executor mechanism is provided.

Functions:
    git_clone(repo: str, location: str, permissions: str): Clone a git repo.

    concat_files(location: str, in_files_glob: str, out_file: str): Concatenate
        a group of files into a single output file.

    grep(match_str: str, in_file: str) -> str | List[str]: grep for text in a
        specific file. Returns the results.

    indexamajig_summary_indexing_rate(stream_file: str) -> Dict[str, str]: Parse
        an output stream file to determine indexed patterns/indexing rate.

    compare_hkl_fom_summary(shell_file: str, figure_display_name: str) ->
        Tuple[Dict[str, str], Optional[ElogSummaryPlots]]: Extract the figure of
        merit and produce a plot of figure of merit/resolution ring.

Usage:
    As tasklets are just functions they can be imported and used within Task
    code normally if needed.

    However, tasklets can also be managed through the Executor in a similar way
    to environment changes. E.g., to add a tasklet to an Executor instance one
    would:

    # First create Executor instance as normal to run the Task
    MyTaskRunner: Executor = Executor("RunMyTask")
    #MyTaskRunner.update_environment(...) # if needed
    MyTaskRunner.add_tasklet(
        tasklet, args, when="before", set_result=False, set_summary=False
    )

    A special substitution syntax can be used in args if specific values
    from a `TaskParameters` object will be needed to run the Tasklet:
        args=("{{ param_to_sub }}", ...)
"""

__all__ = [
    "grep",
    "git_clone",
    "indexamajig_summary_indexing_rate",
    "compare_hkl_fom_summary",
]
__author__ = "Gabriel Dorlhiac"

import logging
import os
import subprocess
from typing import List, Dict, Tuple, Optional

from lute.tasks.dataclasses import ElogSummaryPlots


if __debug__:
    logging.basicConfig(level=logging.DEBUG)
else:
    logging.basicConfig(level=logging.INFO)

logger: logging.Logger = logging.getLogger(__name__)


def concat_files(location: str, in_files_glob: str, out_file: str) -> None:
    """Concatenate a series of files into a single output file.

    Args:
        location (str): Path to the files to concatenate.

        in_files_glob (str): A glob to match a series of files at the specified
            path. These will all be concatenated.

        out_file (str): Name of the concatenated output.
    """
    import shutil
    from pathlib import Path
    from typing import BinaryIO

    in_file_path: Path = Path(f"{location}")
    in_file_list: List[Path] = list(in_file_path.rglob(f"{in_files_glob}"))

    wf: BinaryIO
    with open(out_file, "wb") as wf:
        for in_file in in_file_list:
            rf: BinaryIO
            with open(in_file, "rb") as rf:
                shutil.copyfileobj(rf, wf)


def git_clone(repo: str, location: str, permissions: str) -> None:
    """Clone a git repository.

    Will not overwrite a directory of there is already a folder at the specified
    location.

    Args:
        repo (str): Name of the repository to clone. Should be specified as:
            "<user_or_organization>/<repository_name>"

        location (str): Path to the location to clone to.

        permissions (str): Permissions to set on the repository.
    """
    repo_only: str = repo.split("/")[1]
    if os.path.exists(f"location/{repo_only}"):
        logger.debug(
            f"Repository {repo} already exists at {location}. Will not overwrite."
        )
        return
    cmd: List[str] = ["git", "clone", f"https://github.com/{repo}.git", location]
    out: str
    out, _ = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True
    ).communicate()


def grep(match_str: str, in_file: str) -> List[str]:
    """Grep for specific lines of text output.

    Args:
        match_str (str): String to search for.

        in_file (str): File to search.

    Returns:
        lines (List[str]): The matches. It may be a list with just an empty
            string if nothing is found.
    """
    cmd: List[str] = ["grep", match_str, in_file]
    out: str
    out, _ = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True
    ).communicate()

    lines: List[str] = out.split("\n")
    return lines


def indexamajig_summary_indexing_rate(stream_file: str) -> Dict[str, str]:
    """Return indexing rate from indexamajig output.

    Args:
        stream_file (str): Input stream file.
    """
    res: List[str] = grep("Cell parameters", stream_file)
    n_indexed: int
    if res:
        n_indexed = len(res[:-1])
    else:
        n_indexed = 0
    res = grep("End chunk", stream_file)
    n_hits: int
    rate: float
    if res:
        n_hits = len(res[:-1])
        rate = n_indexed / n_hits
    else:
        n_hits = 0
        rate = 0
    return {
        "Number of lattices indexed": str(n_indexed),
        "Indexing rate": f"{rate:.2f}",
    }


def compare_hkl_fom_summary(
    shell_file: str, figure_display_name: str
) -> Tuple[Dict[str, str], Optional[ElogSummaryPlots]]:
    """Analyze information produced by CrystFEL's compare_hkl.

    Extracts figures of merit information and produces text summary and plots.

    Args:
        shell_file (str): Path to output `shell-file` containing FOM information.

        figure_display_name (str): Display name of the figure in the eLog.
    """
    import numpy as np
    import holoviews as hv
    import panel as pn

    with open(shell_file, "r") as f:
        lines: List[str] = f.readlines()

    header: str = lines[0]
    fom: str = header.split()[2]
    shells_arr: np.ndarray[np.float64] = np.loadtxt(lines[1:])
    run_params: Dict[str, str] = {fom: str(shells_arr[1])}
    if shells_arr.ndim == 1:
        return run_params, None

    hv.extension("bokeh")
    pn.extension()
    xdim: hv.core.dimension.Dimension = hv.Dimension(
        ("Resolution (A)", "Resolution (A)")
    )
    ydim: hv.core.dimension.Dimension = hv.Dimension((fom, fom))

    angs_bins: np.ndarray[np.float64] = 10.0 / shells_arr[:, 0]
    pts: hv.Points = hv.Points((angs_bins, shells_arr[:, 1]), kdims=[xdim, ydim])
    grid: pn.GridSpec = pn.GridSpec(name="Figures of Merit")
    grid[:2, :2] = pts
    tabs = pn.Tabs(grid)
    return run_params, ElogSummaryPlots(figure_display_name, tabs)
