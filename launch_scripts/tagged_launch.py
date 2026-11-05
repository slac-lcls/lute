"""Selector-driven multi-run workflow submission.

Entry points used by `launch_maestro.py::main()` when `--tag` or `--sample`
is passed instead of (or alongside) `-r`/`--run`. Resolves every run matching
the given selector - an eLog tag (`run_tagged_workflow`) or an eLog sample
name (`run_sampled_workflow`) - splits the requested workflow DAG into a
run-dependent subgraph and a non-run-dependent subgraph (see
`dag_partition.py`), submits the run-dependent subgraph once per resolved
run, and - once every one of those has completed successfully - submits the
non-run-dependent subgraph exactly once, with `TAG`/`SAMPLE` (whichever
selector was used) and `EXPERIMENT` exported into its environment for that
subgraph's own Tasks to use (e.g. `MergeCCTBXXFELParameters.phil_
parameters.tag`/`.sample`, see `lute/io/models/sfx_merge.py`).

The two selectors differ only in how a selector value becomes a run list;
everything downstream - DAG partitioning, per-run submission, the barrier
before the non-run-dependent stage - is shared verbatim via
`_run_selected_workflow`.

Deliberately re-invokes the real, unmodified `launch_slurm` binary as a
subprocess for every stage/run rather than calling `load_lute_dag`/
`run_workflow` repeatedly in-process: the DAG execution engine is a compiled
extension (`maestro._maestro._maestro`) whose safety under repeated
in-process invocation within one Python process was not verified, whereas
re-running the already-tested single-run entry point as a fresh OS process
per invocation carries no such risk.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)


class TaggedLaunchError(Exception):
    """Raised when a --tag or --sample workflow submission cannot proceed."""


def _common_args(args: argparse.Namespace, extra_args: List[str]) -> List[str]:
    """Rebuild the CLI args every child `launch_slurm` invocation should get,
    minus -W/-r/--tag/--sample (each stage supplies its own -W; -r varies per
    run for the dependent stage and is fixed to a representative run for the
    non-dependent stage; --tag/--sample are intentionally NOT forwarded, so the
    child process takes the normal single-run code path)."""
    common: List[str] = ["-c", args.config]
    if args.experiment:
        common += ["-e", args.experiment]
    if args.type:
        common += ["--type", args.type]
    if args.debug:
        common += ["-d"]
    if getattr(args, "num_server_threads", None):
        common += ["--num_server_threads", str(args.num_server_threads)]
    if getattr(args, "unbuffered", False):
        common += ["--unbuffered"]
    common += extra_args
    return common


def _submit_stage(launch_slurm_bin: str, dag_path: str, run: int, common: List[str]) -> None:
    cmd = [launch_slurm_bin, "-W", dag_path, "-r", str(run), *common]
    logger.info("Submitting tagged stage for run %s: %s", run, " ".join(cmd))
    subprocess.run(cmd, check=True)


def _run_selected_workflow(
    args: argparse.Namespace,
    extra_args: List[str],
    bin_dir: str,
    lute_location: str,
    *,
    selector_flag: str,
    selector_kind: str,
    selector_value: str,
    env_var: str,
    resolver: Callable[[str, str], List[int]],
) -> None:
    """Submit a workflow against every run matching a selector.

    Shared implementation of `run_tagged_workflow` and
    `run_sampled_workflow`. The only per-selector inputs are the CLI flag name
    (`selector_flag`, for messages), a human-readable noun (`selector_kind`,
    e.g. "tag"), the selector's value, the environment variable the
    non-run-dependent stage should see it in, and the eLog resolver that turns
    (experiment, selector_value) into a run list.
    """
    experiment: Optional[str] = os.getenv("EXPERIMENT") or args.experiment
    if not experiment:
        raise TaggedLaunchError(
            f"{selector_flag} requires -e/--experiment (or the EXPERIMENT env "
            f"var) to resolve which runs carry the {selector_kind}."
        )

    runs: List[int] = sorted(resolver(experiment, selector_value))
    if not runs:
        raise TaggedLaunchError(
            f"No runs found for {selector_kind} '{selector_value}' in "
            f"experiment '{experiment}'. (Note: this is indistinguishable "
            f"today from an eLog auth/network failure - {resolver.__name__} "
            "collapses both cases to an empty list; check your Kerberos "
            f"ticket / the {selector_kind} spelling.)"
        )
    logger.info(
        "%s '%s' resolved to runs: %s",
        selector_kind.capitalize(),
        selector_value,
        runs,
    )

    from launch_scripts.dag_partition import partition_workflow_yaml

    dep_yaml, non_dep_yaml = partition_workflow_yaml(args.workflow_defn)

    tmp_dir = tempfile.mkdtemp(
        prefix=f"lute_{selector_kind}_",
        dir=os.path.dirname(os.path.abspath(args.workflow_defn)),
    )
    logger.info(
        "Writing partitioned DAG(s) for this %s submission to %s",
        selector_flag,
        tmp_dir,
    )

    launch_slurm_bin = f"{bin_dir}/launch_slurm"
    common = _common_args(args, extra_args)

    if dep_yaml is not None:
        dep_path = os.path.join(tmp_dir, "run_dependent.dag")
        with open(dep_path, "w") as f:
            f.write(dep_yaml)

        max_workers = args.max_concurrent_runs or len(runs)
        logger.info(
            "Submitting run-dependent stage for %d run(s), up to %d concurrently",
            len(runs),
            max_workers,
        )
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                run: pool.submit(_submit_stage, launch_slurm_bin, dep_path, run, common)
                for run in runs
            }
            failed: List[int] = []
            for run, future in futures.items():
                try:
                    future.result()
                except subprocess.CalledProcessError as e:
                    logger.error("Run-dependent stage failed for run %s: %s", run, e)
                    failed.append(run)
        if failed:
            raise TaggedLaunchError(
                f"Run-dependent stage failed for runs {sorted(failed)} - not "
                "submitting the non-run-dependent stage on top of incomplete data."
            )

    if non_dep_yaml is not None:
        non_dep_path = os.path.join(tmp_dir, "non_run_dependent.dag")
        with open(non_dep_path, "w") as f:
            f.write(non_dep_yaml)

        os.environ[env_var] = selector_value
        os.environ["EXPERIMENT"] = experiment
        representative_run = runs[0]
        logger.info(
            "Submitting non-run-dependent stage once (%s=%s, representative run=%s)",
            env_var,
            selector_value,
            representative_run,
        )
        _submit_stage(launch_slurm_bin, non_dep_path, representative_run, common)

    logger.info("%s submission for '%s' complete.", selector_flag, selector_value)


def run_tagged_workflow(
    args: argparse.Namespace,
    extra_args: List[str],
    bin_dir: str,
    lute_location: str,
) -> None:
    """Submit a workflow against every run carrying `args.tag` in the eLog."""
    from lute.io.elog import get_elog_runs_by_tag

    _run_selected_workflow(
        args,
        extra_args,
        bin_dir,
        lute_location,
        selector_flag="--tag",
        selector_kind="tag",
        selector_value=args.tag,
        env_var="TAG",
        resolver=get_elog_runs_by_tag,
    )


def run_sampled_workflow(
    args: argparse.Namespace,
    extra_args: List[str],
    bin_dir: str,
    lute_location: str,
) -> None:
    """Submit a workflow against every run associated with `args.sample` in the
    eLog.

    Identical to `run_tagged_workflow` in every respect except the selector:
    runs come from the `sample` field on each run document rather than from
    tags on eLog entries.
    """
    from lute.io.elog import get_elog_runs_by_sample

    _run_selected_workflow(
        args,
        extra_args,
        bin_dir,
        lute_location,
        selector_flag="--sample",
        selector_kind="sample",
        selector_value=args.sample,
        env_var="SAMPLE",
        resolver=get_elog_runs_by_sample,
    )
