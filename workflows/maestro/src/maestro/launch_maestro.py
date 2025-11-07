"""Launch a LUTE workflow using Maestro as the workflow manager."""

__all__ = []
__author__ = "Gabriel Dorlhiac"

import argparse
import logging
import os
import sys
from typing import List, Optional

from maestro._maestro import _maestro
from maestro.parser import load_lute_dag

logger: logging.Logger = logging.getLogger("PyMaestro")
handler: logging.Handler = logging.StreamHandler()
formatter: logging.Formatter = logging.Formatter(logging.BASIC_FORMAT)
handler.setFormatter(formatter)
logger.addHandler(handler)

def main():
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        prog="launch_maestro",
        description="A light-weight workflow manager which executes LUTE Managed Tasks.",
        epilog="Refer to https://github.com/slac-lcls/lute for more information.",
    )
    # We pop out the optional args for changing the order of the help message.
    # We'll add it back later
    optional_args: argparse._ArgumentGroup = parser._action_groups.pop()

    # Required arguments
    required_args: argparse._ArgumentGroup = parser.add_argument_group(
        "required arguments"
    )
    required_args.add_argument(
        "-c", "--config", type=str, help="Path to config YAML file.", required=True
    )
    required_args.add_argument(
        "-W",
        "--workflow_defn",
        type=str,
        help="Path to a YAML file with workflow.",
        required=True,
    )

    # Arguments required for when running from command-line
    non_arp_required_args: argparse._ArgumentGroup = parser.add_argument_group(
        "required arguments when running without the ARP"
    )
    non_arp_required_args.add_argument(
        "-e",
        "--experiment",
        type=str,
        help="Provide an experiment if not running with ARP.",
        required=False,
    )
    non_arp_required_args.add_argument(
        "-r",
        "--run",
        type=str,
        help="Provide a run number if not running with ARP.",
        required=False,
    )

    # Optional Arguments
    optional_args.add_argument(
        "-d", "--debug", help="Run in debug mode.", action="store_true"
    )
    optional_args.add_argument(
        "--unbuffered",
        help=(
            "Flush logs immediately. Warning: This can make output confusing "
            "when running multiple managed Tasks are running in parallel."
        ),
        action="store_true",
    )
    parser._action_groups.append(optional_args)

    args: argparse.Namespace
    extra_args: List[str]  # Should contain all SLURM arguments!
    args, extra_args = parser.parse_known_args()

    # Do we use any APIs now that need kerberos ticket if not using JID?
    # Maybe can get rid of this for the SLURM only submission?
    cache_file: Optional[str] = os.getenv("KRB5CCNAME")
    if os.getenv("EXPERIMENT") is None or os.getenv("RUN_NUM") is None:
        if cache_file is None:
            logger.info("No Kerberos cache. Try running `kinit` and resubmitting.")
            sys.exit(-1)

        if args.experiment is None or args.run is None:
            logger.info(
                (
                    "You must provide a `-e ${EXPERIMENT}` and `-r ${RUN_NUM}` "
                    "if not running with the ARP!\n"
                    "If you submitted this from the eLog and are seeing this error "
                    "please contact the maintainers."
                )
            )
            sys.exit(-1)
        os.environ["EXPERIMENT"] = args.experiment
        os.environ["RUN_NUM"] = args.run

    lute_location: str = os.path.abspath(f"{os.path.dirname(__file__)}/..")

    wf_defn: List[_maestro.JobStep] = load_lute_dag(
        workflow_path=args.workflow_defn,
        lute_location=lute_location,
        config_file=args.config,
        debug=args.debug,
        default_slurm_params=" ".join(extra_args),
    )
    _maestro.run_workflow(wf_defn)


if __name__ == "__main__":
    main()
