"""Script to setup LUTE and workflow definitions."""

__author__ = "Gabriel Dorlhiac"

import argparse
import logging
import os
import requests
import shutil
import subprocess
import sys
from typing import List, Dict, Any, Optional

from krtc import KerberosTicket


logging.basicConfig(level=logging.INFO)
logger: logging.Logger = logging.getLogger(__name__)


def _run_subprocess_log(cmd: List[str], env: Optional[Dict[str, str]] = None) -> None:
    """Run a subprocess with logging."""
    global logger

    out: str
    err: str
    out, err = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        env=env,
    ).communicate()
    if out:
        logger.info(out)
    if err:
        logger.info(err)


def touch(full_path: str) -> None:
    """Touch a file.

    Args:
        full_path (str): The full path to the file to touch.
    """
    cmd: List[str] = ["touch", full_path]
    _run_subprocess_log(cmd)


def database_setup(full_path: str) -> None:
    """Touch a file.

    Args:
        full_path (str): The full path to the file to touch.
    """
    if os.path.exists(full_path):
        logger.error(
            f"LUTE database already exists at: {full_path}. Will not overwrite, exiting."
        )
        sys.exit(-1)
    touch(full_path)
    os.chmod(full_path, 0o664)


def git_clone(repo: str, location: str, tag: str) -> None:
    """Clone a git repository.

    Will not overwrite a directory of there is already a folder at the specified
    location.
    Args:
        repo (str): Name of the repository to clone. Should be specified as:
            "<user_or_organization>/<repository_name>"

        location (str): Path to the location to clone to.
    """
    global logger

    repo_only: str = repo.split("/")[1]
    if os.path.exists(f"{location}/{repo_only}"):
        logger.debug(
            f"Repository {repo} already exists at {location}. Will not overwrite."
        )
        return
    cmd: List[str] = [
        "git",
        "clone",
        f"https://github.com/{repo}.git",
        f"{location}/{repo_only}",
    ]
    _run_subprocess_log(cmd)

    cwd: str = os.getcwd()
    os.chdir(f"{location}/{repo_only}")
    cmd = ["git", "checkout", tag]
    _run_subprocess_log(cmd)
    os.chdir(cwd)


def pip_install(src_dir: str, install_dir: Optional[str] = None) -> None:
    """Install from a source directory to an optionally specified directory.

    If no prefix (`install_dir`) is provided, this will install to the current
    packages directory determined by, e.g., conda environment, etc.

    If a prefix is provided this command will also create the directory (or
    multiple directories) if necessary.

    Args:
        src_dir (str): Directory with the source code and setup.py.

        install_dir (Optional[str]): Optionally provide a directory to install
            install to.
    """
    cmd: List[str] = [
        "pip",
        "install",
        "--no-deps",
        src_dir,
        f'--prefix="{install_dir}"',
    ]
    logging.info(f"Attempting to install from: {src_dir} to: {install_dir}")
    env: Dict[str, str] = os.environ.copy()
    env["PATH"] = (
        f"/sdf/group/lcls/ds/ana/sw/conda1/inst/envs/ana-4.0.63-py3/bin:{env['PATH']}"
    )
    if install_dir is not None:
        cmd.append(f"--prefix={install_dir}")
        if not os.path.exists(install_dir):
            mkdir_cmd: List[str] = ["mkdir", "-p", install_dir]
            _run_subprocess_log(mkdir_cmd)
    _run_subprocess_log(cmd, env)


def inplace_sed(in_file: str, pattern: str) -> None:
    """Perform an in-place operation on a file using sed.

    Args:
        in_file (str): Path to the file to perform the substitution on.

        pattern (str): Operation. E.g. substitute with "s/old_text/new_text/g"
    """
    cmd: List[str] = ["sed", "-i", pattern, in_file]
    _run_subprocess_log(cmd)


def modify_permissions(lute_path: str):
    """Recursively set permissions for a LUTE installation."""
    os.chmod(lute_path, 0o755)
    for root, dirs, files in os.walk(lute_path):
        for d in dirs:
            os.chmod(os.path.join(root, d), 0o755)

        for f in files:
            os.chmod(os.path.join(root, f), 0o755)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="setup_lute",
        description="Setup LUTE work space and eLog workflows for an experiment.",
        epilog="Refer to https://github.com/slac-lcls/lute for more information.",
    )
    parser.add_argument(
        "-d", "--debug", action="store_true", help="Turn on verbose logging."
    )
    parser.add_argument(
        "-e",
        "--experiment",
        type=str,
        help="Experiment to perform setup for.",
        required=True,
    )
    parser.add_argument(
        "-f",
        "--fresh_install",
        help=(
            "Install a new version of LUTE in the experiment folder. This allows "
            "for local modifications of code. Otherwise, the central installation "
            "will be used which cannot be modified."
        ),
        action="store_true",
    )
    parser.add_argument(
        "--test", help="Use test Airflow instance.", action="store_true"
    )
    parser.add_argument(
        "-v",
        "--version",
        type=str,
        help=(
            "Version of LUTE to use. Corresponds to release tag or `dev`. "
            "Defaults to `dev`."
        ),
        default="dev",
    )
    parser.add_argument(
        "-w",
        "--workflow",
        type=str,
        help=("Which analysis workflow to run. Defaults to smd_summaries."),
        default="smd_summaries",
    )
    args: argparse.Namespace
    extra_args: List[str]  # May have additional SLURM arguments
    args, extra_args = parser.parse_known_args()

    hutch: str = args.experiment[:3]

    results_dir: str = f"/sdf/data/lcls/ds/{hutch}/{args.experiment}/results"
    lute_path: str
    arp_executable: str
    launch_executable: str
    std_hutch_config: str
    std_test_config: str
    if args.fresh_install:
        lute_path = f"{results_dir}/lute"
        # pip_install will also create directories at lute_path
        pip_install(os.path.realpath(f"{__file__}/../.."), lute_path)
        modify_permissions(lute_path)
        arp_executable = f"{lute_path}/bin/submit_launch_airflow.sh"
        launch_executable = f"{lute_path}/bin/launch_airflow.py"
        py_ver: str = f"python{sys.version_info.major}.{sys.version_info.minor}"
        std_hutch_config = f"{lute_path}/lib/{py_ver}/site-packages/config/{hutch}.yaml"
        std_test_config = f"{lute_path}/lib/{py_ver}/site-packages/config/test.yaml"
    else:
        lute_path = f"/sdf/group/lcls/ds/tools/lute/{args.version}/lute"
        arp_executable = f"{lute_path}/launch_scripts/submit_launch_airflow.sh"
        launch_executable = f"{lute_path}/launch_scripts/launch_airflow.py"
        std_hutch_config = f"{lute_path}/config/{hutch}.yaml"
        std_test_config = f"{lute_path}/config/test.yaml"

    lute_output_dir: str = f"{results_dir}/lute_output"
    if not os.path.exists(lute_output_dir):
        os.makedirs(lute_output_dir, mode=0o777)
        os.chmod(lute_output_dir, 0o777)
    config_path: str = f"{lute_output_dir}/{hutch}_lute.yaml"
    if os.path.exists(config_path):
        logger.error(
            f"Configuration YAML already exists at: {config_path}. Will not overwrite, exiting."
        )
        sys.exit(-1)
    if not os.path.exists(std_hutch_config):
        shutil.copy(std_test_config, config_path)
    else:
        shutil.copy(std_hutch_config, config_path)
    os.chmod(config_path, 0o666)
    # Substitute the work_dir in LUTE's config to the experiment results folder.
    sed_pattern: str = f's|work_dir:\(.*\)|work_dir: \\"{lute_output_dir}\\"|g'
    inplace_sed(config_path, sed_pattern)

    database_setup(f"{lute_output_dir}/lute.db")  # Setup permissions on database
    param_string: str = f"{launch_executable} -c {config_path} -w {args.workflow}"

    if args.debug:
        param_string = f"{param_string} --debug"
    if args.test:
        param_string = f"{param_string} --test"

    extra_args_str: str = " ".join(extra_args)
    # Check for partition, account and ntasks. ntasks has defaults by workflow
    if "partition" not in extra_args_str:
        logger.warning(
            "No queue/partition provided. Defaulting to milano. Any key to continue. "
            "Ctrl-C to exit."
        )
        try:
            _: str = input()
            extra_args_str = f"{extra_args_str} --partition=milano"
        except KeyboardInterrupt:
            logger.info("Exiting.")
            sys.exit(0)
    if "account" not in extra_args_str:
        account: str = f"lcls:{args.experiment}"
        logger.warning(
            f"No account provided. Defaulting to {account}. Any key to continue. "
            "Ctrl-C to exit."
        )
        try:
            _: str = input()
            extra_args_str = f"{extra_args_str} --account={account}"
        except KeyboardInterrupt:
            logger.info("Exiting.")
            sys.exit(0)
    if "ntasks" not in extra_args_str:
        ncores: int
        if args.workflow in ("smd_xas", "smd_xss", "test"):
            ncores = 2
        elif args.workflow in ("smd_summaries", "smd_xes"):
            ncores = 5
        else:
            ncores = 120
        logger.warning(
            f"No tasks/cores provided. Defaulting to {ncores}. Any key to continue. "
            "Ctrl-C to exit."
        )
        try:
            _: str = input()
            extra_args_str = f"{extra_args_str} --ntasks={ncores}"
        except KeyboardInterrupt:
            logger.info("Exiting.")
            sys.exit(0)

    param_string = f"{param_string} {extra_args_str}"

    main_workflow: Dict[str, str]
    if args.workflow in ("smd_summaries", "smd_xss", "smd_xes", "smd_xss"):
        main_workflow = {
            "name": "lute_smd_summaries",
            "executable": arp_executable,
            "trigger": "RUN_PARAM_IS_VALUE",
            "run_param_name": "SmallData",
            "run_param_value": "done",
            "location": "S3DF",
            "parameters": param_string,
        }
    elif 0:
        # Replace eventually with workflows which use START_OF_RUN
        main_workflow = {
            "name": f"lute_{args.workflow}",
            "executable": arp_executable,
            "trigger": "START_OF_RUN",
            "location": "S3DF",
            "parameters": param_string,
        }
    else:
        main_workflow = {
            "name": f"lute_{args.workflow}",
            "executable": arp_executable,
            "trigger": "END_OF_RUN",
            "location": "S3DF",
            "parameters": param_string,
        }

    workflows: List[Dict[str, str]] = []
    workflows.append(main_workflow)
    # Will want to append additional auxiliary workflows eventually

    for workflow in workflows:
        logger.info(
            f"Creating eLog workflow named {workflow['name']} with parameters: {workflow['parameters']}"
        )
        krbticket: Any = KerberosTicket("HTTP@pswww.slac.stanford.edu")
        krbheaders: dict = krbticket.getAuthHeaders()
        url: str = (
            f"https://pswww.slac.stanford.edu/ws-kerb/lgbk/lgbk/{args.experiment}/ws"
            "/create_update_workflow_def"
        )
        post_params: Dict[str, Any] = {
            "url": url,
            "headers": krbheaders,
            "json": workflow,
        }
        resp: requests.models.Response = requests.post(**post_params)
        resp.raise_for_status()
        # Extra logging and such...


if __name__ == "__main__":
    main()
